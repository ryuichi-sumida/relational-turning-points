#!/usr/bin/env python3
"""
Ablation study: crash/surge detection & forecasting across 7 feature-set combinations.

Feature sets (S=Session, T=Temporal, P=Person-normalized):
  S, T, P, ST, SP, TP, STP

For each: crash + surge × 5 constructs + systemic × EN + XGB with LOPO CV.

Tasks:
  - detection: features from session t predict event at transition t-1→t
  - forecast:  features from session t predict event at transition t→t+1

Usage:
  # Single condition (detection, default)
  python exp_ablation_detection.py single --feature_set SP --event crash --construct familiarity

  # Single condition (forecasting)
  python exp_ablation_detection.py single --task forecast --feature_set SP --event crash --construct familiarity

  # All conditions on this machine (8 parallel jobs)
  python exp_ablation_detection.py all --n_parallel 8

  # Distributed forecasting: worker 0 of 5
  python exp_ablation_detection.py all --task forecast --worker_id 0 --n_workers 5 --n_parallel 10

  # Summarize results
  python exp_ablation_detection.py summarize
  python exp_ablation_detection.py summarize --task forecast
"""

import argparse
import json
import logging
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product as iterproduct
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "full_features_with_temporals.csv"
RESULTS_DIR_DETECTION = BASE / "crash_surge" / "results" / "ablation"
RESULTS_DIR_FORECAST = BASE / "crash_surge" / "results" / "ablation_forecast"

def get_results_dir(task: str) -> Path:
    return RESULTS_DIR_FORECAST if task == "forecast" else RESULTS_DIR_DETECTION

# ─── Constants ───────────────────────────────────────────────
CONSTRUCTS = [
    "familiarity", "social_penetration", "memory",
    "conversational", "enjoyment",
]
TARGETS = CONSTRUCTS + ["systemic"]
TEMPORAL_PREFIXES = (
    "delta_", "max_prior_", "EMA_", "trend_", "ever_", "cumulative_",
)
NON_FEATURE_COLS = {"pid", "session_num", "n_prior_sessions"} | set(CONSTRUCTS)

FEATURE_SETS = ["S", "T", "P", "ST", "SP", "TP", "STP"]
FEATURE_SET_LABELS = {
    "S": "Session", "T": "Temporal", "P": "Person-Norm.",
    "ST": "Session+Temporal", "SP": "Session+Person-Norm.",
    "TP": "Temporal+Person-Norm.", "STP": "All",
}
EVENT_TYPES = ["crash", "surge"]

# Stability selection
STABILITY_N_BOOTSTRAP = 100
STABILITY_SUBSAMPLE_FRAC = 0.8
STABILITY_C_VALUES = [0.01, 0.05, 0.1, 0.5, 1.0]

# EN hyperparameter grid
EN_C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]
EN_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]

# XGB hyperparameter grid
XGB_GRID = list(iterproduct(
    [2, 3, 5], [100, 200, 300], [0.01, 0.05, 0.1], [1, 3, 5],
))

N_INNER_FOLDS = 5


# ═══════════════════════════════════════════════════════════════
#  Feature helpers
# ═══════════════════════════════════════════════════════════════

def is_temporal_feature(col: str) -> bool:
    return any(col.startswith(p) for p in TEMPORAL_PREFIXES) or col == "n_prior_sessions"


def get_session_features(df: pd.DataFrame) -> list[str]:
    all_feats = [c for c in df.columns if c not in NON_FEATURE_COLS]
    return [c for c in all_feats if not is_temporal_feature(c)]


def get_temporal_features(df: pd.DataFrame) -> list[str]:
    all_feats = [c for c in df.columns if c not in NON_FEATURE_COLS]
    return [c for c in all_feats if is_temporal_feature(c)]


def compute_person_normalized(df: pd.DataFrame, session_cols: list[str]):
    """Compute pdev, pz, dvol for each session-level feature (on-the-fly)."""
    df = df.sort_values(["pid", "session_num"]).copy()
    pn_cols = []

    for col in session_cols:
        # ── pdev: deviation from personal running mean (prior sessions) ──
        prior_mean = df.groupby("pid")[col].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        df[f"pdev_{col}"] = (df[col] - prior_mean).fillna(0.0)
        pn_cols.append(f"pdev_{col}")

        # ── pz: z-scored deviation ──
        prior_std = df.groupby("pid")[col].transform(
            lambda x: x.expanding().std().shift(1)
        )
        pz = (df[col] - prior_mean) / prior_std.replace(0, np.nan)
        df[f"pz_{col}"] = pz.fillna(0.0)
        pn_cols.append(f"pz_{col}")

        # ── dvol: delta volatility (replicates exp10 logic) ──
        raw_delta = df.groupby("pid")[col].diff()
        abs_delta = raw_delta.abs()
        shifted_abs = abs_delta.groupby(df["pid"]).shift(1)
        prior_mean_abs = (
            shifted_abs.groupby(df["pid"])
            .expanding(min_periods=1)
            .mean()
        )
        prior_mean_abs = prior_mean_abs.reset_index(level=0, drop=True)
        df[f"dvol_{col}"] = (abs_delta - prior_mean_abs).fillna(0.0)
        pn_cols.append(f"dvol_{col}")

    return df, pn_cols


def select_features(
    df: pd.DataFrame,
    feature_set: str,
    session_cols: list[str],
    temporal_cols: list[str],
    pn_cols: list[str],
) -> list[str]:
    """Return the column names for a given feature-set code."""
    cols = []
    if "S" in feature_set:
        cols += session_cols
    if "T" in feature_set:
        cols += temporal_cols
    if "P" in feature_set:
        cols += pn_cols
    # Only keep columns that exist in df
    return [c for c in cols if c in df.columns]


# ═══════════════════════════════════════════════════════════════
#  Event labeling
# ═══════════════════════════════════════════════════════════════

def compute_event_labels(
    df: pd.DataFrame, construct: str, event_type: str, train_pids: list,
) -> pd.DataFrame:
    """Compute crash/surge labels using LOPO 1-SD threshold (matches exp10)."""
    sub = (
        df[["pid", "session_num", construct]]
        .sort_values(["pid", "session_num"])
        .copy()
    )
    sub["delta"] = sub.groupby("pid")[construct].diff()
    sub = sub.dropna(subset=["delta"]).copy()

    train_deltas = sub.loc[sub["pid"].isin(train_pids), "delta"]
    sd = train_deltas.std()

    if sd == 0 or np.isnan(sd):
        sub["label"] = 0
    elif event_type == "crash":
        sub["label"] = (sub["delta"] <= -1.0 * sd).astype(int)
    else:
        sub["label"] = (sub["delta"] >= 1.0 * sd).astype(int)

    return sub[["pid", "session_num", "label"]]


def compute_systemic_labels(
    df: pd.DataFrame, event_type: str, train_pids: list,
) -> pd.DataFrame:
    """Systemic event = 2+ constructs crash/surge at the same transition."""
    merged = None
    for i, construct in enumerate(CONSTRUCTS):
        lab = compute_event_labels(df, construct, event_type, train_pids)
        lab = lab.rename(columns={"label": f"lab_{i}"})
        if merged is None:
            merged = lab
        else:
            merged = merged.merge(lab, on=["pid", "session_num"])

    lab_cols = [f"lab_{i}" for i in range(len(CONSTRUCTS))]
    merged["label"] = (merged[lab_cols].sum(axis=1) >= 2).astype(int)
    return merged[["pid", "session_num", "label"]]


# ═══════════════════════════════════════════════════════════════
#  Evaluation metrics
# ═══════════════════════════════════════════════════════════════

def safe_auprc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, y_prob))


def best_f1(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    idx = int(np.argmax(f1s))
    t = float(thr[idx]) if idx < len(thr) else 0.5
    return float(f1s[idx]), t


# ═══════════════════════════════════════════════════════════════
#  Stability selection
# ═══════════════════════════════════════════════════════════════

def stratified_subsample(y, n, rng):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = max(1, int(n * len(pos_idx) / len(y)))
    n_neg = n - n_pos
    pos_sample = rng.choice(
        pos_idx, size=min(n_pos, len(pos_idx)), replace=len(pos_idx) < n_pos,
    )
    neg_sample = rng.choice(
        neg_idx, size=min(n_neg, len(neg_idx)), replace=len(neg_idx) < n_neg,
    )
    return np.concatenate([pos_sample, neg_sample])


def stability_selection(X_train, y_train, feature_names, rng):
    n_features = X_train.shape[1]
    selection_counts = np.zeros(n_features)
    total_fits = 0
    n_subsample = int(len(y_train) * STABILITY_SUBSAMPLE_FRAC)

    for _ in range(STABILITY_N_BOOTSTRAP):
        idx = stratified_subsample(y_train, n_subsample, rng)
        X_sub, y_sub = X_train[idx], y_train[idx]
        if len(np.unique(y_sub)) < 2:
            continue
        scaler = StandardScaler()
        X_sub_s = scaler.fit_transform(X_sub)
        for C_val in STABILITY_C_VALUES:
            try:
                m = LogisticRegression(
                    penalty="l1", solver="saga", C=C_val,
                    class_weight="balanced", max_iter=3000, random_state=42,
                )
                m.fit(X_sub_s, y_sub)
                selection_counts += (np.abs(m.coef_[0]) > 1e-8).astype(int)
                total_fits += 1
            except Exception:
                continue

    if total_fits == 0:
        return np.arange(n_features), np.ones(n_features), 0.0

    freq = selection_counts / total_fits

    # Adaptive threshold
    threshold = 0.5
    selected = np.where(freq > threshold)[0]
    if len(selected) < 5:
        threshold = 0.3
        selected = np.where(freq > threshold)[0]
    elif len(selected) > 40:
        threshold = 0.7
        selected = np.where(freq > threshold)[0]
    if len(selected) < 3:
        selected = np.argsort(freq)[::-1][:10]

    return selected, freq, threshold


# ═══════════════════════════════════════════════════════════════
#  Inner CV for hyperparameter tuning
# ═══════════════════════════════════════════════════════════════

def inner_cv_en(X_train, y_train, pids_arr, rng):
    unique = np.unique(pids_arr)
    inner_pids = rng.choice(
        unique, size=min(N_INNER_FOLDS, len(unique)), replace=False,
    )
    best_score, best_params = -1, (0.1, 0.5)

    for C_val in EN_C_VALUES:
        for l1 in EN_L1_RATIOS:
            scores = []
            for ip in inner_pids:
                mask = pids_arr == ip
                Xi_tr, Xi_te = X_train[~mask], X_train[mask]
                yi_tr, yi_te = y_train[~mask], y_train[mask]
                if (
                    len(np.unique(yi_tr)) < 2
                    or Xi_te.shape[0] == 0
                ):
                    continue
                try:
                    sc = StandardScaler()
                    m = LogisticRegression(
                        penalty="elasticnet", solver="saga",
                        C=C_val, l1_ratio=l1,
                        class_weight="balanced",
                        max_iter=5000, random_state=42,
                    )
                    m.fit(sc.fit_transform(Xi_tr), yi_tr)
                    scores.append(
                        safe_auprc(yi_te, m.predict_proba(sc.transform(Xi_te))[:, 1])
                    )
                except Exception:
                    continue
            valid = [s for s in scores if s is not None and not np.isnan(s)]
            if valid and np.mean(valid) > best_score:
                best_score = np.mean(valid)
                best_params = (C_val, l1)
    return best_params


def inner_cv_xgb(X_train, y_train, pids_arr, rng):
    unique = np.unique(pids_arr)
    inner_pids = rng.choice(
        unique, size=min(N_INNER_FOLDS, len(unique)), replace=False,
    )
    best_score, best_params = -1, (3, 100, 0.1, 1)

    for md, ne, lr, mcw in XGB_GRID:
        scores = []
        for ip in inner_pids:
            mask = pids_arr == ip
            Xi_tr, Xi_te = X_train[~mask], X_train[mask]
            yi_tr, yi_te = y_train[~mask], y_train[mask]
            if (
                len(np.unique(yi_tr)) < 2
                or Xi_te.shape[0] == 0
            ):
                continue
            n_pos_i = yi_tr.sum()
            spw_i = (len(yi_tr) - n_pos_i) / max(n_pos_i, 1)
            try:
                sc = StandardScaler()
                m = XGBClassifier(
                    max_depth=md, n_estimators=ne, learning_rate=lr,
                    min_child_weight=mcw, subsample=0.8,
                    colsample_bytree=0.8, scale_pos_weight=spw_i,
                    random_state=42, eval_metric="logloss",
                    verbosity=0, n_jobs=1,
                )
                m.fit(sc.fit_transform(Xi_tr), yi_tr)
                scores.append(
                    safe_auprc(yi_te, m.predict_proba(sc.transform(Xi_te))[:, 1])
                )
            except Exception:
                continue
        valid = [s for s in scores if s is not None and not np.isnan(s)]
        if valid and np.mean(valid) > best_score:
            best_score = np.mean(valid)
            best_params = (md, ne, lr, mcw)
    return best_params


# ═══════════════════════════════════════════════════════════════
#  Main experiment runner
# ═══════════════════════════════════════════════════════════════

def run_condition(feature_set: str, event_type: str, construct: str, task: str = "detection") -> dict:
    """Run one ablation condition. Returns result dict.

    task: "detection" — features from session t predict event at t-1→t
          "forecast"  — features from session t predict event at t→t+1
    """
    t0 = time.time()
    tag = f"{feature_set}.{event_type}.{construct}"
    if task == "forecast":
        tag = f"FC.{tag}"
    log = logging.getLogger(tag)
    log.info("Starting")

    # ── Load data ──
    df = pd.read_csv(DATA_PATH)
    session_cols = get_session_features(df)
    temporal_cols = get_temporal_features(df)

    # ── Compute person-normalized features ──
    if "P" in feature_set:
        df, pn_cols = compute_person_normalized(df, session_cols)
    else:
        pn_cols = []

    # ── Select features ──
    feat_cols = select_features(df, feature_set, session_cols, temporal_cols, pn_cols)
    if not feat_cols:
        log.error("No features selected!")
        return {"error": "no features", "feature_set": feature_set,
                "event": event_type, "construct": construct}

    log.info(f"Features: {len(feat_cols)} ({feature_set})")

    # ── LOPO CV ──
    pids = sorted(df["pid"].unique())
    en_all_y, en_all_p, en_all_pid = [], [], []
    xgb_all_y, xgb_all_p, xgb_all_pid = [], [], []
    en_coefs = defaultdict(list)
    xgb_imps = defaultdict(list)
    n_selected_per_fold = []
    rng = np.random.default_rng(42)

    for fold_i, test_pid in enumerate(pids):
        train_pids = [p for p in pids if p != test_pid]

        # Labels
        if construct == "systemic":
            label_df = compute_systemic_labels(df, event_type, train_pids)
        else:
            label_df = compute_event_labels(df, construct, event_type, train_pids)

        if task == "forecast":
            # For forecasting: features from session t predict event at t→t+1.
            # label_df has labels at the session where the event manifests (t+1).
            # Shift labels back by 1 so they align with features from session t.
            label_df = label_df.copy()
            label_df["session_num"] = label_df["session_num"] - 1
            # session_num < 1 would be invalid (no session 0)
            label_df = label_df[label_df["session_num"] >= 1]

        df_lab = df.merge(label_df, on=["pid", "session_num"], how="inner")
        train_df = df_lab[df_lab["pid"] != test_pid]
        test_df = df_lab[df_lab["pid"] == test_pid]

        if test_df.shape[0] == 0:
            continue

        X_train = np.nan_to_num(
            train_df[feat_cols].values.astype(np.float64), nan=0.0,
        )
        y_train = train_df["label"].values.astype(int)
        pids_train = train_df["pid"].values
        X_test = np.nan_to_num(
            test_df[feat_cols].values.astype(np.float64), nan=0.0,
        )
        y_test = test_df["label"].values.astype(int)

        if y_train.sum() < 2 or len(np.unique(y_train)) < 2:
            continue

        # ── Stability selection ──
        sel_idx, sel_freq, sel_thr = stability_selection(
            X_train, y_train, feat_cols, rng,
        )
        n_selected_per_fold.append(len(sel_idx))
        X_train_sel = X_train[:, sel_idx]
        X_test_sel = X_test[:, sel_idx]
        sel_names = [feat_cols[i] for i in sel_idx]

        # ── EN ──
        best_C, best_l1 = inner_cv_en(X_train_sel, y_train, pids_train, rng)
        sc_en = StandardScaler()
        X_tr_s = sc_en.fit_transform(X_train_sel)
        X_te_s = sc_en.transform(X_test_sel)
        try:
            en_model = LogisticRegression(
                penalty="elasticnet", solver="saga",
                C=best_C, l1_ratio=best_l1,
                class_weight="balanced", max_iter=5000, random_state=42,
            )
            en_model.fit(X_tr_s, y_train)
            en_prob = en_model.predict_proba(X_te_s)[:, 1]
            en_all_y.extend(y_test.tolist())
            en_all_p.extend(en_prob.tolist())
            en_all_pid.extend([int(test_pid)] * len(y_test))
            for j, name in enumerate(sel_names):
                en_coefs[name].append(abs(float(en_model.coef_[0][j])))
        except Exception as e:
            log.warning(f"EN fold {fold_i} failed: {e}")

        # ── XGB ──
        best_md, best_ne, best_lr, best_mcw = inner_cv_xgb(
            X_train_sel, y_train, pids_train, rng,
        )
        n_pos = y_train.sum()
        spw = (len(y_train) - n_pos) / max(n_pos, 1)
        sc_xgb = StandardScaler()
        X_tr_sx = sc_xgb.fit_transform(X_train_sel)
        X_te_sx = sc_xgb.transform(X_test_sel)
        try:
            xgb_model = XGBClassifier(
                max_depth=best_md, n_estimators=best_ne,
                learning_rate=best_lr, min_child_weight=best_mcw,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=spw, random_state=42,
                eval_metric="logloss", verbosity=0, n_jobs=1,
            )
            xgb_model.fit(X_tr_sx, y_train)
            xgb_prob = xgb_model.predict_proba(X_te_sx)[:, 1]
            xgb_all_y.extend(y_test.tolist())
            xgb_all_p.extend(xgb_prob.tolist())
            xgb_all_pid.extend([int(test_pid)] * len(y_test))
            imps = xgb_model.feature_importances_
            for j, name in enumerate(sel_names):
                xgb_imps[name].append(float(imps[j]))
        except Exception as e:
            log.warning(f"XGB fold {fold_i} failed: {e}")

        if (fold_i + 1) % 6 == 0:
            log.info(f"  fold {fold_i + 1}/{len(pids)}")

    # ── Aggregate metrics ──
    en_y = np.array(en_all_y)
    en_p = np.array(en_all_p)
    xgb_y = np.array(xgb_all_y)
    xgb_p = np.array(xgb_all_p)

    en_auprc = safe_auprc(en_y, en_p) if len(en_y) > 0 else np.nan
    xgb_auprc = safe_auprc(xgb_y, xgb_p) if len(xgb_y) > 0 else np.nan
    en_f1, en_thr = best_f1(en_y, en_p) if len(en_y) > 0 else (np.nan, np.nan)
    xgb_f1, xgb_thr = best_f1(xgb_y, xgb_p) if len(xgb_y) > 0 else (np.nan, np.nan)

    # Base rate
    base_rate = float(en_y.mean()) if len(en_y) > 0 else np.nan

    # Top features
    def top_feats(coef_dict, k=15):
        ranked = sorted(coef_dict.items(), key=lambda x: np.mean(x[1]), reverse=True)
        return [
            {"feature": f, "mean_importance": round(float(np.mean(v)), 6),
             "n_folds": len(v)}
            for f, v in ranked[:k]
        ]

    elapsed = time.time() - t0

    result = {
        "task": task,
        "feature_set": feature_set,
        "feature_set_label": FEATURE_SET_LABELS[feature_set],
        "event_type": event_type,
        "construct": construct,
        "n_features_total": len(feat_cols),
        "n_features_selected_mean": (
            round(float(np.mean(n_selected_per_fold)), 1)
            if n_selected_per_fold else 0
        ),
        "base_rate": round(base_rate, 4) if not np.isnan(base_rate) else None,
        "n_samples": int(len(en_y)),
        "n_events": int(en_y.sum()) if len(en_y) > 0 else 0,
        "elastic_net": {
            "AUPRC": round(en_auprc, 4) if not np.isnan(en_auprc) else None,
            "best_F1": round(en_f1, 4) if not np.isnan(en_f1) else None,
            "top_features": top_feats(en_coefs),
        },
        "xgboost": {
            "AUPRC": round(xgb_auprc, 4) if not np.isnan(xgb_auprc) else None,
            "best_F1": round(xgb_f1, 4) if not np.isnan(xgb_f1) else None,
            "top_features": top_feats(xgb_imps),
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    log.info(
        f"Done in {elapsed:.0f}s — EN={en_auprc:.3f} XGB={xgb_auprc:.3f}"
        if not (np.isnan(en_auprc) or np.isnan(xgb_auprc))
        else f"Done in {elapsed:.0f}s — incomplete"
    )
    return result


# ═══════════════════════════════════════════════════════════════
#  Job scheduling
# ═══════════════════════════════════════════════════════════════

def all_conditions():
    """Generate all 84 (feature_set, event_type, construct) tuples."""
    return [
        (fs, ev, con)
        for fs in FEATURE_SETS
        for ev in EVENT_TYPES
        for con in TARGETS
    ]


def worker_conditions(worker_id: int, n_workers: int):
    """Round-robin assignment of conditions to this worker."""
    conds = all_conditions()
    return [c for i, c in enumerate(conds) if i % n_workers == worker_id]


def run_and_save(args_tuple):
    """Wrapper for ProcessPoolExecutor: run one condition and save JSON."""
    fs, ev, con, task = args_tuple
    results_dir = get_results_dir(task)
    out_path = results_dir / f"{fs}_{ev}_{con}.json"

    # Skip if already done
    if out_path.exists():
        try:
            with open(out_path) as f:
                existing = json.load(f)
            if existing.get("elastic_net", {}).get("AUPRC") is not None:
                logging.info(f"SKIP (exists): {fs} {ev} {con}")
                return existing
        except Exception:
            pass

    result = run_condition(fs, ev, con, task=task)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ═══════════════════════════════════════════════════════════════
#  Summarize results
# ═══════════════════════════════════════════════════════════════

def summarize(task: str = "detection"):
    """Collect all ablation JSONs into a summary CSV and print a table."""
    results_dir = get_results_dir(task)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*72}")
    print(f"  Task: {task.upper()}")
    print(f"{'='*72}")
    rows = []
    for p in sorted(results_dir.glob("*.json")):
        if p.name == "summary.json":
            continue
        try:
            with open(p) as f:
                r = json.load(f)
        except Exception:
            continue
        if "error" in r:
            continue
        en = r.get("elastic_net", {})
        xgb = r.get("xgboost", {})
        en_auprc = en.get("AUPRC")
        xgb_auprc = xgb.get("AUPRC")
        best = max(
            en_auprc or 0, xgb_auprc or 0,
        )
        rows.append({
            "feature_set": r["feature_set"],
            "label": r.get("feature_set_label", r["feature_set"]),
            "event": r["event_type"],
            "construct": r["construct"],
            "n_features": r.get("n_features_total", ""),
            "base_rate": r.get("base_rate", ""),
            "EN_AUPRC": en_auprc,
            "XGB_AUPRC": xgb_auprc,
            "best_AUPRC": best if best > 0 else None,
            "EN_F1": en.get("best_F1"),
            "XGB_F1": xgb.get("best_F1"),
        })

    if not rows:
        print("No results found in", results_dir)
        return

    df = pd.DataFrame(rows)
    csv_path = results_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} results to {csv_path}\n")

    # ── Condensed table: mean AUPRC per feature set × event ──
    print("=" * 72)
    print("Mean best AUPRC across constructs (excl. systemic)")
    print("=" * 72)
    non_sys = df[df["construct"] != "systemic"].copy()

    pivot = non_sys.pivot_table(
        index="feature_set",
        columns="event",
        values="best_AUPRC",
        aggfunc="mean",
    )
    # Reorder rows
    order = [fs for fs in FEATURE_SETS if fs in pivot.index]
    pivot = pivot.loc[order]
    pivot["overall"] = pivot.mean(axis=1)
    pivot = pivot.round(4)

    # Add labels
    pivot.insert(0, "label", [FEATURE_SET_LABELS.get(fs, fs) for fs in pivot.index])

    print(pivot.to_string())
    print()

    # ── Detailed table ──
    print("=" * 72)
    print("Detailed results (best of EN/XGB)")
    print("=" * 72)
    detail = df.pivot_table(
        index=["feature_set", "event"],
        columns="construct",
        values="best_AUPRC",
    )
    cols_order = [c for c in CONSTRUCTS + ["systemic"] if c in detail.columns]
    detail = detail[cols_order].round(3)
    print(detail.to_string())
    print()

    # ── Best feature set per construct ──
    print("=" * 72)
    print("Best feature set per construct × event")
    print("=" * 72)
    for ev in EVENT_TYPES:
        print(f"\n  {ev.upper()}:")
        sub = df[df["event"] == ev]
        for con in TARGETS:
            s = sub[sub["construct"] == con]
            if s.empty:
                continue
            best_row = s.loc[s["best_AUPRC"].idxmax()]
            print(
                f"    {con:25s} → {best_row['feature_set']:3s} "
                f"({best_row['label']:22s}) "
                f"AUPRC={best_row['best_AUPRC']:.3f}"
            )
    print()


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Ablation: detection/forecasting across 7 feature-set combinations",
    )
    sub = parser.add_subparsers(dest="mode")

    # Single condition
    single = sub.add_parser("single", help="Run one condition")
    single.add_argument("--task", choices=["detection", "forecast"], default="detection")
    single.add_argument("--feature_set", required=True, choices=FEATURE_SETS)
    single.add_argument("--event", required=True, choices=EVENT_TYPES)
    single.add_argument("--construct", required=True, choices=TARGETS)

    # All conditions
    run_all = sub.add_parser("all", help="Run all (or a worker's share of) conditions")
    run_all.add_argument("--task", choices=["detection", "forecast"], default="detection")
    run_all.add_argument("--worker_id", type=int, default=0)
    run_all.add_argument("--n_workers", type=int, default=1)
    run_all.add_argument("--n_parallel", type=int, default=4)

    # Summarize
    summ = sub.add_parser("summarize", help="Collect results into summary table")
    summ.add_argument("--task", choices=["detection", "forecast"], default="detection")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.mode == "single":
        results_dir = get_results_dir(args.task)
        results_dir.mkdir(parents=True, exist_ok=True)
        result = run_condition(args.feature_set, args.event, args.construct, task=args.task)
        out = results_dir / f"{args.feature_set}_{args.event}_{args.construct}.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {out}")

    elif args.mode == "all":
        results_dir = get_results_dir(args.task)
        results_dir.mkdir(parents=True, exist_ok=True)
        conds = worker_conditions(args.worker_id, args.n_workers)
        # Append task to each condition tuple
        conds = [(fs, ev, con, args.task) for fs, ev, con in conds]
        total = len(conds)
        logging.info(
            f"Worker {args.worker_id}/{args.n_workers} [{args.task}]: "
            f"{total} conditions, {args.n_parallel} parallel"
        )

        done = 0
        with ProcessPoolExecutor(max_workers=args.n_parallel) as pool:
            futures = {pool.submit(run_and_save, c): c for c in conds}
            for fut in as_completed(futures):
                done += 1
                c = futures[fut]
                try:
                    r = fut.result()
                    en = r.get("elastic_net", {}).get("AUPRC", "?")
                    xgb = r.get("xgboost", {}).get("AUPRC", "?")
                    logging.info(
                        f"[{done}/{total}] {c[0]} {c[1]} {c[2]}: "
                        f"EN={en} XGB={xgb}"
                    )
                except Exception as e:
                    logging.error(f"[{done}/{total}] {c} FAILED: {e}")

        logging.info("All done. Run with 'summarize --task %s' to see results.", args.task)

    elif args.mode == "summarize":
        summarize(task=args.task)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
