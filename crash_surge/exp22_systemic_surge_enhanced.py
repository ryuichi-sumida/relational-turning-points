#!/usr/bin/env python3
"""
Experiment 22: Enhanced Systemic Surge Detector with Stability Selection

Mirrors Exp11 (enhanced systemic crash detector) but for surges.
Uses personal baseline features (pdev/pz) + delta-volatility (dvol)
+ stability selection for feature reduction.

Current Exp21 best (raw features): XGB AUPRC=0.248, F1=0.373
Goal: Push higher with richer per-person normalization, matching the
59% improvement seen for crashes in Exp11.
"""

import json
import sys
import time
import warnings
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "full_features_with_temporals.csv"
RESULTS_DIR = BASE / "crash_surge" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONSTRUCTS = ["familiarity", "social_penetration", "memory", "conversational", "enjoyment"]
NON_FEATURE_COLS = {"pid", "session_num", "n_prior_sessions"} | set(CONSTRUCTS)
TEMPORAL_PREFIXES = ("delta_", "max_prior_", "EMA_", "trend_", "ever_", "cumulative_")

# Stability selection config
STABILITY_N_BOOTSTRAP = 100
STABILITY_SUBSAMPLE_FRAC = 0.8
STABILITY_C_VALUES = [0.01, 0.05, 0.1, 0.5, 1.0]

# Model hyperparam grids
EN_C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]
EN_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
XGB_GRID = list(product(
    [2, 3, 5],
    [100, 200, 300],
    [0.01, 0.05, 0.1],
    [1, 3, 5],
))
N_INNER_FOLDS = 5


# -- Feature engineering -----------------------------------------------------

def is_temporal_feature(col):
    return any(col.startswith(p) for p in TEMPORAL_PREFIXES) or col == "n_prior_sessions"


def get_session_level_features(df):
    all_feats = [c for c in df.columns if c not in NON_FEATURE_COLS]
    return [c for c in all_feats if not is_temporal_feature(c)]


def compute_personal_baselines(df, session_level_cols):
    df = df.sort_values(["pid", "session_num"]).copy()
    for col in session_level_cols:
        shifted = df.groupby("pid")[col].shift(1)
        prior_mean = shifted.groupby(df["pid"]).expanding(min_periods=1).mean()
        prior_mean = prior_mean.reset_index(level=0, drop=True)
        prior_std = shifted.groupby(df["pid"]).expanding(min_periods=2).std()
        prior_std = prior_std.reset_index(level=0, drop=True)
        dev = df[col] - prior_mean
        z = dev / prior_std
        df[f"pdev_{col}"] = dev.fillna(0.0)
        df[f"pz_{col}"] = z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df


def compute_delta_volatility(df, session_level_cols):
    df = df.sort_values(["pid", "session_num"]).copy()
    for col in session_level_cols:
        raw_delta = df.groupby("pid")[col].diff()
        abs_delta = raw_delta.abs()
        shifted_abs = abs_delta.groupby(df["pid"]).shift(1)
        prior_mean_abs = shifted_abs.groupby(df["pid"]).expanding(min_periods=1).mean()
        prior_mean_abs = prior_mean_abs.reset_index(level=0, drop=True)
        dvol = abs_delta - prior_mean_abs
        df[f"dvol_{col}"] = dvol.fillna(0.0)
    return df


# -- Systemic SURGE labeling ------------------------------------------------

def compute_systemic_surge_labels(df, train_pids):
    """Systemic surge = 2+ constructs surging simultaneously."""
    surge_flags = pd.DataFrame(index=df.index)
    thresholds = {}
    train_mask = df["pid"].isin(train_pids)

    for construct in CONSTRUCTS:
        deltas = df.groupby("pid")[construct].diff()
        train_deltas = deltas[train_mask].dropna()
        sd = train_deltas.std()
        thresholds[construct] = float(sd)
        surge = pd.Series(False, index=df.index)
        valid = deltas.notna()
        surge[valid] = deltas[valid] >= sd
        surge[~valid] = np.nan
        surge_flags[construct] = surge

    surge_count = surge_flags[CONSTRUCTS].sum(axis=1)
    systemic_mask = surge_count >= 2

    label = pd.Series(np.nan, index=df.index)
    valid_rows = surge_flags[CONSTRUCTS].notna().all(axis=1)
    label[valid_rows] = 0
    label[valid_rows & systemic_mask] = 1

    return label, thresholds


# -- Stability selection ----------------------------------------------------

def stratified_subsample(y, n, rng):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = max(1, int(n * len(pos_idx) / len(y)))
    n_neg = n - n_pos
    pos_sample = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)),
                            replace=len(pos_idx) < n_pos)
    neg_sample = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)),
                            replace=len(neg_idx) < n_neg)
    return np.concatenate([pos_sample, neg_sample])


def stability_selection(X_train, y_train, feature_names, rng):
    n_features = X_train.shape[1]
    selection_counts = np.zeros(n_features)
    total_fits = 0
    n_subsample = int(len(y_train) * STABILITY_SUBSAMPLE_FRAC)

    for b in range(STABILITY_N_BOOTSTRAP):
        idx = stratified_subsample(y_train, n_subsample, rng)
        X_sub = X_train[idx]
        y_sub = y_train[idx]
        if len(np.unique(y_sub)) < 2:
            continue
        scaler = StandardScaler()
        X_sub_s = scaler.fit_transform(X_sub)
        for C_val in STABILITY_C_VALUES:
            try:
                model = LogisticRegression(
                    penalty='l1', solver='saga', C=C_val,
                    class_weight='balanced', max_iter=3000, random_state=42)
                model.fit(X_sub_s, y_sub)
                nonzero = np.abs(model.coef_[0]) > 1e-8
                selection_counts += nonzero.astype(int)
                total_fits += 1
            except Exception:
                continue

    if total_fits == 0:
        return np.arange(n_features), np.ones(n_features), 0.0

    frequencies = selection_counts / total_fits
    threshold = 0.5
    selected = np.where(frequencies > threshold)[0]
    if len(selected) < 5:
        threshold = 0.3
        selected = np.where(frequencies > threshold)[0]
    elif len(selected) > 40:
        threshold = 0.7
        selected = np.where(frequencies > threshold)[0]
    if len(selected) < 3:
        selected = np.argsort(frequencies)[::-1][:10]

    return selected, frequencies, threshold


# -- Metrics ----------------------------------------------------------------

def safe_auprc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_prob)


def recall_at_precision_fn(y_true, y_prob, target_precision):
    if len(np.unique(y_true)) < 2:
        return np.nan
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    valid = prec >= target_precision
    return float(rec[valid].max()) if valid.any() else 0.0


def best_f1_fn(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    idx = np.argmax(f1s)
    thr = thresholds[idx] if idx < len(thresholds) else 0.5
    return float(f1s[idx]), float(thr)


# -- Inner CV ---------------------------------------------------------------

def inner_cv_elasticnet(X_train, y_train, train_pids_arr, rng):
    unique_inner = np.unique(train_pids_arr)
    inner_pids = rng.choice(unique_inner, size=min(N_INNER_FOLDS, len(unique_inner)), replace=False)
    best_score, best_params = -1, (0.1, 0.5)
    for C_val in EN_C_VALUES:
        for l1 in EN_L1_RATIOS:
            scores = []
            for ip in inner_pids:
                mask = train_pids_arr == ip
                Xi_tr, Xi_te = X_train[~mask], X_train[mask]
                yi_tr, yi_te = y_train[~mask], y_train[mask]
                if len(np.unique(yi_tr)) < 2 or len(np.unique(yi_te)) < 2 or Xi_te.shape[0] == 0:
                    continue
                scaler = StandardScaler()
                try:
                    model = LogisticRegression(penalty="elasticnet", solver="saga", C=C_val,
                                               l1_ratio=l1, class_weight="balanced",
                                               max_iter=5000, random_state=42)
                    model.fit(scaler.fit_transform(Xi_tr), yi_tr)
                    scores.append(safe_auprc(yi_te, model.predict_proba(scaler.transform(Xi_te))[:, 1]))
                except Exception:
                    continue
            valid = [s for s in scores if s is not None and not np.isnan(s)]
            if valid and np.mean(valid) > best_score:
                best_score = np.mean(valid)
                best_params = (C_val, l1)
    return best_params


def inner_cv_xgboost(X_train, y_train, train_pids_arr, rng):
    unique_inner = np.unique(train_pids_arr)
    inner_pids = rng.choice(unique_inner, size=min(N_INNER_FOLDS, len(unique_inner)), replace=False)
    best_score, best_params = -1, (3, 100, 0.1, 1)
    for md, ne, lr, mcw in XGB_GRID:
        scores = []
        for ip in inner_pids:
            mask = train_pids_arr == ip
            Xi_tr, Xi_te = X_train[~mask], X_train[mask]
            yi_tr, yi_te = y_train[~mask], y_train[mask]
            if len(np.unique(yi_tr)) < 2 or len(np.unique(yi_te)) < 2 or Xi_te.shape[0] == 0:
                continue
            scaler = StandardScaler()
            n_pos_i = yi_tr.sum()
            spw_i = (len(yi_tr) - n_pos_i) / max(n_pos_i, 1)
            try:
                model = XGBClassifier(max_depth=md, n_estimators=ne, learning_rate=lr,
                                       min_child_weight=mcw, subsample=0.8, colsample_bytree=0.8,
                                       scale_pos_weight=spw_i, random_state=42,
                                       eval_metric="logloss", verbosity=0, n_jobs=1)
                model.fit(scaler.fit_transform(Xi_tr), yi_tr)
                scores.append(safe_auprc(yi_te, model.predict_proba(scaler.transform(Xi_te))[:, 1]))
            except Exception:
                continue
        valid = [s for s in scores if s is not None and not np.isnan(s)]
        if valid and np.mean(valid) > best_score:
            best_score = np.mean(valid)
            best_params = (md, ne, lr, mcw)
    return best_params


# -- Main experiment --------------------------------------------------------

def run_experiment(feature_mode="all"):
    t_total = time.time()

    print("=" * 72)
    print(f"EXPERIMENT 22: SYSTEMIC SURGE DETECTOR — ENHANCED FEATURES ({feature_mode})")
    print("=" * 72)

    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    session_level_cols = get_session_level_features(df)
    print(f"Session-level features: {len(session_level_cols)}")

    # 2. Compute enhanced features
    df = compute_personal_baselines(df, session_level_cols)
    df = compute_delta_volatility(df, session_level_cols)

    pdev_cols = sorted([c for c in df.columns if c.startswith("pdev_")])
    pz_cols = sorted([c for c in df.columns if c.startswith("pz_")])
    dvol_cols = sorted([c for c in df.columns if c.startswith("dvol_")])

    # 3. Build feature set based on mode
    if feature_mode == "pdev_pz":
        feature_cols = sorted(session_level_cols + pdev_cols + pz_cols)
    elif feature_mode == "dvol":
        feature_cols = sorted(session_level_cols + dvol_cols)
    else:  # "all"
        feature_cols = sorted(session_level_cols + pdev_cols + pz_cols + dvol_cols)

    print(f"Total candidate features: {len(feature_cols)}")

    pids = sorted(df["pid"].unique())
    n_pids = len(pids)
    rng = np.random.RandomState(42)

    # 4. LOPO loop
    en_preds = {"pid": [], "y_true": [], "y_prob": []}
    xgb_preds = {"pid": [], "y_true": [], "y_prob": []}
    en_coefs_dict = defaultdict(list)
    xgb_importance_dict = defaultdict(list)
    feature_selection_counts = defaultdict(int)
    fold_info = {}

    for fold_i, test_pid in enumerate(pids):
        t0 = time.time()
        train_pids = [p for p in pids if p != test_pid]

        # Compute systemic SURGE labels using training thresholds
        systemic_label, thresholds = compute_systemic_surge_labels(df, train_pids)

        valid = systemic_label.notna()
        train_mask = df["pid"].isin(train_pids)
        test_mask = df["pid"] == test_pid
        train_valid = train_mask & valid
        test_valid = test_mask & valid

        if test_valid.sum() == 0:
            continue

        X_train_raw = np.nan_to_num(df.loc[train_valid, feature_cols].values.astype(np.float64), nan=0.0)
        y_train = systemic_label[train_valid].values.astype(int)
        pids_train = df.loc[train_valid, "pid"].values
        X_test_raw = np.nan_to_num(df.loc[test_valid, feature_cols].values.astype(np.float64), nan=0.0)
        y_test = systemic_label[test_valid].values.astype(int)

        if len(np.unique(y_train)) < 2:
            continue

        # Stability selection
        selected_idx, frequencies, ss_threshold = stability_selection(
            X_train_raw, y_train, feature_cols, rng
        )
        selected_names = [feature_cols[i] for i in selected_idx]
        for name in selected_names:
            feature_selection_counts[name] += 1

        fold_info[int(test_pid)] = {"n_selected": len(selected_idx)}

        X_train = X_train_raw[:, selected_idx]
        X_test = X_test_raw[:, selected_idx]

        # Elastic-Net
        best_C, best_l1 = inner_cv_elasticnet(X_train, y_train, pids_train, rng)
        scaler_en = StandardScaler()
        X_train_s = scaler_en.fit_transform(X_train)
        X_test_s = scaler_en.transform(X_test)
        en_model = LogisticRegression(penalty="elasticnet", solver="saga", C=best_C,
                                       l1_ratio=best_l1, class_weight="balanced",
                                       max_iter=5000, random_state=42)
        en_model.fit(X_train_s, y_train)
        en_prob = en_model.predict_proba(X_test_s)[:, 1]
        en_preds["pid"].extend([test_pid] * len(y_test))
        en_preds["y_true"].extend(y_test.tolist())
        en_preds["y_prob"].extend(en_prob.tolist())
        for fname, coef in zip(selected_names, en_model.coef_[0]):
            en_coefs_dict[fname].append(abs(coef))

        # XGBoost
        best_md, best_ne, best_lr, best_mcw = inner_cv_xgboost(
            X_train, y_train, pids_train, rng
        )
        n_pos = y_train.sum()
        spw = (len(y_train) - n_pos) / max(n_pos, 1)
        scaler_xgb = StandardScaler()
        X_train_sx = scaler_xgb.fit_transform(X_train)
        X_test_sx = scaler_xgb.transform(X_test)
        xgb_model = XGBClassifier(max_depth=best_md, n_estimators=best_ne,
                                    learning_rate=best_lr, min_child_weight=best_mcw,
                                    subsample=0.8, colsample_bytree=0.8,
                                    scale_pos_weight=spw, random_state=42,
                                    eval_metric="logloss", verbosity=0, n_jobs=1)
        xgb_model.fit(X_train_sx, y_train)
        xgb_prob = xgb_model.predict_proba(X_test_sx)[:, 1]
        xgb_preds["pid"].extend([test_pid] * len(y_test))
        xgb_preds["y_true"].extend(y_test.tolist())
        xgb_preds["y_prob"].extend(xgb_prob.tolist())
        for fname, imp in zip(selected_names, xgb_model.feature_importances_):
            xgb_importance_dict[fname].append(imp)

        elapsed = time.time() - t0
        print(f"  Fold {fold_i+1}/{n_pids} (pid={test_pid}): "
              f"sel={len(selected_idx)}, {elapsed:.0f}s")
        sys.stdout.flush()

    # 5. Compute metrics
    def compute_metrics(preds_dict, coefs_dict, model_name):
        y_true = np.array(preds_dict["y_true"])
        y_prob = np.array(preds_dict["y_prob"])
        auprc = safe_auprc(y_true, y_prob)
        r50 = recall_at_precision_fn(y_true, y_prob, 0.50)
        bf1, bf1_thr = best_f1_fn(y_true, y_prob)
        mf1_val = f1_score(
            y_true, (y_prob >= (bf1_thr if not np.isnan(bf1_thr) else 0.5)).astype(int),
            average="macro", zero_division=0
        )
        top_feats = sorted(coefs_dict.items(), key=lambda x: np.mean(x[1]), reverse=True)[:20]
        return {
            "n_samples": int(len(y_true)),
            "n_systemic_surge": int(y_true.sum()),
            "systemic_surge_rate": round(float(y_true.mean()), 4),
            "AUPRC": round(float(auprc), 4) if not np.isnan(auprc) else None,
            "recall_at_50pct_precision": round(float(r50), 4) if not np.isnan(r50) else None,
            "best_F1": round(float(bf1), 4) if not np.isnan(bf1) else None,
            "macro_F1": round(float(mf1_val), 4),
            "top20_features": [
                {"feature": name, f"mean_{model_name}": round(float(np.mean(vals)), 6),
                 "n_folds_used": len(vals)}
                for name, vals in top_feats
            ],
        }

    en_metrics = compute_metrics(en_preds, en_coefs_dict, "abs_coef")
    xgb_metrics = compute_metrics(xgb_preds, xgb_importance_dict, "importance")

    # Stable features
    stable_features = {k: v for k, v in feature_selection_counts.items()
                       if v >= n_pids * 0.5}

    n_sel_per_fold = [info["n_selected"] for info in fold_info.values()]

    result = {
        "experiment": f"exp22_systemic_surge_enhanced_{feature_mode}",
        "feature_mode": feature_mode,
        "n_candidate_features": len(feature_cols),
        "elastic_net": en_metrics,
        "xgboost": xgb_metrics,
        "stability_selection": {
            "mean_selected": round(float(np.mean(n_sel_per_fold)), 1),
            "median_selected": int(np.median(n_sel_per_fold)),
            "n_stable_features": len(stable_features),
            "stable_features": sorted(stable_features.keys(),
                                       key=lambda x: stable_features[x], reverse=True),
            "feature_fold_counts": dict(sorted(
                feature_selection_counts.items(), key=lambda x: x[1], reverse=True
            )[:30]),
        },
        "comparison": {
            "exp21_en_auprc": 0.204,
            "exp21_xgb_auprc": 0.248,
            "exp21_xgb_best_f1": 0.373,
            "exp11_crash_en_auprc": 0.242,
            "exp11_crash_xgb_auprc": None,
        },
    }

    # Print summary
    print(f"\n{'=' * 72}")
    print(f"RESULTS: Systemic Surge Detector — {feature_mode}")
    print(f"{'=' * 72}")
    print(f"  EN:  AUPRC={en_metrics['AUPRC']}, F1={en_metrics['best_F1']}")
    print(f"  XGB: AUPRC={xgb_metrics['AUPRC']}, F1={xgb_metrics['best_F1']}")
    print(f"  (Exp21 baseline: EN=0.204, XGB=0.248)")
    print(f"  Stable features ({len(stable_features)}): {list(stable_features.keys())[:10]}")
    print(f"  Total time: {time.time()-t_total:.0f}s ({(time.time()-t_total)/60:.1f}min)")

    out_path = RESULTS_DIR / f"exp22_systemic_surge_{feature_mode}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {out_path}")

    return result


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "--all-modes":
        for m in ["pdev_pz", "dvol", "all"]:
            run_experiment(m)
    else:
        run_experiment(mode)
