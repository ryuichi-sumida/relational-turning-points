#!/usr/bin/env python3
"""
Experiment 26: Paired Bootstrap Test — Crash vs Surge Detectability

Direct statistical comparison of whether surge AUPRC > crash AUPRC
per construct. Uses the SAME LOPO folds for both crash and surge
predictions, enabling a paired bootstrap test.

Key claim to test: "Surges are more detectable than crashes"
(Conv: 0.395 vs 0.236, Memory: 0.227 vs 0.138)
"""

import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")

from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "full_features_with_temporals.csv"
RESULTS_DIR = BASE / "crash_surge" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONSTRUCTS = ["familiarity", "social_penetration", "memory", "conversational", "enjoyment"]
NON_FEATURE_COLS = {"pid", "session_num", "n_prior_sessions"} | set(CONSTRUCTS)

N_BOOTSTRAP = 2000

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not available, skipping XGB models")


def safe_auprc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_prob)


def get_all_feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def collect_paired_predictions(df, feat_cols, construct, pids):
    """
    Run LOPO for both crash and surge detection on the SAME folds.
    Returns per-fold predictions for paired comparison.
    """
    crash_preds = {"y_true": [], "y_score_en": [], "y_score_xgb": [], "fold_ids": []}
    surge_preds = {"y_true": [], "y_score_en": [], "y_score_xgb": [], "fold_ids": []}

    for fold_i, test_pid in enumerate(pids):
        train_pids = [p for p in pids if p != test_pid]
        train_mask = df["pid"].isin(train_pids)
        test_mask = df["pid"] == test_pid

        deltas = df.groupby("pid")[construct].diff()
        train_deltas = deltas[train_mask].dropna()
        sd = train_deltas.std()

        # Crash labels
        crash_label = pd.Series(np.nan, index=df.index)
        valid = deltas.notna()
        crash_label[valid & (deltas <= -sd)] = 1.0
        crash_label[valid & (deltas > -sd)] = 0.0

        # Surge labels
        surge_label = pd.Series(np.nan, index=df.index)
        surge_label[valid & (deltas >= sd)] = 1.0
        surge_label[valid & (deltas < sd)] = 0.0

        # Common valid rows
        crash_valid = crash_label.notna()
        surge_valid = surge_label.notna()

        train_crash_valid = train_mask & crash_valid
        test_crash_valid = test_mask & crash_valid
        train_surge_valid = train_mask & surge_valid
        test_surge_valid = test_mask & surge_valid

        if test_crash_valid.sum() == 0 or test_surge_valid.sum() == 0:
            continue

        # --- CRASH ---
        X_train_c = np.nan_to_num(df.loc[train_crash_valid, feat_cols].values.astype(np.float32), nan=0.0)
        y_train_c = crash_label[train_crash_valid].values.astype(int)
        X_test_c = np.nan_to_num(df.loc[test_crash_valid, feat_cols].values.astype(np.float32), nan=0.0)
        y_test_c = crash_label[test_crash_valid].values.astype(int)

        if len(np.unique(y_train_c)) < 2:
            continue

        scaler_c = StandardScaler()
        X_train_cs = scaler_c.fit_transform(X_train_c)
        X_test_cs = scaler_c.transform(X_test_c)

        # EN crash
        clf_en_c = LogisticRegression(C=0.1, l1_ratio=0.5, penalty="elasticnet",
                                       solver="saga", class_weight="balanced",
                                       max_iter=5000, random_state=42)
        clf_en_c.fit(X_train_cs, y_train_c)
        crash_preds["y_true"].extend(y_test_c.tolist())
        crash_preds["y_score_en"].extend(clf_en_c.predict_proba(X_test_cs)[:, 1].tolist())
        crash_preds["fold_ids"].extend([fold_i] * len(y_test_c))

        # XGB crash
        if HAS_XGB:
            spw_c = float((y_train_c == 0).sum()) / max(float((y_train_c == 1).sum()), 1)
            clf_xgb_c = XGBClassifier(max_depth=5, n_estimators=100, learning_rate=0.1,
                                       scale_pos_weight=spw_c, random_state=42,
                                       eval_metric="logloss", verbosity=0, n_jobs=1)
            clf_xgb_c.fit(X_train_cs, y_train_c)
            crash_preds["y_score_xgb"].extend(clf_xgb_c.predict_proba(X_test_cs)[:, 1].tolist())

        # --- SURGE ---
        X_train_s = np.nan_to_num(df.loc[train_surge_valid, feat_cols].values.astype(np.float32), nan=0.0)
        y_train_s = surge_label[train_surge_valid].values.astype(int)
        X_test_s = np.nan_to_num(df.loc[test_surge_valid, feat_cols].values.astype(np.float32), nan=0.0)
        y_test_s = surge_label[test_surge_valid].values.astype(int)

        if len(np.unique(y_train_s)) < 2:
            continue

        scaler_s = StandardScaler()
        X_train_ss = scaler_s.fit_transform(X_train_s)
        X_test_ss = scaler_s.transform(X_test_s)

        # EN surge
        clf_en_s = LogisticRegression(C=0.1, l1_ratio=0.5, penalty="elasticnet",
                                       solver="saga", class_weight="balanced",
                                       max_iter=5000, random_state=42)
        clf_en_s.fit(X_train_ss, y_train_s)
        surge_preds["y_true"].extend(y_test_s.tolist())
        surge_preds["y_score_en"].extend(clf_en_s.predict_proba(X_test_ss)[:, 1].tolist())
        surge_preds["fold_ids"].extend([fold_i] * len(y_test_s))

        # XGB surge
        if HAS_XGB:
            spw_s = float((y_train_s == 0).sum()) / max(float((y_train_s == 1).sum()), 1)
            clf_xgb_s = XGBClassifier(max_depth=5, n_estimators=100, learning_rate=0.1,
                                       scale_pos_weight=spw_s, random_state=42,
                                       eval_metric="logloss", verbosity=0, n_jobs=1)
            clf_xgb_s.fit(X_train_ss, y_train_s)
            surge_preds["y_score_xgb"].extend(clf_xgb_s.predict_proba(X_test_ss)[:, 1].tolist())

    return crash_preds, surge_preds


def fold_level_bootstrap(crash_preds, surge_preds, model_key, n_bootstrap=N_BOOTSTRAP, rng=None):
    """
    Fold-level bootstrap: resample LOPO folds, compute AUPRC for crash
    and surge separately, then take the difference.
    This accounts for the fact that crash and surge have different labels
    but share the same data splits.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    crash_y = np.array(crash_preds["y_true"])
    crash_scores = np.array(crash_preds[f"y_score_{model_key}"])
    crash_folds = np.array(crash_preds["fold_ids"])

    surge_y = np.array(surge_preds["y_true"])
    surge_scores = np.array(surge_preds[f"y_score_{model_key}"])
    surge_folds = np.array(surge_preds["fold_ids"])

    unique_folds = np.unique(crash_folds)

    crash_point = safe_auprc(crash_y, crash_scores)
    surge_point = safe_auprc(surge_y, surge_scores)
    point_diff = surge_point - crash_point

    boot_diffs = []
    for _ in range(n_bootstrap):
        # Resample folds with replacement
        sampled_folds = rng.choice(unique_folds, size=len(unique_folds), replace=True)

        # Build crash predictions from sampled folds
        crash_idx = np.concatenate([np.where(crash_folds == f)[0] for f in sampled_folds])
        cy, cs = crash_y[crash_idx], crash_scores[crash_idx]

        # Build surge predictions from sampled folds
        surge_idx = np.concatenate([np.where(surge_folds == f)[0] for f in sampled_folds])
        sy, ss = surge_y[surge_idx], surge_scores[surge_idx]

        if len(np.unique(cy)) < 2 or len(np.unique(sy)) < 2:
            continue

        c_auprc = average_precision_score(cy, cs)
        s_auprc = average_precision_score(sy, ss)
        boot_diffs.append(s_auprc - c_auprc)

    if len(boot_diffs) < 100:
        return point_diff, np.nan, np.nan, False, surge_point, crash_point

    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)
    significant = ci_lower > 0  # surge > crash

    return point_diff, ci_lower, ci_upper, significant, surge_point, crash_point


def run_experiment():
    t_total = time.time()

    print("=" * 72)
    print("EXPERIMENT 26: PAIRED BOOTSTRAP — CRASH vs SURGE DETECTABILITY")
    print("=" * 72)

    df = pd.read_csv(DATA_PATH)
    feat_cols = get_all_feature_cols(df)
    pids = sorted(df["pid"].unique())
    rng = np.random.RandomState(42)

    print(f"Loaded {len(df)} rows, {len(pids)} participants, {len(feat_cols)} features")
    print(f"Bootstrap iterations: {N_BOOTSTRAP}")

    results = {
        "experiment": "exp26_crash_vs_surge_paired_bootstrap",
        "n_bootstrap": N_BOOTSTRAP,
        "per_construct": {},
    }

    for construct in CONSTRUCTS:
        t_c = time.time()
        print(f"\n{'─' * 60}")
        print(f"Construct: {construct}")
        print(f"{'─' * 60}")

        print("  Collecting paired crash + surge predictions (LOPO)...")
        crash_preds, surge_preds = collect_paired_predictions(df, feat_cols, construct, pids)

        crash_rate = float(np.array(crash_preds["y_true"]).mean()) if crash_preds["y_true"] else 0.0
        surge_rate = float(np.array(surge_preds["y_true"]).mean()) if surge_preds["y_true"] else 0.0

        construct_result = {
            "crash_rate": round(crash_rate, 4),
            "surge_rate": round(surge_rate, 4),
            "n_crash_samples": len(crash_preds["y_true"]),
            "n_surge_samples": len(surge_preds["y_true"]),
        }

        for model_key in ["en", "xgb"]:
            if not crash_preds[f"y_score_{model_key}"] or not surge_preds[f"y_score_{model_key}"]:
                continue

            print(f"  Computing fold-level bootstrap ({model_key.upper()})...")
            diff, ci_lo, ci_hi, sig, surge_auprc, crash_auprc = fold_level_bootstrap(
                crash_preds, surge_preds, model_key, rng=rng
            )

            def _round(val):
                return round(float(val), 4) if not np.isnan(val) else None

            construct_result[model_key] = {
                "crash_AUPRC": _round(crash_auprc),
                "surge_AUPRC": _round(surge_auprc),
                "diff_surge_minus_crash": _round(diff),
                "CI_lower": _round(ci_lo),
                "CI_upper": _round(ci_hi),
                "surge_significantly_better": bool(sig),
            }

            sig_str = "SIGNIFICANT" if sig else "not significant"
            print(f"  {model_key.upper()}: surge={_round(surge_auprc)} crash={_round(crash_auprc)} "
                  f"diff={_round(diff)} [{_round(ci_lo)}, {_round(ci_hi)}] {sig_str}")

        results["per_construct"][construct] = construct_result
        print(f"  Time: {time.time()-t_c:.0f}s")

    # Summary
    print(f"\n{'=' * 72}")
    print("SUMMARY: SURGE vs CRASH DETECTABILITY (Paired Bootstrap)")
    print(f"{'=' * 72}")
    print(f"  {'Construct':<22} {'Model':>5} {'Surge':>8} {'Crash':>8} {'Diff':>8} {'95% CI':>20} {'Sig?':>12}")
    print(f"  {'─' * 85}")
    for c in CONSTRUCTS:
        r = results["per_construct"][c]
        for model_key in ["en", "xgb"]:
            if model_key not in r:
                continue
            m = r[model_key]
            sig_str = "YES" if m["surge_significantly_better"] else "no"
            print(f"  {c:<22} {model_key.upper():>5} {m['surge_AUPRC']:>8} {m['crash_AUPRC']:>8} "
                  f"{m['diff_surge_minus_crash']:>8} [{m['CI_lower']:.3f}, {m['CI_upper']:.3f}] {sig_str:>12}")

    total_time = time.time() - t_total
    print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")

    out_path = RESULTS_DIR / "exp26_crash_vs_surge_paired_bootstrap.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}")

    return results


if __name__ == "__main__":
    run_experiment()
