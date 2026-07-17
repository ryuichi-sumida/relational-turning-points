#!/usr/bin/env python3
"""
Experiment 27: Proper Bootstrap Test — Detection vs Forecasting

For each construct × event type (crash/surge), collect LOPO predictions
for both same-session detection and next-session forecasting, then run
an independent bootstrap to test whether the AUPRC difference is significant.

This replaces the CI-overlap check in exp17/exp25 with a proper test.
Results are used for Figure 6 significance markers.
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


def build_labels(df, construct, train_pids, event_type="crash"):
    """Build binary labels for crash or surge events."""
    deltas = df.groupby("pid")[construct].diff()
    train_mask = df["pid"].isin(train_pids)
    train_deltas = deltas[train_mask].dropna()
    sd = train_deltas.std()

    binary = pd.Series(np.nan, index=df.index)
    valid = deltas.notna()
    if event_type == "crash":
        binary[valid & (deltas <= -sd)] = 1.0
        binary[valid & (deltas > -sd)] = 0.0
    else:  # surge
        binary[valid & (deltas >= sd)] = 1.0
        binary[valid & (deltas < sd)] = 0.0

    return binary, sd


def build_forecast_labels(df, construct, train_pids, event_type="crash"):
    """Features at session t, label at session t+1."""
    binary, sd = build_labels(df, construct, train_pids, event_type)

    forecast_label = pd.Series(np.nan, index=df.index)
    for pid, group in df.groupby("pid"):
        idx = group.index.tolist()
        for i in range(len(idx) - 1):
            forecast_label.iloc[idx[i]] = binary.iloc[idx[i + 1]]

    return forecast_label, sd


def collect_predictions(df, feat_cols, construct, pids, task="detection", event_type="crash"):
    """Run LOPO to collect per-sample predictions for EN + XGB."""
    en_y_true, en_y_score = [], []
    xgb_y_true, xgb_y_score = [], []

    for test_pid in pids:
        train_pids = [p for p in pids if p != test_pid]
        train_mask = df["pid"].isin(train_pids)
        test_mask = df["pid"] == test_pid

        if task == "detection":
            labels, sd = build_labels(df, construct, train_pids, event_type)
        else:
            labels, sd = build_forecast_labels(df, construct, train_pids, event_type)

        valid = labels.notna()
        train_valid = train_mask & valid
        test_valid = test_mask & valid

        if test_valid.sum() == 0:
            continue

        X_train = np.nan_to_num(df.loc[train_valid, feat_cols].values.astype(np.float32), nan=0.0)
        y_train = labels[train_valid].values.astype(int)
        X_test = np.nan_to_num(df.loc[test_valid, feat_cols].values.astype(np.float32), nan=0.0)
        y_test = labels[test_valid].values.astype(int)

        if len(np.unique(y_train)) < 2:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # EN
        clf_en = LogisticRegression(
            C=0.1, l1_ratio=0.5, penalty="elasticnet",
            solver="saga", class_weight="balanced",
            max_iter=5000, random_state=42
        )
        clf_en.fit(X_train_s, y_train)
        en_y_true.extend(y_test.tolist())
        en_y_score.extend(clf_en.predict_proba(X_test_s)[:, 1].tolist())

        # XGB
        if HAS_XGB:
            spw = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)
            clf_xgb = XGBClassifier(
                max_depth=5, n_estimators=100, learning_rate=0.1,
                scale_pos_weight=spw, random_state=42,
                eval_metric="logloss", verbosity=0, n_jobs=1
            )
            clf_xgb.fit(X_train_s, y_train)
            xgb_y_true.extend(y_test.tolist())
            xgb_y_score.extend(clf_xgb.predict_proba(X_test_s)[:, 1].tolist())

    return {
        "en": (np.array(en_y_true), np.array(en_y_score)),
        "xgb": (np.array(xgb_y_true), np.array(xgb_y_score)),
    }


def independent_bootstrap_comparison(y_true_a, y_score_a, y_true_b, y_score_b,
                                      n_bootstrap=N_BOOTSTRAP, rng=None):
    """
    Independent bootstrap test for AUPRC difference (A - B).
    Handles different sample sizes (detection vs forecasting).

    For each iteration:
    1. Resample A's predictions → compute AUPRC_A
    2. Resample B's predictions → compute AUPRC_B
    3. Record diff = AUPRC_A - AUPRC_B

    Returns: point_diff, ci_lower, ci_upper, significant, direction
    direction: 'a_better' if A sig > B, 'b_better' if B sig > A, 'neither'
    """
    if rng is None:
        rng = np.random.RandomState(42)

    y_true_a = np.array(y_true_a)
    y_score_a = np.array(y_score_a)
    y_true_b = np.array(y_true_b)
    y_score_b = np.array(y_score_b)

    point_a = safe_auprc(y_true_a, y_score_a)
    point_b = safe_auprc(y_true_b, y_score_b)
    point_diff = point_a - point_b

    n_a = len(y_true_a)
    n_b = len(y_true_b)

    boot_diffs = []
    for _ in range(n_bootstrap):
        # Resample A
        idx_a = rng.choice(n_a, size=n_a, replace=True)
        yt_a, ys_a = y_true_a[idx_a], y_score_a[idx_a]
        if len(np.unique(yt_a)) < 2:
            continue

        # Resample B
        idx_b = rng.choice(n_b, size=n_b, replace=True)
        yt_b, ys_b = y_true_b[idx_b], y_score_b[idx_b]
        if len(np.unique(yt_b)) < 2:
            continue

        auprc_a = average_precision_score(yt_a, ys_a)
        auprc_b = average_precision_score(yt_b, ys_b)
        boot_diffs.append(auprc_a - auprc_b)

    if len(boot_diffs) < 100:
        return point_diff, np.nan, np.nan, False, "neither"

    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)

    if ci_lower > 0:
        direction = "a_better"
        significant = True
    elif ci_upper < 0:
        direction = "b_better"
        significant = True
    else:
        direction = "neither"
        significant = False

    return point_diff, ci_lower, ci_upper, significant, direction


def run_experiment():
    t_total = time.time()

    print("=" * 72)
    print("EXPERIMENT 27: DETECTION vs FORECASTING — PROPER BOOTSTRAP")
    print("=" * 72)

    df = pd.read_csv(DATA_PATH)
    feat_cols = get_all_feature_cols(df)
    pids = sorted(df["pid"].unique())
    rng = np.random.RandomState(42)

    print(f"Loaded {len(df)} rows, {len(pids)} participants, {len(feat_cols)} features")
    print(f"Bootstrap iterations: {N_BOOTSTRAP}")

    results = {
        "experiment": "exp27_detection_vs_forecast_bootstrap",
        "n_bootstrap": N_BOOTSTRAP,
        "crash": {},
        "surge": {},
    }

    for event_type in ["crash", "surge"]:
        print(f"\n{'=' * 72}")
        print(f"EVENT TYPE: {event_type.upper()}")
        print(f"{'=' * 72}")

        for construct in CONSTRUCTS:
            t_construct = time.time()
            print(f"\n  {'─' * 56}")
            print(f"  {event_type}/{construct}")
            print(f"  {'─' * 56}")

            # Collect detection predictions
            print(f"    Collecting detection predictions...")
            det_preds = collect_predictions(df, feat_cols, construct, pids,
                                           task="detection", event_type=event_type)

            # Collect forecasting predictions
            print(f"    Collecting forecast predictions...")
            fore_preds = collect_predictions(df, feat_cols, construct, pids,
                                            task="forecasting", event_type=event_type)

            # For each model (EN, XGB), compare forecast vs detection
            entry = {}
            for model_name in ["en", "xgb"]:
                if model_name == "xgb" and not HAS_XGB:
                    continue

                det_true, det_score = det_preds[model_name]
                fore_true, fore_score = fore_preds[model_name]

                det_auprc = safe_auprc(det_true, det_score)
                fore_auprc = safe_auprc(fore_true, fore_score)

                print(f"    {model_name.upper()}: det={det_auprc:.4f} (n={len(det_true)}), "
                      f"fore={fore_auprc:.4f} (n={len(fore_true)})")

                # Independent bootstrap: forecast - detection
                diff, ci_lo, ci_hi, sig, direction = independent_bootstrap_comparison(
                    fore_true, fore_score, det_true, det_score,
                    n_bootstrap=N_BOOTSTRAP, rng=rng
                )

                print(f"    {model_name.upper()}: diff(fore-det)={diff:.4f} "
                      f"CI=[{ci_lo:.4f}, {ci_hi:.4f}] sig={sig} ({direction})")

                entry[model_name] = {
                    "detection_AUPRC": round(float(det_auprc), 4),
                    "detection_n": int(len(det_true)),
                    "forecast_AUPRC": round(float(fore_auprc), 4),
                    "forecast_n": int(len(fore_true)),
                    "diff_forecast_minus_detection": round(float(diff), 4),
                    "CI_lower": round(float(ci_lo), 4) if not np.isnan(ci_lo) else None,
                    "CI_upper": round(float(ci_hi), 4) if not np.isnan(ci_hi) else None,
                    "significant": bool(sig),
                    "direction": direction,
                    "forecast_sig_better": bool(sig and direction == "a_better"),
                    "detection_sig_better": bool(sig and direction == "b_better"),
                }

            # Pick best model summary
            best_model = "xgb" if HAS_XGB else "en"
            if HAS_XGB and "en" in entry and "xgb" in entry:
                # Use whichever model has the bigger absolute AUPRC (best of detection or forecast)
                en_best = max(entry["en"]["detection_AUPRC"], entry["en"]["forecast_AUPRC"])
                xgb_best = max(entry["xgb"]["detection_AUPRC"], entry["xgb"]["forecast_AUPRC"])
                best_model = "xgb" if xgb_best >= en_best else "en"

            entry["best_model"] = best_model
            entry["forecast_sig_better"] = entry[best_model]["forecast_sig_better"]
            entry["detection_sig_better"] = entry[best_model]["detection_sig_better"]

            elapsed = time.time() - t_construct
            print(f"    Time: {elapsed:.0f}s | Best model: {best_model}")

            results[event_type][construct] = entry

    # Summary
    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")

    for event_type in ["crash", "surge"]:
        print(f"\n  {event_type.upper()}:")
        print(f"  {'Construct':<22} {'Det AUPRC':>10} {'Fore AUPRC':>11} {'Diff':>8} {'Sig':>6} {'Direction':>18}")
        print(f"  {'─' * 78}")
        for c in CONSTRUCTS:
            e = results[event_type][c]
            bm = e["best_model"]
            d = e[bm]
            print(f"  {c:<22} {d['detection_AUPRC']:>10.4f} {d['forecast_AUPRC']:>11.4f} "
                  f"{d['diff_forecast_minus_detection']:>8.4f} {'*' if d['significant'] else '':>6} "
                  f"{d['direction']:>18}")

    total_time = time.time() - t_total
    print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")

    out_path = RESULTS_DIR / "exp27_detection_vs_forecast_bootstrap.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}")

    return results


if __name__ == "__main__":
    run_experiment()
