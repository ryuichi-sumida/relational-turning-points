#!/usr/bin/env python3
"""
Experiment 29: Bootstrap tests for EN+STP FORECASTING (surge vs crash).
Analogous to exp28b but for forecasting task (features from t predict event at t→t+1).
"""

import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exp_ablation_detection import (
    get_session_features, get_temporal_features, compute_person_normalized,
    select_features, compute_event_labels, compute_systemic_labels,
    stability_selection, inner_cv_en, safe_auprc,
)

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "full_features_with_temporals.csv"
RESULTS_DIR = BASE / "crash_surge" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONSTRUCTS = ["familiarity", "social_penetration", "memory", "conversational", "enjoyment"]
TARGETS = CONSTRUCTS + ["systemic"]
N_BOOTSTRAP = 2000


def _r(val):
    return round(float(val), 4) if val is not None and not np.isnan(val) else None


def collect_lopo_predictions_forecast(df, feat_cols, construct, event_type, pids, rng):
    """Run LOPO pipeline for FORECASTING: features from t predict event at t→t+1."""
    preds = {"y_true": [], "y_score": [], "fold_ids": []}

    for fold_i, test_pid in enumerate(pids):
        train_pids = [p for p in pids if p != test_pid]

        if construct == "systemic":
            label_df = compute_systemic_labels(df, event_type, train_pids)
        else:
            label_df = compute_event_labels(df, construct, event_type, train_pids)

        # FORECASTING: shift labels back by 1 session
        label_df = label_df.copy()
        label_df["session_num"] = label_df["session_num"] - 1
        label_df = label_df[label_df["session_num"] >= 1]

        df_lab = df.merge(label_df, on=["pid", "session_num"], how="inner")
        train_df = df_lab[df_lab["pid"] != test_pid]
        test_df = df_lab[df_lab["pid"] == test_pid]

        if test_df.shape[0] == 0:
            continue

        X_train = np.nan_to_num(train_df[feat_cols].values.astype(np.float64), nan=0.0)
        y_train = train_df["label"].values.astype(int)
        pids_train = train_df["pid"].values
        X_test = np.nan_to_num(test_df[feat_cols].values.astype(np.float64), nan=0.0)
        y_test = test_df["label"].values.astype(int)

        if y_train.sum() < 2 or len(np.unique(y_train)) < 2:
            continue

        sel_idx, sel_freq, sel_thr = stability_selection(X_train, y_train, feat_cols, rng)
        X_train_sel = X_train[:, sel_idx]
        X_test_sel = X_test[:, sel_idx]

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
            preds["y_true"].extend(y_test.tolist())
            preds["y_score"].extend(en_prob.tolist())
            preds["fold_ids"].extend([fold_i] * len(y_test))
        except Exception:
            continue

    return preds


def fold_level_bootstrap(crash_preds, surge_preds, n_bootstrap=N_BOOTSTRAP, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    crash_y = np.array(crash_preds["y_true"])
    crash_scores = np.array(crash_preds["y_score"])
    crash_folds = np.array(crash_preds["fold_ids"])
    surge_y = np.array(surge_preds["y_true"])
    surge_scores = np.array(surge_preds["y_score"])
    surge_folds = np.array(surge_preds["fold_ids"])

    unique_folds = np.unique(crash_folds)
    crash_point = safe_auprc(crash_y, crash_scores)
    surge_point = safe_auprc(surge_y, surge_scores)
    point_diff = surge_point - crash_point

    boot_diffs = []
    for _ in range(n_bootstrap):
        sampled_folds = rng.choice(unique_folds, size=len(unique_folds), replace=True)
        crash_idx = np.concatenate([np.where(crash_folds == f)[0] for f in sampled_folds])
        surge_idx = np.concatenate([np.where(surge_folds == f)[0] for f in sampled_folds])
        cy, cs = crash_y[crash_idx], crash_scores[crash_idx]
        sy, ss = surge_y[surge_idx], surge_scores[surge_idx]
        if len(np.unique(cy)) < 2 or len(np.unique(sy)) < 2:
            continue
        boot_diffs.append(average_precision_score(sy, ss) - average_precision_score(cy, cs))

    if len(boot_diffs) < 100:
        return point_diff, np.nan, np.nan, False, surge_point, crash_point

    ci_lo = np.percentile(boot_diffs, 2.5)
    ci_hi = np.percentile(boot_diffs, 97.5)
    return point_diff, ci_lo, ci_hi, bool(ci_lo > 0), surge_point, crash_point


def process_target(args):
    target, df, feat_cols, pids = args
    rng_crash = np.random.default_rng(42)
    rng_surge = np.random.default_rng(42)
    t0 = time.time()

    print(f"  [{target}] Starting forecasting LOPO...")
    crash_preds = collect_lopo_predictions_forecast(df, feat_cols, target, "crash", pids, rng_crash)
    surge_preds = collect_lopo_predictions_forecast(df, feat_cols, target, "surge", pids, rng_surge)

    crash_y = np.array(crash_preds["y_true"])
    surge_y = np.array(surge_preds["y_true"])
    crash_rate = float(crash_y.mean()) if len(crash_y) > 0 else 0.0
    surge_rate = float(surge_y.mean()) if len(surge_y) > 0 else 0.0

    rng_boot = np.random.default_rng(123)
    diff, ci_lo, ci_hi, sig, surge_auprc, crash_auprc = fold_level_bootstrap(
        crash_preds, surge_preds, rng=rng_boot
    )

    elapsed = time.time() - t0
    print(f"  [{target}] Done in {elapsed:.0f}s. Crash={_r(crash_auprc)} Surge={_r(surge_auprc)} "
          f"Diff={_r(diff)} CI=[{_r(ci_lo)},{_r(ci_hi)}] Sig={sig}")

    return target, {
        "crash_rate": _r(crash_rate),
        "surge_rate": _r(surge_rate),
        "n_crash_samples": len(crash_y),
        "n_surge_samples": len(surge_y),
        "crash_AUPRC": _r(crash_auprc),
        "surge_AUPRC": _r(surge_auprc),
        "paired_comparison": {
            "diff_surge_minus_crash": _r(diff),
            "CI_lower": _r(ci_lo),
            "CI_upper": _r(ci_hi),
            "surge_significantly_better": sig,
        },
        "elapsed_seconds": round(elapsed, 1),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="*", default=None,
                        help="Subset of targets to run (for distributed execution)")
    args = parser.parse_args()

    t_start = time.time()
    print("Experiment 29: Forecasting surge-vs-crash bootstrap (EN+STP)")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    session_cols = get_session_features(df)
    temporal_cols = get_temporal_features(df)
    df, pn_cols = compute_person_normalized(df, session_cols)
    feat_cols = select_features(df, "STP", session_cols, temporal_cols, pn_cols)
    pids = sorted(df["pid"].unique())

    targets = args.targets if args.targets else TARGETS
    print(f"Data: {len(df)} rows, {len(pids)} participants, {len(feat_cols)} STP features")
    print(f"Targets: {targets}")
    print()

    results = {"experiment": "exp29_forecast_bootstrap", "task": "forecast",
               "n_bootstrap": N_BOOTSTRAP, "per_target": {}}

    args_list = [(t, df, feat_cols, pids) for t in targets]

    with ProcessPoolExecutor(max_workers=min(6, len(targets))) as executor:
        futures = {executor.submit(process_target, a): a[0] for a in args_list}
        for future in as_completed(futures):
            target, result = future.result()
            results["per_target"][target] = result

    results["total_elapsed_seconds"] = round(time.time() - t_start, 1)

    suffix = "_" + "_".join(targets) if len(targets) < len(TARGETS) else ""
    out_path = RESULTS_DIR / f"exp29_forecast_bootstrap{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Forecasting surge vs crash (EN+STP)")
    print("=" * 60)
    for t in targets:
        r = results["per_target"].get(t, {})
        pc = r.get("paired_comparison", {})
        sig_str = "YES" if pc.get("surge_significantly_better") else "no"
        print(f"  {t:20s}  Crash={r.get('crash_AUPRC','?'):>6}  Surge={r.get('surge_AUPRC','?'):>6}  "
              f"Sig={sig_str}  CI=[{pc.get('CI_lower','?')},{pc.get('CI_upper','?')}]")


if __name__ == "__main__":
    main()
