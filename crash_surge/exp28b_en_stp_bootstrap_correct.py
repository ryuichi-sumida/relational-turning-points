#!/usr/bin/env python3
"""
Experiment 28b: Bootstrap tests for EN+STP matching the ablation pipeline EXACTLY.

Key difference from exp28: includes stability selection (bootstrap Lasso)
before EN fitting, matching exp_ablation_detection.py line-for-line.

Produces per-fold predictions for both crash and surge under EN+STP,
then runs bootstrap tests for:
  1. AUPRC vs base rate (crash and surge separately)
  2. Paired surge-vs-crash comparison (fold-level bootstrap)
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

# Add parent dir so we can import from the ablation module
sys.path.insert(0, str(Path(__file__).parent))
from exp_ablation_detection import (
    get_session_features, get_temporal_features, compute_person_normalized,
    select_features, compute_event_labels, compute_systemic_labels,
    stability_selection, inner_cv_en, safe_auprc,
    EN_C_VALUES, EN_L1_RATIOS, N_INNER_FOLDS,
    STABILITY_N_BOOTSTRAP, STABILITY_SUBSAMPLE_FRAC, STABILITY_C_VALUES,
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


def collect_lopo_predictions(df, feat_cols, construct, event_type, pids, rng):
    """
    Run the EXACT ablation LOPO pipeline for one construct+event_type.
    Returns per-fold predictions (y_true, y_score, fold_ids).
    """
    preds = {"y_true": [], "y_score": [], "fold_ids": []}

    for fold_i, test_pid in enumerate(pids):
        train_pids = [p for p in pids if p != test_pid]

        # Labels (matches ablation pipeline exactly)
        if construct == "systemic":
            label_df = compute_systemic_labels(df, event_type, train_pids)
        else:
            label_df = compute_event_labels(df, construct, event_type, train_pids)

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

        # Stability selection (matches ablation exactly)
        sel_idx, sel_freq, sel_thr = stability_selection(
            X_train, y_train, feat_cols, rng,
        )
        X_train_sel = X_train[:, sel_idx]
        X_test_sel = X_test[:, sel_idx]

        # Nested CV for EN hyperparameters
        best_C, best_l1 = inner_cv_en(X_train_sel, y_train, pids_train, rng)

        # Fit EN
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


def bootstrap_auprc_vs_base(y_true, y_score, n_bootstrap=N_BOOTSTRAP, rng=None):
    """Bootstrap: AUPRC + CI, and test vs base rate."""
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(y_true)
    point = safe_auprc(y_true, y_score)
    base_rate = float(y_true.mean())

    boot_auprcs = []
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        auprc = average_precision_score(yt, ys)
        boot_auprcs.append(auprc)
        boot_diffs.append(auprc - float(yt.mean()))

    if len(boot_auprcs) < 100:
        return point, np.nan, np.nan, False, (None, None)

    ci_lo = np.percentile(boot_auprcs, 2.5)
    ci_hi = np.percentile(boot_auprcs, 97.5)
    diff_ci_lo = np.percentile(boot_diffs, 2.5)
    diff_ci_hi = np.percentile(boot_diffs, 97.5)
    sig = diff_ci_lo > 0

    return point, ci_lo, ci_hi, bool(sig), (_r(diff_ci_lo), _r(diff_ci_hi))


def fold_level_bootstrap(crash_preds, surge_preds, n_bootstrap=N_BOOTSTRAP, rng=None):
    """Paired fold-level bootstrap: surge AUPRC - crash AUPRC."""
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
    """Worker: run crash+surge for one target, compute bootstrap tests."""
    target, df, feat_cols, pids = args
    # Use SAME rng sequence as ablation: np.random.default_rng(42)
    # The ablation reinitializes rng per run_condition call
    rng_crash = np.random.default_rng(42)
    rng_surge = np.random.default_rng(42)
    t0 = time.time()

    crash_preds = collect_lopo_predictions(df, feat_cols, target, "crash", pids, rng_crash)
    surge_preds = collect_lopo_predictions(df, feat_cols, target, "surge", pids, rng_surge)

    crash_y = np.array(crash_preds["y_true"])
    surge_y = np.array(surge_preds["y_true"])
    crash_rate = float(crash_y.mean()) if len(crash_y) > 0 else 0.0
    surge_rate = float(surge_y.mean()) if len(surge_y) > 0 else 0.0

    # Bootstrap CIs + vs base rate
    rng_boot = np.random.default_rng(123)
    crash_auprc, crash_ci_lo, crash_ci_hi, crash_sig, crash_base_ci = \
        bootstrap_auprc_vs_base(crash_y, np.array(crash_preds["y_score"]), rng=rng_boot)
    surge_auprc, surge_ci_lo, surge_ci_hi, surge_sig, surge_base_ci = \
        bootstrap_auprc_vs_base(surge_y, np.array(surge_preds["y_score"]), rng=rng_boot)

    # Paired comparison
    diff, ci_lo, ci_hi, sig, s_auprc, c_auprc = \
        fold_level_bootstrap(crash_preds, surge_preds, rng=rng_boot)

    elapsed = time.time() - t0
    tag = "SIG" if sig else "n.s."
    print(f"  [{target}] done in {elapsed:.0f}s — "
          f"crash={_r(crash_auprc)} surge={_r(surge_auprc)} "
          f"diff={_r(diff)} [{_r(ci_lo)},{_r(ci_hi)}] {tag}  "
          f"crash>base:{'SIG' if crash_sig else 'n.s.'} surge>base:{'SIG' if surge_sig else 'n.s.'}")

    return target, {
        "crash_rate": _r(crash_rate),
        "surge_rate": _r(surge_rate),
        "n_crash_samples": len(crash_y),
        "n_surge_samples": len(surge_y),
        "crash": {
            "AUPRC": _r(crash_auprc), "CI_lower": _r(crash_ci_lo), "CI_upper": _r(crash_ci_hi),
            "vs_base_rate": {"CI_lower": crash_base_ci[0], "CI_upper": crash_base_ci[1],
                             "significant": crash_sig},
        },
        "surge": {
            "AUPRC": _r(surge_auprc), "CI_lower": _r(surge_ci_lo), "CI_upper": _r(surge_ci_hi),
            "vs_base_rate": {"CI_lower": surge_base_ci[0], "CI_upper": surge_base_ci[1],
                             "significant": surge_sig},
        },
        "paired_comparison": {
            "diff_surge_minus_crash": _r(diff), "CI_lower": _r(ci_lo), "CI_upper": _r(ci_hi),
            "surge_significantly_better": sig,
        },
    }


def run_experiment():
    t_total = time.time()
    print("=" * 72)
    print("EXPERIMENT 28b: EN+STP BOOTSTRAP (matching ablation pipeline)")
    print("  Includes: stability selection + nested CV + LOPO")
    print(f"  Running {len(TARGETS)} targets in parallel")
    print("=" * 72)

    # Feature prep (once)
    print("\nPreparing STP features...")
    t_feat = time.time()
    df = pd.read_csv(DATA_PATH)
    session_cols = get_session_features(df)
    temporal_cols = get_temporal_features(df)
    df, pn_cols = compute_person_normalized(df, session_cols)
    feat_cols = select_features(df, "STP", session_cols, temporal_cols, pn_cols)
    pids = sorted(df["pid"].unique())

    print(f"  Feature prep: {time.time()-t_feat:.1f}s")
    print(f"  Rows: {len(df)}, Participants: {len(pids)}, STP features: {len(feat_cols)}")

    results = {
        "experiment": "exp28b_en_stp_bootstrap_correct",
        "description": "Bootstrap tests for EN+STP with stability selection (matches ablation pipeline)",
        "n_bootstrap": N_BOOTSTRAP,
        "n_stp_features": len(feat_cols),
        "per_target": {},
    }

    # Parallel execution
    args_list = [(target, df, feat_cols, pids) for target in TARGETS]

    print(f"\nLaunching {len(TARGETS)} parallel workers...\n")
    with ProcessPoolExecutor(max_workers=len(TARGETS)) as executor:
        futures = {executor.submit(process_target, a): a[0] for a in args_list}
        for future in as_completed(futures):
            target, result = future.result()
            results["per_target"][target] = result

    # Summary
    print(f"\n{'=' * 72}")
    print("SUMMARY: EN+STP BOOTSTRAP (with stability selection)")
    print(f"{'=' * 72}")
    print(f"  {'Target':<22} {'Crash':>8} {'Surge':>8} {'Diff':>8} {'95% CI':>22} {'Sig?':>6}  "
          f"{'Cr>base':>7} {'Sr>base':>7}")
    print(f"  {'─' * 90}")
    for t in TARGETS:
        r = results["per_target"][t]
        pc = r["paired_comparison"]
        cb = "SIG" if r["crash"]["vs_base_rate"]["significant"] else "n.s."
        sb = "SIG" if r["surge"]["vs_base_rate"]["significant"] else "n.s."
        sig = "YES" if pc["surge_significantly_better"] else "no"
        print(f"  {t:<22} {r['crash']['AUPRC']:>8} {r['surge']['AUPRC']:>8} "
              f"{pc['diff_surge_minus_crash']:>8} [{pc['CI_lower']:.4f}, {pc['CI_upper']:.4f}] "
              f"{sig:>6}  {cb:>7} {sb:>7}")

    total_time = time.time() - t_total
    print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")

    out_path = RESULTS_DIR / "exp28b_en_stp_bootstrap.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
