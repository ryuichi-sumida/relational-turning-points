"""
Experiment 1: Event Counts — Descriptive statistics of crashes, surges, and stable
transitions across all 5 relational constructs.

Reports:
  1. Per-construct event counts (crash / surge / stable) and rates
  2. Distributional properties of deltas (mean, SD, skewness, kurtosis)
  3. Per-participant breakdown (crashes/surges per person)
  4. LOPO threshold sensitivity (how much thresholds vary across folds)
  5. Sensitivity analysis at 0.75, 1.0, 1.25, 1.5 SD thresholds
"""

import sys
import json
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from config import CONSTRUCTS, RESULTS_DIR
from data_loader import load_features_and_labels, compute_deltas, get_lopo_folds


def run_experiment():
    # ---- Load data ----
    df = load_features_and_labels()
    print(f"Loaded {len(df)} rows, {df['pid'].nunique()} participants")
    print(f"Sessions per participant: {df.groupby('pid')['session_num'].count().describe().to_dict()}\n")

    results = {}

    # ==================================================================
    # 1. Per-construct event counts and rates
    # ==================================================================
    print("=" * 72)
    print("1. PER-CONSTRUCT EVENT COUNTS (1-SD threshold)")
    print("=" * 72)

    construct_stats = {}
    all_deltas = {}  # store for later use

    for construct in CONSTRUCTS:
        deltas = compute_deltas(df, construct)
        valid_deltas = deltas.dropna()
        sd = valid_deltas.std()
        mean = valid_deltas.mean()

        all_deltas[construct] = valid_deltas

        crash_mask = valid_deltas <= -sd
        surge_mask = valid_deltas >= sd
        stable_mask = ~crash_mask & ~surge_mask

        n_crash = crash_mask.sum()
        n_surge = surge_mask.sum()
        n_stable = stable_mask.sum()
        n_total = len(valid_deltas)

        construct_stats[construct] = {
            "n_total_transitions": int(n_total),
            "n_crash": int(n_crash),
            "n_surge": int(n_surge),
            "n_stable": int(n_stable),
            "crash_rate": round(float(n_crash / n_total), 4),
            "surge_rate": round(float(n_surge / n_total), 4),
            "stable_rate": round(float(n_stable / n_total), 4),
            "threshold_sd": round(float(sd), 4),
            "threshold_crash": round(float(-sd), 4),
            "threshold_surge": round(float(sd), 4),
        }

    # Print table
    header = f"{'Construct':<22} {'Total':>6} {'Crash':>6} {'Surge':>6} {'Stable':>6} {'Crash%':>8} {'Surge%':>8} {'SD':>8}"
    print(header)
    print("-" * len(header))
    for c in CONSTRUCTS:
        s = construct_stats[c]
        print(f"{c:<22} {s['n_total_transitions']:>6} {s['n_crash']:>6} {s['n_surge']:>6} "
              f"{s['n_stable']:>6} {s['crash_rate']:>8.1%} {s['surge_rate']:>8.1%} {s['threshold_sd']:>8.4f}")

    results["per_construct_events"] = construct_stats

    # ==================================================================
    # 2. Distributional properties of deltas
    # ==================================================================
    print("\n" + "=" * 72)
    print("2. DISTRIBUTIONAL PROPERTIES OF DELTAS")
    print("=" * 72)

    dist_stats = {}
    for construct in CONSTRUCTS:
        d = all_deltas[construct].values
        dist_stats[construct] = {
            "mean": round(float(np.mean(d)), 4),
            "std": round(float(np.std(d, ddof=1)), 4),
            "median": round(float(np.median(d)), 4),
            "min": round(float(np.min(d)), 4),
            "max": round(float(np.max(d)), 4),
            "skewness": round(float(sp_stats.skew(d)), 4),
            "kurtosis": round(float(sp_stats.kurtosis(d)), 4),
            "iqr": round(float(np.percentile(d, 75) - np.percentile(d, 25)), 4),
            "shapiro_W": None,
            "shapiro_p": None,
        }
        # Shapiro-Wilk test (limited to 5000 samples)
        if len(d) <= 5000:
            w, p = sp_stats.shapiro(d)
            dist_stats[construct]["shapiro_W"] = round(float(w), 4)
            dist_stats[construct]["shapiro_p"] = float(p)

    header = f"{'Construct':<22} {'Mean':>8} {'SD':>8} {'Median':>8} {'Skew':>8} {'Kurt':>8} {'IQR':>8}"
    print(header)
    print("-" * len(header))
    for c in CONSTRUCTS:
        s = dist_stats[c]
        print(f"{c:<22} {s['mean']:>8.4f} {s['std']:>8.4f} {s['median']:>8.4f} "
              f"{s['skewness']:>8.4f} {s['kurtosis']:>8.4f} {s['iqr']:>8.4f}")

    # Print Shapiro-Wilk results
    print("\nShapiro-Wilk normality test:")
    for c in CONSTRUCTS:
        s = dist_stats[c]
        if s["shapiro_W"] is not None:
            sig = "***" if s["shapiro_p"] < 0.001 else "**" if s["shapiro_p"] < 0.01 else "*" if s["shapiro_p"] < 0.05 else "ns"
            print(f"  {c:<22} W={s['shapiro_W']:.4f}  p={s['shapiro_p']:.6f}  {sig}")

    results["delta_distributions"] = dist_stats

    # ==================================================================
    # 3. Per-participant breakdown
    # ==================================================================
    print("\n" + "=" * 72)
    print("3. PER-PARTICIPANT BREAKDOWN")
    print("=" * 72)

    per_participant = {}
    for construct in CONSTRUCTS:
        deltas = compute_deltas(df, construct)
        sd = all_deltas[construct].std()

        pid_stats = {}
        for pid, group in df.groupby("pid"):
            pid_deltas = deltas.loc[group.index].dropna()
            n_transitions = len(pid_deltas)
            n_crash = (pid_deltas <= -sd).sum()
            n_surge = (pid_deltas >= sd).sum()
            pid_stats[str(int(pid))] = {
                "n_transitions": int(n_transitions),
                "n_crash": int(n_crash),
                "n_surge": int(n_surge),
                "crash_rate": round(float(n_crash / n_transitions), 4) if n_transitions > 0 else 0,
                "surge_rate": round(float(n_surge / n_transitions), 4) if n_transitions > 0 else 0,
            }
        per_participant[construct] = pid_stats

    # Summary table: aggregate across constructs
    print(f"\n{'PID':<8}", end="")
    for c in CONSTRUCTS:
        print(f" {c[:6]+'_C':>8} {c[:6]+'_S':>8}", end="")
    print(f" {'Total_C':>8} {'Total_S':>8}")
    print("-" * (8 + len(CONSTRUCTS) * 18 + 18))

    pids_sorted = sorted(df["pid"].unique())
    for pid in pids_sorted:
        total_c, total_s = 0, 0
        print(f"{int(pid):<8}", end="")
        for c in CONSTRUCTS:
            ps = per_participant[c][str(int(pid))]
            total_c += ps["n_crash"]
            total_s += ps["n_surge"]
            print(f" {ps['n_crash']:>8} {ps['n_surge']:>8}", end="")
        print(f" {total_c:>8} {total_s:>8}")

    # Distribution summary
    print("\nPer-participant crash count distribution (summed across constructs):")
    total_crashes_per_pid = []
    total_surges_per_pid = []
    for pid in pids_sorted:
        tc = sum(per_participant[c][str(int(pid))]["n_crash"] for c in CONSTRUCTS)
        ts = sum(per_participant[c][str(int(pid))]["n_surge"] for c in CONSTRUCTS)
        total_crashes_per_pid.append(tc)
        total_surges_per_pid.append(ts)

    tc_arr = np.array(total_crashes_per_pid)
    ts_arr = np.array(total_surges_per_pid)
    print(f"  Crashes: mean={tc_arr.mean():.2f}, SD={tc_arr.std():.2f}, "
          f"min={tc_arr.min()}, max={tc_arr.max()}, median={np.median(tc_arr):.1f}")
    print(f"  Surges:  mean={ts_arr.mean():.2f}, SD={ts_arr.std():.2f}, "
          f"min={ts_arr.min()}, max={ts_arr.max()}, median={np.median(ts_arr):.1f}")

    per_participant_summary = {
        "total_crashes_per_pid": {
            "mean": round(float(tc_arr.mean()), 2),
            "std": round(float(tc_arr.std()), 2),
            "min": int(tc_arr.min()),
            "max": int(tc_arr.max()),
            "median": round(float(np.median(tc_arr)), 1),
        },
        "total_surges_per_pid": {
            "mean": round(float(ts_arr.mean()), 2),
            "std": round(float(ts_arr.std()), 2),
            "min": int(ts_arr.min()),
            "max": int(ts_arr.max()),
            "median": round(float(np.median(ts_arr)), 1),
        },
    }

    results["per_participant"] = per_participant
    results["per_participant_summary"] = per_participant_summary

    # ==================================================================
    # 4. LOPO threshold sensitivity
    # ==================================================================
    print("\n" + "=" * 72)
    print("4. LOPO THRESHOLD SENSITIVITY")
    print("=" * 72)

    lopo_thresholds = {}
    for construct in CONSTRUCTS:
        fold_sds = []
        for test_pid, train_pids, train_mask, test_mask in get_lopo_folds(df):
            deltas = compute_deltas(df, construct)
            train_deltas = deltas[train_mask].dropna()
            fold_sd = train_deltas.std()
            fold_sds.append(float(fold_sd))

        fold_sds = np.array(fold_sds)
        global_sd = float(all_deltas[construct].std())

        lopo_thresholds[construct] = {
            "global_sd": round(global_sd, 4),
            "lopo_mean_sd": round(float(fold_sds.mean()), 4),
            "lopo_std_sd": round(float(fold_sds.std()), 4),
            "lopo_min_sd": round(float(fold_sds.min()), 4),
            "lopo_max_sd": round(float(fold_sds.max()), 4),
            "lopo_range_sd": round(float(fold_sds.max() - fold_sds.min()), 4),
            "lopo_cv_sd": round(float(fold_sds.std() / fold_sds.mean()), 4),
            "fold_sds": [round(x, 4) for x in fold_sds.tolist()],
        }

    header = f"{'Construct':<22} {'Global SD':>10} {'LOPO Mean':>10} {'LOPO Std':>10} {'LOPO Range':>11} {'CV':>8}"
    print(header)
    print("-" * len(header))
    for c in CONSTRUCTS:
        s = lopo_thresholds[c]
        print(f"{c:<22} {s['global_sd']:>10.4f} {s['lopo_mean_sd']:>10.4f} "
              f"{s['lopo_std_sd']:>10.4f} {s['lopo_range_sd']:>11.4f} {s['lopo_cv_sd']:>8.4f}")

    results["lopo_threshold_sensitivity"] = lopo_thresholds

    # ==================================================================
    # 5. Sensitivity analysis at multiple SD thresholds
    # ==================================================================
    print("\n" + "=" * 72)
    print("5. SENSITIVITY ANALYSIS AT MULTIPLE SD THRESHOLDS")
    print("=" * 72)

    sd_multipliers = [0.75, 1.0, 1.25, 1.5]
    sensitivity = {}

    for construct in CONSTRUCTS:
        valid_deltas = all_deltas[construct]
        sd = valid_deltas.std()
        n_total = len(valid_deltas)

        sensitivity[construct] = {}
        for mult in sd_multipliers:
            threshold = mult * sd
            n_crash = (valid_deltas <= -threshold).sum()
            n_surge = (valid_deltas >= threshold).sum()
            n_stable = n_total - n_crash - n_surge

            sensitivity[construct][str(mult)] = {
                "threshold_value": round(float(threshold), 4),
                "n_crash": int(n_crash),
                "n_surge": int(n_surge),
                "n_stable": int(n_stable),
                "crash_rate": round(float(n_crash / n_total), 4),
                "surge_rate": round(float(n_surge / n_total), 4),
            }

    # Print sensitivity table
    for construct in CONSTRUCTS:
        print(f"\n  {construct}:")
        header = f"    {'Multiplier':>10} {'Threshold':>10} {'Crash':>6} {'Surge':>6} {'Stable':>6} {'Crash%':>8} {'Surge%':>8}"
        print(header)
        print("    " + "-" * (len(header) - 4))
        for mult in sd_multipliers:
            s = sensitivity[construct][str(mult)]
            print(f"    {mult:>10.2f} {s['threshold_value']:>10.4f} {s['n_crash']:>6} {s['n_surge']:>6} "
                  f"{s['n_stable']:>6} {s['crash_rate']:>8.1%} {s['surge_rate']:>8.1%}")

    results["sensitivity_analysis"] = sensitivity

    # ==================================================================
    # Save results
    # ==================================================================
    output_path = RESULTS_DIR / "exp1_event_counts.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_experiment()
