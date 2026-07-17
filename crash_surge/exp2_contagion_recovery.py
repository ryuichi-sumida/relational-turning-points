"""
Experiment 2: Contagion + Recovery — Analyzes cross-construct crash propagation,
lagged contagion, recovery dynamics, cascade patterns, and recovery asymmetry.

Reports:
  1. Cross-construct contagion: 5x5 co-occurrence matrix at time t
  2. Asymmetric (lagged) contagion: construct A crash at t -> construct B crash at t+1
  3. Recovery analysis: sessions until pre-crash level restored, per construct
  4. Cascade patterns: most common multi-construct crash combinations (2+)
  5. Recovery asymmetry: compare recovery speeds across constructs
"""

import sys
import json
import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter

from config import CONSTRUCTS, RESULTS_DIR
from data_loader import load_features_and_labels, compute_deltas


def _make_header(row_label, col_labels):
    """Build a table header without backslashes in f-strings."""
    return f"{row_label:<22}" + "".join(f" {c[:8]:>10}" for c in col_labels)


def run_experiment():
    # ---- Load data ----
    df = load_features_and_labels()
    print(f"Loaded {len(df)} rows, {df['pid'].nunique()} participants\n")

    results = {}

    # ---- Precompute deltas and labels for all constructs ----
    deltas = {}
    labels = {}
    sds = {}

    for construct in CONSTRUCTS:
        d = compute_deltas(df, construct)
        deltas[construct] = d
        sd = d.dropna().std()
        sds[construct] = sd

        lab = pd.Series(pd.NA, index=df.index, dtype="string")
        valid = d.notna()
        lab[valid & (d <= -sd)] = "crash"
        lab[valid & (d >= sd)] = "surge"
        lab[valid & ~(d <= -sd) & ~(d >= sd)] = "stable"
        labels[construct] = lab

    # Combine labels into a DataFrame for convenience
    label_df = pd.DataFrame(labels)

    # ==================================================================
    # 1. Cross-construct contagion (co-occurrence at time t)
    # ==================================================================
    print("=" * 72)
    print("1. CROSS-CONSTRUCT CONTAGION (co-occurrence at time t)")
    print("=" * 72)

    # For each pair (A, B): P(B crashes | A crashes at same t)
    cooccurrence = {}
    cooccurrence_counts = {}

    for c_a in CONSTRUCTS:
        cooccurrence[c_a] = {}
        cooccurrence_counts[c_a] = {}
        for c_b in CONSTRUCTS:
            valid_mask = label_df[c_a].notna() & label_df[c_b].notna()
            a_crash = (label_df[c_a] == "crash") & valid_mask
            b_crash = (label_df[c_b] == "crash") & valid_mask

            n_a_crash = int(a_crash.sum())
            n_both_crash = int((a_crash & b_crash).sum())

            prob = float(n_both_crash / n_a_crash) if n_a_crash > 0 else 0.0

            cooccurrence[c_a][c_b] = round(prob, 4)
            cooccurrence_counts[c_a][c_b] = {
                "n_a_crash": n_a_crash,
                "n_both_crash": n_both_crash,
            }

    # Print matrix
    print("\nP(B crashes | A crashes) at same transition:")
    print(_make_header("A \\\\ B", CONSTRUCTS))
    print("-" * 80)
    for c_a in CONSTRUCTS:
        row = f"{c_a:<22}"
        for c_b in CONSTRUCTS:
            row += f" {cooccurrence[c_a][c_b]:>10.3f}"
        print(row)

    results["cooccurrence_matrix"] = cooccurrence
    results["cooccurrence_counts"] = cooccurrence_counts

    # ==================================================================
    # 2. Asymmetric (lagged) contagion: A crash at t -> B crash at t+1
    # ==================================================================
    print("\n" + "=" * 72)
    print("2. ASYMMETRIC (LAGGED) CONTAGION: A crash at t -> B crash at t+1")
    print("=" * 72)

    # For each participant, shift labels by 1 to get next transition's labels
    lagged_labels = {}
    for construct in CONSTRUCTS:
        lagged_labels[construct] = df.groupby("pid").apply(
            lambda g: labels[construct].loc[g.index].shift(-1),
            include_groups=False
        ).droplevel(0).sort_index()

    lagged_cooccurrence = {}
    lagged_counts = {}

    for c_a in CONSTRUCTS:
        lagged_cooccurrence[c_a] = {}
        lagged_counts[c_a] = {}
        for c_b in CONSTRUCTS:
            a_crash_now = label_df[c_a] == "crash"
            b_crash_next = lagged_labels[c_b] == "crash"
            valid = label_df[c_a].notna() & lagged_labels[c_b].notna()

            a_crash_valid = a_crash_now & valid
            both = a_crash_valid & b_crash_next & valid

            n_a = int(a_crash_valid.sum())
            n_both = int(both.sum())

            prob = float(n_both / n_a) if n_a > 0 else 0.0
            lagged_cooccurrence[c_a][c_b] = round(prob, 4)
            lagged_counts[c_a][c_b] = {
                "n_a_crash_t": n_a,
                "n_b_crash_t1": n_both,
            }

    # Base rates of crash at t+1
    base_rates = {}
    for c in CONSTRUCTS:
        valid_next = lagged_labels[c].notna()
        if valid_next.sum() > 0:
            base_rates[c] = round(float((lagged_labels[c][valid_next] == "crash").mean()), 4)
        else:
            base_rates[c] = 0.0

    print("\nP(B crashes at t+1 | A crashes at t):")
    print(_make_header("A(t) / B(t+1)", CONSTRUCTS))
    print("-" * 80)
    for c_a in CONSTRUCTS:
        row = f"{c_a:<22}"
        for c_b in CONSTRUCTS:
            row += f" {lagged_cooccurrence[c_a][c_b]:>10.3f}"
        print(row)

    print(f"\n{'Base rate':<22}", end="")
    for c in CONSTRUCTS:
        print(f" {base_rates[c]:>10.3f}", end="")
    print()

    # Compute lift (lagged probability / base rate)
    print("\nLift (lagged P / base rate):")
    print(_make_header("A(t) / B(t+1)", CONSTRUCTS))
    print("-" * 80)
    lagged_lift = {}
    for c_a in CONSTRUCTS:
        lagged_lift[c_a] = {}
        row = f"{c_a:<22}"
        for c_b in CONSTRUCTS:
            if base_rates[c_b] > 0:
                lift = lagged_cooccurrence[c_a][c_b] / base_rates[c_b]
            else:
                lift = 0.0
            lagged_lift[c_a][c_b] = round(lift, 2)
            row += f" {lift:>10.2f}"
        print(row)

    results["lagged_contagion"] = lagged_cooccurrence
    results["lagged_counts"] = lagged_counts
    results["lagged_base_rates"] = base_rates
    results["lagged_lift"] = lagged_lift

    # ==================================================================
    # 3. Recovery analysis: sessions until pre-crash level restored
    # ==================================================================
    print("\n" + "=" * 72)
    print("3. RECOVERY ANALYSIS (sessions to return to pre-crash level)")
    print("=" * 72)

    recovery_times = {c: [] for c in CONSTRUCTS}
    recovery_details = {c: [] for c in CONSTRUCTS}

    for pid, group in df.groupby("pid"):
        group = group.sort_values("session_num").reset_index(drop=True)
        for construct in CONSTRUCTS:
            ratings = group[construct].values
            d = group[construct].diff()
            sd = sds[construct]

            for i in range(1, len(ratings)):
                delta_val = d.iloc[i]
                if pd.isna(delta_val):
                    continue
                if delta_val <= -sd:
                    # This is a crash: pre-crash level is ratings[i-1]
                    pre_crash_level = ratings[i - 1]
                    crash_level = ratings[i]

                    # Look forward for recovery
                    recovered = False
                    recovery_session_count = None
                    for j in range(i + 1, len(ratings)):
                        if ratings[j] >= pre_crash_level:
                            recovery_session_count = j - i
                            recovered = True
                            break

                    if recovered:
                        recovery_times[construct].append(recovery_session_count)
                    else:
                        recovery_times[construct].append(np.nan)

                    recovery_details[construct].append({
                        "pid": int(pid),
                        "crash_session": int(group["session_num"].iloc[i]),
                        "pre_crash_level": round(float(pre_crash_level), 2),
                        "crash_level": round(float(crash_level), 2),
                        "drop_size": round(float(pre_crash_level - crash_level), 2),
                        "recovered": recovered,
                        "recovery_sessions": int(recovery_session_count) if recovered else None,
                    })

    # Summary
    recovery_summary = {}
    print(f"\n{'Construct':<22} {'N_crash':>8} {'Recovered':>10} {'Recov%':>8} "
          f"{'Mean_t':>8} {'Med_t':>8} {'SD_t':>8} {'Min_t':>6} {'Max_t':>6}")
    print("-" * 100)

    for construct in CONSTRUCTS:
        rt = recovery_times[construct]
        n_total = len(rt)
        rt_valid = [x for x in rt if not (isinstance(x, float) and np.isnan(x))]
        n_recovered = len(rt_valid)
        n_unrecovered = n_total - n_recovered

        if rt_valid:
            rt_arr = np.array(rt_valid)
            recovery_summary[construct] = {
                "n_crashes": n_total,
                "n_recovered": n_recovered,
                "n_unrecovered": n_unrecovered,
                "recovery_rate": round(float(n_recovered / n_total), 4) if n_total > 0 else 0,
                "mean_recovery_sessions": round(float(rt_arr.mean()), 2),
                "median_recovery_sessions": round(float(np.median(rt_arr)), 2),
                "std_recovery_sessions": round(float(rt_arr.std()), 2),
                "min_recovery_sessions": int(rt_arr.min()),
                "max_recovery_sessions": int(rt_arr.max()),
            }
            s = recovery_summary[construct]
            print(f"{construct:<22} {s['n_crashes']:>8} {s['n_recovered']:>10} "
                  f"{s['recovery_rate']:>8.1%} {s['mean_recovery_sessions']:>8.2f} "
                  f"{s['median_recovery_sessions']:>8.2f} {s['std_recovery_sessions']:>8.2f} "
                  f"{s['min_recovery_sessions']:>6} {s['max_recovery_sessions']:>6}")
        else:
            recovery_summary[construct] = {
                "n_crashes": n_total,
                "n_recovered": 0,
                "n_unrecovered": n_unrecovered,
                "recovery_rate": 0,
                "mean_recovery_sessions": None,
                "median_recovery_sessions": None,
                "std_recovery_sessions": None,
                "min_recovery_sessions": None,
                "max_recovery_sessions": None,
            }
            print(f"{construct:<22} {n_total:>8} {0:>10} {'0.0%':>8} "
                  f"{'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>6} {'N/A':>6}")

    results["recovery_summary"] = recovery_summary
    results["recovery_details"] = {c: recovery_details[c] for c in CONSTRUCTS}

    # ==================================================================
    # 4. Cascade patterns: multi-construct crash combinations (2+)
    # ==================================================================
    print("\n" + "=" * 72)
    print("4. CASCADE PATTERNS (multi-construct crash combinations)")
    print("=" * 72)

    # For each transition (row), identify which constructs crash simultaneously
    valid_rows = label_df.dropna(how="any")

    cascade_counter = Counter()
    crash_count_dist = Counter()

    for idx in valid_rows.index:
        crashing = [c for c in CONSTRUCTS if label_df.loc[idx, c] == "crash"]
        crash_count_dist[len(crashing)] += 1

        if len(crashing) >= 2:
            combo = tuple(sorted(crashing))
            cascade_counter[combo] += 1

    n_valid = len(valid_rows)
    print(f"\nDistribution of simultaneous crash counts (n={n_valid} transitions):")
    print(f"  {'# Constructs crashing':>25} {'Count':>8} {'Rate':>8}")
    print("  " + "-" * 45)
    for k in sorted(crash_count_dist.keys()):
        print(f"  {k:>25} {crash_count_dist[k]:>8} {crash_count_dist[k]/n_valid:>8.1%}")

    crash_count_distribution = {str(k): {"count": v, "rate": round(v / n_valid, 4)}
                                for k, v in sorted(crash_count_dist.items())}

    # Most common cascade combinations
    print(f"\nMost common multi-construct crash combinations (2+ constructs):")
    print(f"  {'Combination':<55} {'Count':>6} {'Rate':>8}")
    print("  " + "-" * 72)

    cascade_list = []
    for combo, count in cascade_counter.most_common():
        rate = count / n_valid
        combo_str = " + ".join(combo)
        print(f"  {combo_str:<55} {count:>6} {rate:>8.1%}")
        cascade_list.append({
            "constructs": list(combo),
            "count": count,
            "rate": round(rate, 4),
        })

    results["crash_count_distribution"] = crash_count_distribution
    results["cascade_patterns"] = cascade_list

    # ==================================================================
    # 5. Recovery asymmetry: compare recovery speed across constructs
    # ==================================================================
    print("\n" + "=" * 72)
    print("5. RECOVERY ASYMMETRY")
    print("=" * 72)

    print("\nRecovery speed comparison (mean sessions to recover):")
    recovery_ranking = []
    for c in CONSTRUCTS:
        s = recovery_summary[c]
        if s["mean_recovery_sessions"] is not None:
            recovery_ranking.append((c, s["mean_recovery_sessions"],
                                     s["median_recovery_sessions"],
                                     s["recovery_rate"], s["n_crashes"]))

    recovery_ranking.sort(key=lambda x: x[1])

    print(f"  {'Rank':>4} {'Construct':<22} {'Mean':>8} {'Median':>8} {'Recov%':>8} {'N':>6}")
    print("  " + "-" * 60)
    for rank, (c, mean_t, med_t, rec_rate, n) in enumerate(recovery_ranking, 1):
        print(f"  {rank:>4} {c:<22} {mean_t:>8.2f} {med_t:>8.2f} {rec_rate:>8.1%} {n:>6}")

    # Statistical tests
    from scipy.stats import kruskal, mannwhitneyu

    valid_recovery_per_construct = {}
    for c in CONSTRUCTS:
        rt = recovery_times[c]
        valid = [x for x in rt if not (isinstance(x, float) and np.isnan(x))]
        if valid:
            valid_recovery_per_construct[c] = valid

    kw_result = None
    if len(valid_recovery_per_construct) >= 2:
        groups = list(valid_recovery_per_construct.values())
        if all(len(g) >= 2 for g in groups):
            stat, p = kruskal(*groups)
            kw_result = {"H_statistic": round(float(stat), 4), "p_value": float(p)}
            print(f"\nKruskal-Wallis test across constructs: H={stat:.4f}, p={p:.6f}")

            # Pairwise Mann-Whitney U tests
            print("\nPairwise Mann-Whitney U tests:")
            construct_names = list(valid_recovery_per_construct.keys())
            pairwise_tests = {}
            for i, c_a in enumerate(construct_names):
                for c_b in construct_names[i + 1:]:
                    u_stat, u_p = mannwhitneyu(
                        valid_recovery_per_construct[c_a],
                        valid_recovery_per_construct[c_b],
                        alternative="two-sided"
                    )
                    sig = "***" if u_p < 0.001 else "**" if u_p < 0.01 else "*" if u_p < 0.05 else "ns"
                    print(f"  {c_a} vs {c_b}: U={u_stat:.1f}, p={u_p:.4f} {sig}")
                    pair_key = c_a + "_vs_" + c_b
                    pairwise_tests[pair_key] = {
                        "U_statistic": round(float(u_stat), 2),
                        "p_value": round(float(u_p), 6),
                    }

            results["recovery_kruskal_wallis"] = kw_result
            results["recovery_pairwise_tests"] = pairwise_tests

    # Recovery time distribution per construct
    recovery_distributions = {}
    for c in CONSTRUCTS:
        if c in valid_recovery_per_construct:
            rt_arr = np.array(valid_recovery_per_construct[c])
            recovery_distributions[c] = {
                "values": [int(x) for x in rt_arr],
                "mean": round(float(rt_arr.mean()), 2),
                "median": round(float(np.median(rt_arr)), 2),
                "std": round(float(rt_arr.std()), 2),
                "q25": round(float(np.percentile(rt_arr, 25)), 2),
                "q75": round(float(np.percentile(rt_arr, 75)), 2),
            }
    results["recovery_distributions"] = recovery_distributions

    # Surge decay analysis (bonus)
    print("\n--- Surge decay analysis (bonus) ---")
    surge_decay = {c: [] for c in CONSTRUCTS}

    for pid, group in df.groupby("pid"):
        group = group.sort_values("session_num").reset_index(drop=True)
        for construct in CONSTRUCTS:
            ratings = group[construct].values
            d = group[construct].diff()
            sd = sds[construct]

            for i in range(1, len(ratings)):
                delta_val = d.iloc[i]
                if pd.isna(delta_val):
                    continue
                if delta_val >= sd:
                    pre_surge_level = ratings[i - 1]
                    decayed = False
                    for j in range(i + 1, len(ratings)):
                        if ratings[j] <= pre_surge_level:
                            surge_decay[construct].append(j - i)
                            decayed = True
                            break
                    if not decayed:
                        surge_decay[construct].append(np.nan)

    surge_decay_summary = {}
    print(f"\n{'Construct':<22} {'N_surge':>8} {'Decayed':>8} {'Decay%':>8} {'Mean_t':>8} {'Med_t':>8}")
    print("-" * 70)
    for construct in CONSTRUCTS:
        sd_list = surge_decay[construct]
        n_total = len(sd_list)
        valid = [x for x in sd_list if not (isinstance(x, float) and np.isnan(x))]
        n_decayed = len(valid)

        if valid:
            arr = np.array(valid)
            surge_decay_summary[construct] = {
                "n_surges": n_total,
                "n_decayed": n_decayed,
                "decay_rate": round(float(n_decayed / n_total), 4) if n_total > 0 else 0,
                "mean_decay_sessions": round(float(arr.mean()), 2),
                "median_decay_sessions": round(float(np.median(arr)), 2),
            }
            s = surge_decay_summary[construct]
            print(f"{construct:<22} {s['n_surges']:>8} {s['n_decayed']:>8} "
                  f"{s['decay_rate']:>8.1%} {s['mean_decay_sessions']:>8.2f} "
                  f"{s['median_decay_sessions']:>8.2f}")
        else:
            surge_decay_summary[construct] = {
                "n_surges": n_total,
                "n_decayed": 0,
                "decay_rate": 0,
                "mean_decay_sessions": None,
                "median_decay_sessions": None,
            }
            print(f"{construct:<22} {n_total:>8} {0:>8} {'0.0%':>8} {'N/A':>8} {'N/A':>8}")

    results["surge_decay_summary"] = surge_decay_summary

    # ==================================================================
    # Save results
    # ==================================================================
    output_path = RESULTS_DIR / "exp2_contagion_recovery.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_experiment()
