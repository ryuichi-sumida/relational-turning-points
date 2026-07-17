#!/usr/bin/env python3
"""
Experiment 20: Surge Contagion & Recovery (Persistence)

Parallels Exp2 (crash contagion/recovery) but for positive events:
  1. Cross-construct surge co-occurrence at time t
  2. Lagged contagion: surge at t -> surge at t+1 (lift over base rate)
  3. Surge persistence: how long do surges "stick" before decaying back?
  4. Cascade patterns: multi-construct surge combinations
"""

import json
import warnings

import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter

warnings.filterwarnings("ignore")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONSTRUCTS, RESULTS_DIR
from data_loader import load_features_and_labels, compute_deltas


def run_experiment():
    print("=" * 72)
    print("EXPERIMENT 20: SURGE CONTAGION & RECOVERY (PERSISTENCE)")
    print("=" * 72)

    df = load_features_and_labels()
    print(f"Loaded {len(df)} rows, {df['pid'].nunique()} participants\n")

    results = {}

    # Precompute deltas and labels
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

    label_df = pd.DataFrame(labels)

    # ==================================================================
    # 1. Surge co-occurrence at time t
    # ==================================================================
    print("=" * 72)
    print("1. SURGE CO-OCCURRENCE (same transition)")
    print("=" * 72)

    cooccurrence = {}
    for c_a in CONSTRUCTS:
        cooccurrence[c_a] = {}
        for c_b in CONSTRUCTS:
            valid_mask = label_df[c_a].notna() & label_df[c_b].notna()
            a_surge = (label_df[c_a] == "surge") & valid_mask
            b_surge = (label_df[c_b] == "surge") & valid_mask
            n_a = int(a_surge.sum())
            n_both = int((a_surge & b_surge).sum())
            cooccurrence[c_a][c_b] = round(n_both / n_a, 4) if n_a > 0 else 0.0

    print("\nP(B surges | A surges):")
    header = "A \\ B"
    print(f"{header:<22}" + "".join(f" {c[:8]:>10}" for c in CONSTRUCTS))
    print("-" * 80)
    for c_a in CONSTRUCTS:
        row = f"{c_a:<22}" + "".join(f" {cooccurrence[c_a][c_b]:>10.3f}" for c_b in CONSTRUCTS)
        print(row)

    results["surge_cooccurrence"] = cooccurrence

    # ==================================================================
    # 2. Lagged contagion: surge at t -> surge at t+1
    # ==================================================================
    print("\n" + "=" * 72)
    print("2. LAGGED SURGE CONTAGION (A surges at t -> B surges at t+1)")
    print("=" * 72)

    lagged_labels = {}
    for construct in CONSTRUCTS:
        lagged_labels[construct] = df.groupby("pid").apply(
            lambda g: labels[construct].loc[g.index].shift(-1),
            include_groups=False
        ).droplevel(0).sort_index()

    # Base rates of surge at t+1
    base_rates = {}
    for c in CONSTRUCTS:
        valid_next = lagged_labels[c].notna()
        if valid_next.sum() > 0:
            base_rates[c] = round(float((lagged_labels[c][valid_next] == "surge").mean()), 4)
        else:
            base_rates[c] = 0.0

    lagged_cooccurrence = {}
    lagged_lift = {}
    for c_a in CONSTRUCTS:
        lagged_cooccurrence[c_a] = {}
        lagged_lift[c_a] = {}
        for c_b in CONSTRUCTS:
            a_surge_now = label_df[c_a] == "surge"
            b_surge_next = lagged_labels[c_b] == "surge"
            valid = label_df[c_a].notna() & lagged_labels[c_b].notna()

            a_valid = a_surge_now & valid
            both = a_valid & b_surge_next

            n_a = int(a_valid.sum())
            n_both = int(both.sum())
            prob = float(n_both / n_a) if n_a > 0 else 0.0
            lift = prob / base_rates[c_b] if base_rates[c_b] > 0 else 0.0

            lagged_cooccurrence[c_a][c_b] = round(prob, 4)
            lagged_lift[c_a][c_b] = round(lift, 2)

    print("\nLift (lagged P / base rate) for surges:")
    header2 = "A(t) / B(t+1)"
    print(f"{header2:<22}" + "".join(f" {c[:8]:>10}" for c in CONSTRUCTS))
    print("-" * 80)
    for c_a in CONSTRUCTS:
        row = f"{c_a:<22}" + "".join(f" {lagged_lift[c_a][c_b]:>10.2f}" for c_b in CONSTRUCTS)
        print(row)

    print(f"\n{'Base rate':<22}" + "".join(f" {base_rates[c]:>10.3f}" for c in CONSTRUCTS))

    # Notable lift values
    print("\nNotable lagged lift values (>1.5x):")
    for c_a in CONSTRUCTS:
        for c_b in CONSTRUCTS:
            if c_a != c_b and lagged_lift[c_a][c_b] > 1.5:
                print(f"  {c_a} surge at t -> {c_b} surge at t+1: {lagged_lift[c_a][c_b]}x")

    results["lagged_surge_contagion"] = lagged_cooccurrence
    results["lagged_surge_lift"] = lagged_lift
    results["surge_base_rates"] = base_rates

    # ==================================================================
    # 3. Surge persistence: sessions until surge decays back
    # ==================================================================
    print("\n" + "=" * 72)
    print("3. SURGE PERSISTENCE (sessions until rating drops back to pre-surge level)")
    print("=" * 72)

    persistence_times = {c: [] for c in CONSTRUCTS}

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
                            persistence_times[construct].append(j - i)
                            decayed = True
                            break
                    if not decayed:
                        persistence_times[construct].append(np.nan)  # Persisted to end

    persistence_summary = {}
    print(f"\n{'Construct':<22} {'N_surge':>8} {'Persisted':>10} {'Persist%':>9} {'Mean_t':>8} {'Med_t':>8}")
    print("-" * 70)

    for construct in CONSTRUCTS:
        pt = persistence_times[construct]
        n_total = len(pt)
        persisted = [x for x in pt if isinstance(x, float) and np.isnan(x)]
        decayed = [x for x in pt if not (isinstance(x, float) and np.isnan(x))]
        n_persisted = len(persisted)
        n_decayed = len(decayed)
        persist_rate = n_persisted / n_total if n_total > 0 else 0

        if decayed:
            arr = np.array(decayed)
            persistence_summary[construct] = {
                "n_surges": n_total,
                "n_persisted": n_persisted,
                "n_decayed": n_decayed,
                "persistence_rate": round(persist_rate, 4),
                "mean_decay_sessions": round(float(arr.mean()), 2),
                "median_decay_sessions": round(float(np.median(arr)), 2),
            }
            s = persistence_summary[construct]
            print(f"{construct:<22} {n_total:>8} {n_persisted:>10} {persist_rate:>9.1%} "
                  f"{s['mean_decay_sessions']:>8.2f} {s['median_decay_sessions']:>8.2f}")
        else:
            persistence_summary[construct] = {
                "n_surges": n_total, "n_persisted": n_persisted, "n_decayed": 0,
                "persistence_rate": round(persist_rate, 4),
                "mean_decay_sessions": None, "median_decay_sessions": None,
            }
            print(f"{construct:<22} {n_total:>8} {n_persisted:>10} {persist_rate:>9.1%} {'N/A':>8} {'N/A':>8}")

    results["surge_persistence"] = persistence_summary

    # ==================================================================
    # 4. Multi-construct surge patterns
    # ==================================================================
    print("\n" + "=" * 72)
    print("4. MULTI-CONSTRUCT SURGE PATTERNS")
    print("=" * 72)

    valid_rows = label_df.dropna(how="any")
    surge_counter = Counter()
    surge_count_dist = Counter()

    for idx in valid_rows.index:
        surging = [c for c in CONSTRUCTS if label_df.loc[idx, c] == "surge"]
        surge_count_dist[len(surging)] += 1
        if len(surging) >= 2:
            surge_counter[tuple(sorted(surging))] += 1

    n_valid = len(valid_rows)
    print(f"\nDistribution of simultaneous surge counts (n={n_valid}):")
    for k in sorted(surge_count_dist.keys()):
        print(f"  {k} constructs surging: {surge_count_dist[k]} ({surge_count_dist[k]/n_valid:.1%})")

    print(f"\nMost common multi-construct surge combinations:")
    surge_list = []
    for combo, count in surge_counter.most_common():
        rate = count / n_valid
        print(f"  {' + '.join(combo):<55} {count:>4} ({rate:.1%})")
        surge_list.append({"constructs": list(combo), "count": count, "rate": round(rate, 4)})

    results["surge_count_distribution"] = {str(k): {"count": v, "rate": round(v/n_valid, 4)}
                                            for k, v in sorted(surge_count_dist.items())}
    results["surge_cascade_patterns"] = surge_list

    # ==================================================================
    # 5. Compare crash vs surge contagion patterns
    # ==================================================================
    print("\n" + "=" * 72)
    print("5. CRASH vs SURGE COMPARISON")
    print("=" * 72)

    # Crash co-occurrence for comparison
    crash_cooccurrence = {}
    for c_a in CONSTRUCTS:
        crash_cooccurrence[c_a] = {}
        for c_b in CONSTRUCTS:
            valid_mask = label_df[c_a].notna() & label_df[c_b].notna()
            a_crash = (label_df[c_a] == "crash") & valid_mask
            b_crash = (label_df[c_b] == "crash") & valid_mask
            n_a = int(a_crash.sum())
            n_both = int((a_crash & b_crash).sum())
            crash_cooccurrence[c_a][c_b] = round(n_both / n_a, 4) if n_a > 0 else 0.0

    print("\nMean off-diagonal co-occurrence:")
    for event_type, matrix in [("Crash", crash_cooccurrence), ("Surge", cooccurrence)]:
        vals = []
        for c_a in CONSTRUCTS:
            for c_b in CONSTRUCTS:
                if c_a != c_b:
                    vals.append(matrix[c_a][c_b])
        print(f"  {event_type}: {np.mean(vals):.3f} (range {np.min(vals):.3f}-{np.max(vals):.3f})")

    results["crash_vs_surge_comparison"] = {
        "crash_mean_cooccurrence": round(float(np.mean([crash_cooccurrence[a][b] for a in CONSTRUCTS for b in CONSTRUCTS if a != b])), 4),
        "surge_mean_cooccurrence": round(float(np.mean([cooccurrence[a][b] for a in CONSTRUCTS for b in CONSTRUCTS if a != b])), 4),
    }

    # Save
    out_path = RESULTS_DIR / "exp20_surge_contagion_recovery.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    return results


if __name__ == "__main__":
    run_experiment()
