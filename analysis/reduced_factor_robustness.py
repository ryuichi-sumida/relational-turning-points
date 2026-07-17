"""
Reduced-Factor Robustness Check for Crash-Surge Asymmetry
==========================================================
Addresses discriminant validity concern: high HTMT ratios suggest some constructs
may partially overlap (Familiarity-ConvQual=0.97, Memory-ConvQual=0.98).

This script tests whether the crash-surge asymmetry holds under a reduced
3-factor structure that merges the problematic pairs:
  Factor 1 ("Impression"): Familiarity + Conv. Quality  (Q1,Q2,Q7,Q8)
  Factor 2 ("Depth"):      Social Penetration + Enjoyment (Q3,Q4,Q9,Q10)
  Factor 3 ("Memory"):     Perceived Memory              (Q5,Q6)

We also test a single composite ("Overall Relational Quality": all 10 items).

For each factor structure, we recompute:
  1. Crash/surge event rates
  2. Cross-session contagion (lift ratios) -- 3-factor only
  3. Recovery / persistence rates
  4. Summary comparison with the 5-factor results
"""

import pandas as pd
import numpy as np
from itertools import product
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = str(Path(__file__).resolve().parents[1] / "data" / "user_assessment_labels.csv")
OUTPUT_PATH = str(Path(__file__).resolve().parent / "reduced_factor_robustness_results.json")

# ── Factor definitions ──────────────────────────────────────────────────────
FIVE_FACTORS = {
    "Familiarity": ["Q1", "Q2"],
    "SocPen": ["Q3", "Q4"],
    "Memory": ["Q5", "Q6"],
    "ConvQual": ["Q7", "Q8"],
    "Enjoyment": ["Q9", "Q10"],
}

THREE_FACTORS = {
    "Impression": ["Q1", "Q2", "Q7", "Q8"],      # Familiarity + ConvQual
    "Depth": ["Q3", "Q4", "Q9", "Q10"],           # SocPen + Enjoyment
    "Memory": ["Q5", "Q6"],                        # unchanged
}

ONE_FACTOR = {
    "Overall": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10"],
}


def load_and_compute(df, factor_def):
    """Compute factor scores as item means."""
    scores = pd.DataFrame()
    scores["user_id"] = df["user_id"]
    scores["session"] = df["session"]
    for name, items in factor_def.items():
        scores[name] = df[items].mean(axis=1)
    return scores.sort_values(["user_id", "session"]).reset_index(drop=True)


def compute_deltas(scores, constructs):
    """Compute session-to-session deltas for each construct."""
    rows = []
    for uid, grp in scores.groupby("user_id"):
        grp = grp.sort_values("session")
        for i in range(1, len(grp)):
            row = {"user_id": uid, "session": int(grp.iloc[i]["session"])}
            for c in constructs:
                row[f"delta_{c}"] = grp.iloc[i][c] - grp.iloc[i - 1][c]
                row[c] = grp.iloc[i][c]
                row[f"{c}_prev"] = grp.iloc[i - 1][c]
            rows.append(row)
    return pd.DataFrame(rows)


def label_events_lopo(deltas_df, constructs):
    """Label crash/surge events using LOPO thresholds (1-SD)."""
    users = deltas_df["user_id"].unique()
    event_rows = []
    for uid in users:
        others = deltas_df[deltas_df["user_id"] != uid]
        held = deltas_df[deltas_df["user_id"] == uid]
        for _, row in held.iterrows():
            erow = {"user_id": row["user_id"], "session": row["session"]}
            for c in constructs:
                col = f"delta_{c}"
                mu = others[col].mean()
                sigma = others[col].std()
                d = row[col]
                erow[f"crash_{c}"] = int(d < mu - sigma)
                erow[f"surge_{c}"] = int(d > mu + sigma)
            event_rows.append(erow)
    return pd.DataFrame(event_rows)


def event_rates(events_df, constructs):
    """Compute crash/surge rates per construct."""
    rates = {}
    n = len(events_df)
    for c in constructs:
        rates[c] = {
            "crash_rate": float(events_df[f"crash_{c}"].sum() / n),
            "surge_rate": float(events_df[f"surge_{c}"].sum() / n),
            "n_crashes": int(events_df[f"crash_{c}"].sum()),
            "n_surges": int(events_df[f"surge_{c}"].sum()),
        }
    return rates


def cross_session_contagion(events_df, constructs, event_type="crash"):
    """Compute cross-session contagion lift ratios."""
    prefix = event_type
    # Sort
    events_df = events_df.sort_values(["user_id", "session"]).reset_index(drop=True)

    # Create next-session events
    next_events = []
    for uid, grp in events_df.groupby("user_id"):
        grp = grp.sort_values("session")
        for i in range(len(grp) - 1):
            row = {"user_id": uid}
            for c in constructs:
                row[f"{prefix}_{c}_t"] = grp.iloc[i][f"{prefix}_{c}"]
                row[f"{prefix}_{c}_t1"] = grp.iloc[i + 1][f"{prefix}_{c}"]
            next_events.append(row)
    ne_df = pd.DataFrame(next_events)

    if len(ne_df) == 0:
        return {}

    # Compute lift: P(B crashes at t+1 | A crashes at t) / P(B crashes at t+1)
    lift_matrix = {}
    for a in constructs:
        for b in constructs:
            if a == b:
                continue
            col_a = f"{prefix}_{a}_t"
            col_b = f"{prefix}_{b}_t1"
            base_rate = ne_df[col_b].mean()
            mask = ne_df[col_a] == 1
            if mask.sum() == 0 or base_rate == 0:
                continue
            cond_rate = ne_df.loc[mask, col_b].mean()
            lift = cond_rate / base_rate
            lift_matrix[f"{a}->{b}"] = {
                "lift": float(lift),
                "cond_rate": float(cond_rate),
                "base_rate": float(base_rate),
                "n_given": int(mask.sum()),
            }
    return lift_matrix


def count_high_lift_pairs(lift_dict, threshold=1.5):
    """Count pairs with lift > threshold."""
    return sum(1 for v in lift_dict.values() if v["lift"] > threshold)


def recovery_persistence(events_df, deltas_df, constructs):
    """Compute crash recovery and surge persistence rates."""
    events_df = events_df.sort_values(["user_id", "session"]).reset_index(drop=True)
    deltas_df = deltas_df.sort_values(["user_id", "session"]).reset_index(drop=True)

    results = {}
    for c in constructs:
        # Crash recovery: did the construct go back up next session?
        crash_recovery = []
        surge_persist = []

        for uid, grp in events_df.groupby("user_id"):
            grp = grp.sort_values("session")
            d_grp = deltas_df[deltas_df["user_id"] == uid].sort_values("session")

            for i in range(len(grp) - 1):
                sess_curr = grp.iloc[i]["session"]
                sess_next = grp.iloc[i + 1]["session"]

                # Check crash recovery
                if grp.iloc[i][f"crash_{c}"] == 1:
                    next_delta_row = d_grp[d_grp["session"] == sess_next]
                    if len(next_delta_row) > 0:
                        next_delta = next_delta_row.iloc[0][f"delta_{c}"]
                        crash_recovery.append(1 if next_delta > 0 else 0)

                # Check surge persistence
                if grp.iloc[i][f"surge_{c}"] == 1:
                    next_delta_row = d_grp[d_grp["session"] == sess_next]
                    if len(next_delta_row) > 0:
                        next_delta = next_delta_row.iloc[0][f"delta_{c}"]
                        surge_persist.append(1 if next_delta >= 0 else 0)

        results[c] = {
            "crash_recovery_rate": float(np.mean(crash_recovery)) if crash_recovery else None,
            "surge_persistence_rate": float(np.mean(surge_persist)) if surge_persist else None,
            "n_crash_followups": len(crash_recovery),
            "n_surge_followups": len(surge_persist),
        }
    return results


def same_session_cooccurrence(events_df, constructs, event_type="crash"):
    """Compute same-session co-occurrence rates."""
    prefix = event_type
    n_constructs = len(constructs)
    cooc = np.zeros((n_constructs, n_constructs))

    for i, ci in enumerate(constructs):
        for j, cj in enumerate(constructs):
            if i == j:
                cooc[i, j] = 1.0
                continue
            mask_i = events_df[f"{prefix}_{ci}"] == 1
            if mask_i.sum() == 0:
                cooc[i, j] = 0.0
                continue
            cooc[i, j] = events_df.loc[mask_i, f"{prefix}_{cj}"].mean()

    # Mean off-diagonal
    off_diag = []
    for i in range(n_constructs):
        for j in range(n_constructs):
            if i != j:
                off_diag.append(cooc[i, j])
    mean_off = float(np.mean(off_diag)) if off_diag else 0.0

    return {
        "matrix": {f"{constructs[i]}-{constructs[j]}": float(cooc[i, j])
                    for i in range(n_constructs) for j in range(n_constructs) if i != j},
        "mean_off_diagonal": mean_off,
    }


def run_analysis(df, factor_def, label):
    """Run full crash-surge analysis for a given factor structure."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    constructs = list(factor_def.keys())
    scores = load_and_compute(df, factor_def)
    deltas = compute_deltas(scores, constructs)
    events = label_events_lopo(deltas, constructs)

    n_transitions = len(events)
    print(f"  Transitions: {n_transitions}")

    # 1. Event rates
    rates = event_rates(events, constructs)
    print(f"\n  Event Rates:")
    for c, r in rates.items():
        print(f"    {c}: crash={r['crash_rate']:.1%} ({r['n_crashes']}), "
              f"surge={r['surge_rate']:.1%} ({r['n_surges']})")

    # 2. Same-session co-occurrence
    cooc_crash = same_session_cooccurrence(events, constructs, "crash")
    cooc_surge = same_session_cooccurrence(events, constructs, "surge")
    print(f"\n  Same-Session Co-occurrence (mean off-diagonal):")
    print(f"    Crashes: {cooc_crash['mean_off_diagonal']:.3f}")
    print(f"    Surges:  {cooc_surge['mean_off_diagonal']:.3f}")

    # 3. Cross-session contagion
    lift_crash = cross_session_contagion(events, constructs, "crash")
    lift_surge = cross_session_contagion(events, constructs, "surge")
    n_high_crash = count_high_lift_pairs(lift_crash)
    n_high_surge = count_high_lift_pairs(lift_surge)

    total_pairs = len(constructs) * (len(constructs) - 1)
    print(f"\n  Cross-Session Contagion (lift > 1.5x):")
    print(f"    Crashes: {n_high_crash}/{total_pairs} pairs")
    print(f"    Surges:  {n_high_surge}/{total_pairs} pairs")
    if lift_crash:
        parts = [f"{k}={v['lift']:.2f}" for k, v in sorted(lift_crash.items(), key=lambda x: -x[1]["lift"])]
        print(f"    Crash lift values: {', '.join(parts)}")
    if lift_surge:
        parts = [f"{k}={v['lift']:.2f}" for k, v in sorted(lift_surge.items(), key=lambda x: -x[1]["lift"])]
        print(f"    Surge lift values: {', '.join(parts)}")

    # 4. Recovery / persistence
    rec_pers = recovery_persistence(events, deltas, constructs)
    print(f"\n  Recovery / Persistence:")
    crash_recs = []
    surge_pers = []
    for c, rp in rec_pers.items():
        cr = rp['crash_recovery_rate']
        sp = rp['surge_persistence_rate']
        cr_str = f"{cr:.0%}" if cr is not None else "N/A"
        sp_str = f"{sp:.0%}" if sp is not None else "N/A"
        print(f"    {c}: crash_recovery={cr_str} (n={rp['n_crash_followups']}), "
              f"surge_persist={sp_str} (n={rp['n_surge_followups']})")
        if cr is not None:
            crash_recs.append(cr)
        if sp is not None:
            surge_pers.append(sp)

    avg_crash_rec = float(np.mean(crash_recs)) if crash_recs else None
    avg_surge_per = float(np.mean(surge_pers)) if surge_pers else None
    print(f"    Average: crash_recovery={avg_crash_rec:.0%}, surge_persist={avg_surge_per:.0%}")

    # 5. Asymmetry summary
    print(f"\n  ASYMMETRY SUMMARY:")
    print(f"    Contagion: crashes {n_high_crash} > surges {n_high_surge} high-lift pairs  "
          f"=> {'YES' if n_high_crash > n_high_surge else 'NO'}")
    if avg_surge_per is not None and avg_crash_rec is not None:
        print(f"    Persistence: surge persist {avg_surge_per:.0%} vs crash recover {avg_crash_rec:.0%}  "
              f"=> {'YES (surges stickier)' if avg_surge_per > avg_crash_rec else 'NO'}")

    return {
        "label": label,
        "n_constructs": len(constructs),
        "constructs": constructs,
        "n_transitions": n_transitions,
        "event_rates": rates,
        "cooccurrence_crash": cooc_crash,
        "cooccurrence_surge": cooc_surge,
        "contagion_crash": lift_crash,
        "contagion_surge": lift_surge,
        "n_high_lift_crash": n_high_crash,
        "n_high_lift_surge": n_high_surge,
        "recovery_persistence": rec_pers,
        "avg_crash_recovery": avg_crash_rec,
        "avg_surge_persistence": avg_surge_per,
        "asymmetry_contagion_holds": n_high_crash > n_high_surge,
        "asymmetry_persistence_holds": (avg_surge_per > avg_crash_rec) if (avg_surge_per and avg_crash_rec) else None,
    }


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} observations, {df['user_id'].nunique()} participants")

    results = {}

    # Original 5-factor
    results["5_factor"] = run_analysis(df, FIVE_FACTORS, "5-Factor (Original)")

    # Reduced 3-factor
    results["3_factor"] = run_analysis(df, THREE_FACTORS, "3-Factor (Merged high-HTMT pairs)")

    # Single composite
    results["1_factor"] = run_analysis(df, ONE_FACTOR, "1-Factor (Overall composite)")

    # ── Comparison table ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  COMPARISON ACROSS FACTOR STRUCTURES")
    print(f"{'='*70}")
    print(f"\n{'Structure':<15} {'Contagion Asym.':<20} {'Persist. Asym.':<20} {'Both Hold?':<12}")
    print("-" * 67)
    for key in ["5_factor", "3_factor", "1_factor"]:
        r = results[key]
        cont = f"crash {r['n_high_lift_crash']} > surge {r['n_high_lift_surge']}"
        pers_holds = r["asymmetry_persistence_holds"]
        if r["avg_surge_persistence"] is not None and r["avg_crash_recovery"] is not None:
            pers = f"{r['avg_surge_persistence']:.0%} > {r['avg_crash_recovery']:.0%}"
        else:
            pers = "N/A"
        both = "YES" if (r["asymmetry_contagion_holds"] and pers_holds) else "PARTIAL" if (r["asymmetry_contagion_holds"] or pers_holds) else "NO"
        print(f"{key:<15} {cont:<20} {pers:<20} {both:<12}")

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
