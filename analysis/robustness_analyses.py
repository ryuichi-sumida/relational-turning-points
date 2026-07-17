#!/usr/bin/env python3
"""
Five Robustness Analyses for ICMI '26 Paper
=============================================
Addresses the two main reviewer concerns: small sample size (N=24) and
discriminant validity (high HTMT ratios between constructs).

Analyses:
  1. Delta-level HTMT — discriminant validity on session-to-session changes
  2. 3-Factor robustness — contagion, persistence, AND detectability under merged constructs
  3. Permutation test for crash-surge asymmetry
  4. Bootstrap CIs on contagion lift ratios
  5. Leave-one-participant-out sensitivity (jackknife)

Usage:
  uv run python analysis/robustness_analyses.py
"""

import json
import time
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not available, XGB models will be skipped")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "user_assessment_labels.csv"
FEATURES_PATH = BASE / "data" / "full_features_with_temporals.csv"
OUTPUT_DIR = BASE / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Construct definitions ──────────────────────────────────────────────────
FIVE_FACTORS = {
    "Familiarity": ["Q1", "Q2"],
    "SocPen": ["Q3", "Q4"],
    "Memory": ["Q5", "Q6"],
    "ConvQual": ["Q7", "Q8"],
    "Enjoyment": ["Q9", "Q10"],
}

THREE_FACTORS = {
    "Impression": ["Q1", "Q2", "Q7", "Q8"],
    "Depth": ["Q3", "Q4", "Q9", "Q10"],
    "Memory": ["Q5", "Q6"],
}

CONSTRUCTS_5 = ["familiarity", "social_penetration", "memory", "conversational", "enjoyment"]
NON_FEATURE_COLS = {"pid", "session_num", "n_prior_sessions"} | set(CONSTRUCTS_5)


def load_ratings():
    return pd.read_csv(DATA_PATH)


def load_features():
    return pd.read_csv(FEATURES_PATH)


def compute_factor_scores(df, factor_def):
    scores = pd.DataFrame()
    scores["user_id"] = df["user_id"]
    scores["session"] = df["session"]
    for name, items in factor_def.items():
        scores[name] = df[items].mean(axis=1)
    return scores.sort_values(["user_id", "session"]).reset_index(drop=True)


def compute_deltas_df(scores, constructs):
    rows = []
    for uid, grp in scores.groupby("user_id"):
        grp = grp.sort_values("session")
        for i in range(1, len(grp)):
            row = {"user_id": uid, "session": int(grp.iloc[i]["session"])}
            for c in constructs:
                row[f"delta_{c}"] = grp.iloc[i][c] - grp.iloc[i - 1][c]
                row[c] = grp.iloc[i][c]
            rows.append(row)
    return pd.DataFrame(rows)


def label_events_lopo(deltas_df, constructs):
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


def cross_session_contagion(events_df, constructs, event_type="crash"):
    prefix = event_type
    events_df = events_df.sort_values(["user_id", "session"]).reset_index(drop=True)
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


def recovery_persistence(events_df, deltas_df, constructs):
    """
    Level-based recovery/persistence (matching paper Section 4.2.4):
    - Crash recovery: does the construct return to pre-crash LEVEL at session t+1?
    - Surge persistence: does the construct maintain at/above surge LEVEL at session t+1?
    """
    events_df = events_df.sort_values(["user_id", "session"]).reset_index(drop=True)
    deltas_df = deltas_df.sort_values(["user_id", "session"]).reset_index(drop=True)
    results = {}
    for c in constructs:
        crash_recovery = []
        surge_persist = []
        for uid, grp in events_df.groupby("user_id"):
            grp = grp.sort_values("session")
            d_grp = deltas_df[deltas_df["user_id"] == uid].sort_values("session")
            for i in range(len(grp) - 1):
                sess_curr = grp.iloc[i]["session"]
                sess_next = grp.iloc[i + 1]["session"]
                curr_row = d_grp[d_grp["session"] == sess_curr]
                next_row = d_grp[d_grp["session"] == sess_next]
                if len(curr_row) == 0 or len(next_row) == 0:
                    continue

                if grp.iloc[i][f"crash_{c}"] == 1:
                    # Pre-crash level = current level - delta (i.e., previous session's level)
                    # current_level = curr_row[c], pre_crash = current - delta = prev session
                    # next_level = next_row[c]
                    # Recovery = next_level >= pre_crash_level
                    pre_crash_level = curr_row.iloc[0][c] - curr_row.iloc[0][f"delta_{c}"]
                    next_level = next_row.iloc[0][c]
                    crash_recovery.append(1 if next_level >= pre_crash_level else 0)

                if grp.iloc[i][f"surge_{c}"] == 1:
                    # Surge level = current level
                    # Persistence = next_level >= surge_level
                    surge_level = curr_row.iloc[0][c]
                    next_level = next_row.iloc[0][c]
                    surge_persist.append(1 if next_level >= surge_level else 0)

        results[c] = {
            "crash_recovery_rate": float(np.mean(crash_recovery)) if crash_recovery else None,
            "surge_persistence_rate": float(np.mean(surge_persist)) if surge_persist else None,
            "n_crash": len(crash_recovery),
            "n_surge": len(surge_persist),
        }
    return results


def safe_auprc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_prob)


# ============================================================================
# ANALYSIS 1: Delta-Level HTMT
# ============================================================================
def analysis_1_delta_htmt(df_raw):
    """
    Compute HTMT ratios on session-to-session deltas (not raw scores).
    If delta-level correlations are lower, it shows constructs behave
    distinctly where it matters for the crash/surge analysis.
    """
    print("\n" + "=" * 72)
    print("  ANALYSIS 1: Delta-Level HTMT (Discriminant Validity on Changes)")
    print("=" * 72)

    constructs = list(FIVE_FACTORS.keys())
    items_per_construct = FIVE_FACTORS

    # Compute deltas for each Q item
    q_cols = [f"Q{i}" for i in range(1, 11)]
    df_sorted = df_raw.sort_values(["user_id", "session"]).reset_index(drop=True)
    delta_items = df_sorted.groupby("user_id")[q_cols].diff()
    delta_items = delta_items.dropna()

    # Also compute raw-level HTMT for comparison
    raw_items = df_raw[[f"Q{i}" for i in range(1, 11)]].copy()

    results = {"delta_htmt": {}, "raw_htmt": {}, "comparison": {}}

    for level_name, item_df in [("raw", raw_items), ("delta", delta_items)]:
        htmt_key = f"{level_name}_htmt"
        print(f"\n  {level_name.upper()}-level HTMT ratios:")
        print(f"  {'Pair':<35} {'HTMT':>8}")
        print(f"  {'-'*45}")

        for (c1, c2) in combinations(constructs, 2):
            items_1 = items_per_construct[c1]
            items_2 = items_per_construct[c2]

            # Between-construct correlations (hetero-trait, hetero-method)
            between_corrs = []
            for i1 in items_1:
                for i2 in items_2:
                    if i1 in item_df.columns and i2 in item_df.columns:
                        r = item_df[i1].corr(item_df[i2])
                        if not np.isnan(r):
                            between_corrs.append(abs(r))

            # Within-construct correlations (mono-trait, hetero-method)
            within_1 = []
            for i, j in combinations(items_1, 2):
                if i in item_df.columns and j in item_df.columns:
                    r = item_df[i].corr(item_df[j])
                    if not np.isnan(r):
                        within_1.append(abs(r))

            within_2 = []
            for i, j in combinations(items_2, 2):
                if i in item_df.columns and j in item_df.columns:
                    r = item_df[i].corr(item_df[j])
                    if not np.isnan(r):
                        within_2.append(abs(r))

            if between_corrs and within_1 and within_2:
                mean_between = np.mean(between_corrs)
                mean_within_1 = np.mean(within_1)
                mean_within_2 = np.mean(within_2)
                denom = np.sqrt(mean_within_1 * mean_within_2)
                htmt = mean_between / denom if denom > 0 else np.nan
            else:
                htmt = np.nan

            results[htmt_key][f"{c1}-{c2}"] = round(float(htmt), 4) if not np.isnan(htmt) else None
            print(f"  {c1}-{c2:<28} {htmt:>8.3f}")

    # Comparison
    print(f"\n  COMPARISON (Raw vs Delta HTMT):")
    print(f"  {'Pair':<35} {'Raw':>8} {'Delta':>8} {'Reduction':>10}")
    print(f"  {'-'*65}")
    for pair in results["raw_htmt"]:
        raw_val = results["raw_htmt"][pair]
        delta_val = results["delta_htmt"][pair]
        if raw_val is not None and delta_val is not None:
            reduction = raw_val - delta_val
            results["comparison"][pair] = {
                "raw": raw_val, "delta": delta_val,
                "reduction": round(reduction, 4),
                "pct_reduction": round(100 * reduction / raw_val, 1) if raw_val > 0 else None,
            }
            print(f"  {pair:<35} {raw_val:>8.3f} {delta_val:>8.3f} {reduction:>+10.3f}")

    # Summary statistics
    raw_vals = [v for v in results["raw_htmt"].values() if v is not None]
    delta_vals = [v for v in results["delta_htmt"].values() if v is not None]
    print(f"\n  Mean HTMT:  Raw={np.mean(raw_vals):.3f}  Delta={np.mean(delta_vals):.3f}")
    print(f"  Max HTMT:   Raw={np.max(raw_vals):.3f}  Delta={np.max(delta_vals):.3f}")
    n_above_90_raw = sum(1 for v in raw_vals if v > 0.90)
    n_above_90_delta = sum(1 for v in delta_vals if v > 0.90)
    print(f"  Pairs > 0.90:  Raw={n_above_90_raw}/{len(raw_vals)}  Delta={n_above_90_delta}/{len(delta_vals)}")

    results["summary"] = {
        "mean_raw_htmt": round(float(np.mean(raw_vals)), 4),
        "mean_delta_htmt": round(float(np.mean(delta_vals)), 4),
        "max_raw_htmt": round(float(np.max(raw_vals)), 4),
        "max_delta_htmt": round(float(np.max(delta_vals)), 4),
        "n_above_90_raw": n_above_90_raw,
        "n_above_90_delta": n_above_90_delta,
    }

    return results


# ============================================================================
# ANALYSIS 2: 3-Factor Robustness (Contagion + Persistence + Detectability)
# ============================================================================
def analysis_2_three_factor_robustness(df_raw, df_features):
    """
    Test whether ALL asymmetry dimensions hold under 3-factor structure:
    contagion, persistence, AND detectability (the existing script only
    checked contagion and persistence).
    """
    print("\n" + "=" * 72)
    print("  ANALYSIS 2: 3-Factor Robustness (Full Asymmetry Check)")
    print("=" * 72)

    results = {}

    for label, factor_def in [("5_factor", FIVE_FACTORS), ("3_factor", THREE_FACTORS)]:
        constructs = list(factor_def.keys())
        scores = compute_factor_scores(df_raw, factor_def)
        deltas = compute_deltas_df(scores, constructs)
        events = label_events_lopo(deltas, constructs)

        # A. Contagion
        lift_crash = cross_session_contagion(events, constructs, "crash")
        lift_surge = cross_session_contagion(events, constructs, "surge")
        n_high_crash = sum(1 for v in lift_crash.values() if v["lift"] > 1.5)
        n_high_surge = sum(1 for v in lift_surge.values() if v["lift"] > 1.5)

        # B. Persistence
        rec_pers = recovery_persistence(events, deltas, constructs)
        crash_recs = [v["crash_recovery_rate"] for v in rec_pers.values() if v["crash_recovery_rate"] is not None]
        surge_pers = [v["surge_persistence_rate"] for v in rec_pers.values() if v["surge_persistence_rate"] is not None]
        avg_crash_rec = float(np.mean(crash_recs)) if crash_recs else None
        avg_surge_per = float(np.mean(surge_pers)) if surge_pers else None

        # C. Detectability (run LOPO detection for each construct)
        detect_results = {}
        if label == "3_factor":
            # Map 3-factor names to feature-space construct names for detection
            factor_to_q = factor_def
            feat_cols = [c for c in df_features.columns if c not in NON_FEATURE_COLS]
            pids = sorted(df_features["pid"].unique())

            for construct_name in constructs:
                q_items = factor_to_q[construct_name]
                # Compute the 3-factor score in the features dataframe
                # We need to merge ratings into the features df
                ratings_for_merge = df_raw[["user_id", "session"] + q_items].copy()
                ratings_for_merge["_factor_score"] = ratings_for_merge[q_items].mean(axis=1)
                ratings_for_merge = ratings_for_merge.rename(columns={"user_id": "pid", "session": "session_num"})

                df_merged = df_features.merge(
                    ratings_for_merge[["pid", "session_num", "_factor_score"]],
                    on=["pid", "session_num"], how="left"
                )

                crash_auprcs, surge_auprcs = [], []

                for test_pid in pids:
                    train_pids = [p for p in pids if p != test_pid]
                    train_mask = df_merged["pid"].isin(train_pids)
                    test_mask = df_merged["pid"] == test_pid

                    deltas_col = df_merged.groupby("pid")["_factor_score"].diff()
                    train_deltas = deltas_col[train_mask].dropna()
                    if train_deltas.std() == 0:
                        continue
                    sd = train_deltas.std()

                    valid = deltas_col.notna()

                    # Crash
                    crash_label = pd.Series(np.nan, index=df_merged.index)
                    crash_label[valid & (deltas_col <= -sd)] = 1.0
                    crash_label[valid & (deltas_col > -sd)] = 0.0

                    # Surge
                    surge_label = pd.Series(np.nan, index=df_merged.index)
                    surge_label[valid & (deltas_col >= sd)] = 1.0
                    surge_label[valid & (deltas_col < sd)] = 0.0

                    for event_type, label_col, auprc_list in [
                        ("crash", crash_label, crash_auprcs),
                        ("surge", surge_label, surge_auprcs),
                    ]:
                        ev_valid = label_col.notna()
                        tr = train_mask & ev_valid
                        te = test_mask & ev_valid
                        if te.sum() == 0 or tr.sum() == 0:
                            continue

                        X_tr = np.nan_to_num(df_merged.loc[tr, feat_cols].values.astype(np.float32), nan=0.0)
                        y_tr = label_col[tr].values.astype(int)
                        X_te = np.nan_to_num(df_merged.loc[te, feat_cols].values.astype(np.float32), nan=0.0)
                        y_te = label_col[te].values.astype(int)

                        if len(np.unique(y_tr)) < 2:
                            continue

                        scaler = StandardScaler()
                        X_tr_s = scaler.fit_transform(X_tr)
                        X_te_s = scaler.transform(X_te)

                        clf = LogisticRegression(C=0.1, l1_ratio=0.5, penalty="elasticnet",
                                                 solver="saga", class_weight="balanced",
                                                 max_iter=5000, random_state=42)
                        clf.fit(X_tr_s, y_tr)
                        auprc_list.append((y_te, clf.predict_proba(X_te_s)[:, 1]))

                # Aggregate predictions across folds
                crash_auprc = surge_auprc = None
                if crash_auprcs:
                    y_all = np.concatenate([y for y, _ in crash_auprcs])
                    s_all = np.concatenate([s for _, s in crash_auprcs])
                    crash_auprc = safe_auprc(y_all, s_all)
                if surge_auprcs:
                    y_all = np.concatenate([y for y, _ in surge_auprcs])
                    s_all = np.concatenate([s for _, s in surge_auprcs])
                    surge_auprc = safe_auprc(y_all, s_all)

                detect_results[construct_name] = {
                    "crash_AUPRC": round(float(crash_auprc), 4) if crash_auprc is not None and not np.isnan(crash_auprc) else None,
                    "surge_AUPRC": round(float(surge_auprc), 4) if surge_auprc is not None and not np.isnan(surge_auprc) else None,
                    "surge_gt_crash": bool(surge_auprc > crash_auprc) if (surge_auprc is not None and crash_auprc is not None and not np.isnan(surge_auprc) and not np.isnan(crash_auprc)) else None,
                }

        r = {
            "constructs": constructs,
            "contagion": {
                "n_high_lift_crash": n_high_crash,
                "n_high_lift_surge": n_high_surge,
                "crash_gt_surge": n_high_crash > n_high_surge,
                "crash_lifts": {k: round(v["lift"], 3) for k, v in lift_crash.items()},
                "surge_lifts": {k: round(v["lift"], 3) for k, v in lift_surge.items()},
            },
            "persistence": {
                "avg_crash_recovery": round(avg_crash_rec, 3) if avg_crash_rec else None,
                "avg_surge_persistence": round(avg_surge_per, 3) if avg_surge_per else None,
                "surge_stickier": bool(avg_surge_per > avg_crash_rec) if (avg_surge_per and avg_crash_rec) else None,
                "per_construct": {c: {
                    "crash_recovery": round(v["crash_recovery_rate"], 3) if v["crash_recovery_rate"] is not None else None,
                    "surge_persistence": round(v["surge_persistence_rate"], 3) if v["surge_persistence_rate"] is not None else None,
                } for c, v in rec_pers.items()},
            },
        }
        if detect_results:
            n_surge_wins = sum(1 for v in detect_results.values() if v.get("surge_gt_crash"))
            r["detectability"] = {
                "per_construct": detect_results,
                "n_surge_gt_crash": n_surge_wins,
                "total_constructs": len(detect_results),
                "surge_more_detectable": n_surge_wins > len(detect_results) / 2,
            }
        results[label] = r

        total_pairs = len(constructs) * (len(constructs) - 1)
        print(f"\n  [{label}] Contagion: crash {n_high_crash}/{total_pairs} > surge {n_high_surge}/{total_pairs} high-lift pairs")
        if avg_crash_rec and avg_surge_per:
            print(f"  [{label}] Persistence: surge persist {avg_surge_per:.0%} vs crash recover {avg_crash_rec:.0%}")
        if detect_results:
            print(f"  [{label}] Detectability:")
            for c, d in detect_results.items():
                print(f"    {c}: crash={d['crash_AUPRC']}  surge={d['surge_AUPRC']}  surge>crash={d['surge_gt_crash']}")

    # Summary
    print(f"\n  ASYMMETRY PRESERVATION UNDER 3-FACTOR:")
    r3 = results["3_factor"]
    checks = []
    c1 = r3["contagion"]["crash_gt_surge"]
    checks.append(c1)
    print(f"    Contagion asymmetry: {'HOLDS' if c1 else 'DOES NOT HOLD'}")
    c2 = r3["persistence"]["surge_stickier"]
    checks.append(c2)
    print(f"    Persistence asymmetry: {'HOLDS' if c2 else 'DOES NOT HOLD'}")
    if "detectability" in r3:
        c3 = r3["detectability"]["surge_more_detectable"]
        checks.append(c3)
        print(f"    Detectability asymmetry: {'HOLDS' if c3 else 'DOES NOT HOLD'}")
    all_hold = all(c for c in checks if c is not None)
    print(f"    ALL DIMENSIONS: {'ALL HOLD' if all_hold else 'PARTIAL'}")
    results["all_asymmetries_hold_3factor"] = all_hold

    return results


# ============================================================================
# ANALYSIS 3: Permutation Test for Crash-Surge Asymmetry
# ============================================================================
def _fast_contagion_lift_counts(events_np, user_ids, constructs, threshold=1.5):
    """
    Vectorized computation of cross-session contagion high-lift pair counts.
    events_np: dict of construct -> (crash_array, surge_array) each shape (N,)
    Returns (n_high_crash, n_high_surge).
    """
    n_c = len(constructs)
    # Build t -> t+1 pairs per user
    unique_users = np.unique(user_ids)
    # Pre-build index pairs
    t_indices = []
    t1_indices = []
    for u in unique_users:
        mask = np.where(user_ids == u)[0]
        if len(mask) < 2:
            continue
        t_indices.extend(mask[:-1])
        t1_indices.extend(mask[1:])
    t_idx = np.array(t_indices)
    t1_idx = np.array(t1_indices)

    if len(t_idx) == 0:
        return 0, 0

    n_high_crash = 0
    n_high_surge = 0
    for i in range(n_c):
        for j in range(n_c):
            if i == j:
                continue
            ci, cj = constructs[i], constructs[j]
            for event_type in ["crash", "surge"]:
                a_t = events_np[ci][event_type][t_idx]
                b_t1 = events_np[cj][event_type][t1_idx]
                base_rate = b_t1.mean()
                given_mask = a_t == 1
                n_given = given_mask.sum()
                if n_given == 0 or base_rate == 0:
                    continue
                cond_rate = b_t1[given_mask].mean()
                lift = cond_rate / base_rate
                if lift > threshold:
                    if event_type == "crash":
                        n_high_crash += 1
                    else:
                        n_high_surge += 1
    return n_high_crash, n_high_surge


def _fast_persistence_diff(events_np, deltas_np, levels_np, user_ids, constructs):
    """
    Level-based persistence/recovery (matching paper Section 4.2.4).
    - Crash recovery: next_level >= pre_crash_level
    - Surge persistence: next_level >= surge_level
    Returns (mean_surge_persist - mean_crash_recover).
    """
    unique_users = np.unique(user_ids)
    t_indices = []
    t1_indices = []
    for u in unique_users:
        mask = np.where(user_ids == u)[0]
        if len(mask) < 2:
            continue
        t_indices.extend(mask[:-1])
        t1_indices.extend(mask[1:])
    t_idx = np.array(t_indices)
    t1_idx = np.array(t1_indices)

    if len(t_idx) == 0:
        return 0.0

    crash_recs = []
    surge_pers = []
    for c in constructs:
        crash_at_t = events_np[c]["crash"][t_idx]
        surge_at_t = events_np[c]["surge"][t_idx]
        level_at_t = levels_np[c][t_idx]
        delta_at_t = deltas_np[c][t_idx]
        level_at_t1 = levels_np[c][t1_idx]

        # Crash recovery: next_level >= pre_crash_level (level_at_t - delta_at_t)
        cm = crash_at_t == 1
        if cm.sum() > 0:
            pre_crash = level_at_t[cm] - delta_at_t[cm]
            crash_recs.append(float((level_at_t1[cm] >= pre_crash).mean()))

        # Surge persistence: next_level >= surge_level
        sm = surge_at_t == 1
        if sm.sum() > 0:
            surge_pers.append(float((level_at_t1[sm] >= level_at_t[sm]).mean()))

    if crash_recs and surge_pers:
        return float(np.mean(surge_pers) - np.mean(crash_recs))
    return 0.0


def analysis_3_permutation_test(df_raw, n_perm=2000):
    """
    Permutation test: randomly shuffle crash/surge labels and test whether
    the observed asymmetry (in contagion and persistence) is unlikely
    under the null hypothesis of symmetry.
    """
    print("\n" + "=" * 72)
    print("  ANALYSIS 3: Permutation Test for Crash-Surge Asymmetry")
    print("=" * 72)

    rng = np.random.RandomState(42)
    constructs = list(FIVE_FACTORS.keys())
    scores = compute_factor_scores(df_raw, FIVE_FACTORS)
    deltas = compute_deltas_df(scores, constructs)
    events = label_events_lopo(deltas, constructs)

    # Sort events by user_id + session for vectorized t->t+1 pairing
    events = events.sort_values(["user_id", "session"]).reset_index(drop=True)
    user_ids = events["user_id"].values

    # Convert to numpy for speed
    events_np = {}
    deltas_np = {}
    levels_np = {}
    for c in constructs:
        events_np[c] = {
            "crash": events[f"crash_{c}"].values.astype(np.int8),
            "surge": events[f"surge_{c}"].values.astype(np.int8),
        }
        # Merge delta and level values
        delta_level_cols = deltas.set_index(["user_id", "session"])[[f"delta_{c}", c]]
        merged = events.set_index(["user_id", "session"]).join(delta_level_cols).reset_index()
        deltas_np[c] = merged[f"delta_{c}"].values
        levels_np[c] = merged[c].values

    n = len(events)

    # Observed statistics
    n_high_crash_obs, n_high_surge_obs = _fast_contagion_lift_counts(events_np, user_ids, constructs)
    contagion_diff_obs = n_high_crash_obs - n_high_surge_obs
    persistence_diff_obs = _fast_persistence_diff(events_np, deltas_np, levels_np, user_ids, constructs)

    # Also get individual values for reporting
    rec_pers_obs = recovery_persistence(events, deltas, constructs)
    crash_recs_obs = [v["crash_recovery_rate"] for v in rec_pers_obs.values() if v["crash_recovery_rate"] is not None]
    surge_pers_obs = [v["surge_persistence_rate"] for v in rec_pers_obs.values() if v["surge_persistence_rate"] is not None]

    print(f"  Observed contagion asymmetry: crash high-lift={n_high_crash_obs}, surge high-lift={n_high_surge_obs}, diff={contagion_diff_obs}")
    print(f"  Observed persistence asymmetry: surge_persist={np.mean(surge_pers_obs):.3f}, crash_recover={np.mean(crash_recs_obs):.3f}, diff={persistence_diff_obs:.3f}")
    print(f"  Running {n_perm} permutations (vectorized)...")

    perm_contagion_diffs = np.zeros(n_perm)
    perm_persistence_diffs = np.zeros(n_perm)

    for perm_i in range(n_perm):
        # For each construct, randomly swap crash/surge labels per row
        events_perm = {}
        for c in constructs:
            swap = rng.random(n) < 0.5
            crash_orig = events_np[c]["crash"].copy()
            surge_orig = events_np[c]["surge"].copy()
            new_crash = np.where(swap, surge_orig, crash_orig)
            new_surge = np.where(swap, crash_orig, surge_orig)
            events_perm[c] = {"crash": new_crash, "surge": new_surge}

        nhc, nhs = _fast_contagion_lift_counts(events_perm, user_ids, constructs)
        perm_contagion_diffs[perm_i] = nhc - nhs
        perm_persistence_diffs[perm_i] = _fast_persistence_diff(events_perm, deltas_np, levels_np, user_ids, constructs)

        if (perm_i + 1) % 500 == 0:
            print(f"    {perm_i + 1}/{n_perm} permutations done")

    # One-sided p-values (observed > permuted)
    p_contagion = float(np.mean(perm_contagion_diffs >= contagion_diff_obs))
    p_persistence = float(np.mean(perm_persistence_diffs >= persistence_diff_obs))

    print(f"\n  PERMUTATION TEST RESULTS:")
    print(f"    Contagion asymmetry: observed diff={contagion_diff_obs}, p={p_contagion:.4f} {'***' if p_contagion < 0.001 else '**' if p_contagion < 0.01 else '*' if p_contagion < 0.05 else 'n.s.'}")
    print(f"    Persistence asymmetry: observed diff={persistence_diff_obs:.3f}, p={p_persistence:.4f} {'***' if p_persistence < 0.001 else '**' if p_persistence < 0.01 else '*' if p_persistence < 0.05 else 'n.s.'}")

    results = {
        "n_permutations": n_perm,
        "contagion": {
            "observed_diff": contagion_diff_obs,
            "observed_crash_high_lift": n_high_crash_obs,
            "observed_surge_high_lift": n_high_surge_obs,
            "p_value": round(p_contagion, 4),
            "significant_05": p_contagion < 0.05,
            "perm_mean": round(float(np.mean(perm_contagion_diffs)), 3),
            "perm_std": round(float(np.std(perm_contagion_diffs)), 3),
        },
        "persistence": {
            "observed_diff": round(persistence_diff_obs, 4),
            "observed_surge_persist": round(float(np.mean(surge_pers_obs)), 4),
            "observed_crash_recover": round(float(np.mean(crash_recs_obs)), 4),
            "p_value": round(p_persistence, 4),
            "significant_05": p_persistence < 0.05,
            "perm_mean": round(float(np.mean(perm_persistence_diffs)), 4),
            "perm_std": round(float(np.std(perm_persistence_diffs)), 4),
        },
    }
    return results


# ============================================================================
# ANALYSIS 4: Bootstrap CIs on Contagion Lift Ratios
# ============================================================================
def analysis_4_bootstrap_contagion_ci(df_raw, n_boot=2000):
    """
    Bootstrap confidence intervals on cross-session contagion lift ratios.
    Resample participants (cluster bootstrap) and recompute lift ratios.
    """
    print("\n" + "=" * 72)
    print("  ANALYSIS 4: Bootstrap CIs on Contagion Lift Ratios")
    print("=" * 72)

    rng = np.random.RandomState(42)
    constructs = list(FIVE_FACTORS.keys())
    scores = compute_factor_scores(df_raw, FIVE_FACTORS)
    deltas = compute_deltas_df(scores, constructs)
    events = label_events_lopo(deltas, constructs)
    users = sorted(events["user_id"].unique())

    # Observed lift ratios
    lift_crash_obs = cross_session_contagion(events, constructs, "crash")
    lift_surge_obs = cross_session_contagion(events, constructs, "surge")

    print(f"  Running {n_boot} cluster-bootstrap iterations (resampling participants)...")

    # Pre-index events by user for fast lookup
    user_event_dfs = {u: events[events["user_id"] == u].copy() for u in users}

    boot_crash_lifts = {k: [] for k in lift_crash_obs}
    boot_surge_lifts = {k: [] for k in lift_surge_obs}
    boot_n_high_crash = []
    boot_n_high_surge = []

    for b in range(n_boot):
        sampled_users = rng.choice(users, size=len(users), replace=True)
        # Build bootstrap sample with reassigned user_ids
        new_events = []
        for new_uid, u in enumerate(sampled_users):
            sub = user_event_dfs[u].copy()
            sub["user_id"] = new_uid
            new_events.append(sub)
        boot_events = pd.concat(new_events, ignore_index=True)

        lc = cross_session_contagion(boot_events, constructs, "crash")
        ls = cross_session_contagion(boot_events, constructs, "surge")

        for k in boot_crash_lifts:
            boot_crash_lifts[k].append(lc.get(k, {}).get("lift", np.nan))
        for k in boot_surge_lifts:
            boot_surge_lifts[k].append(ls.get(k, {}).get("lift", np.nan))

        boot_n_high_crash.append(sum(1 for v in lc.values() if v["lift"] > 1.5))
        boot_n_high_surge.append(sum(1 for v in ls.values() if v["lift"] > 1.5))

        if (b + 1) % 500 == 0:
            print(f"    {b + 1}/{n_boot} bootstrap iterations done")

    results = {"n_bootstrap": n_boot, "crash_lifts": {}, "surge_lifts": {}}

    # Crash lift CIs
    print(f"\n  CRASH contagion lift ratios with 95% bootstrap CIs:")
    print(f"  {'Pair':<25} {'Observed':>10} {'CI_lower':>10} {'CI_upper':>10} {'Excl. 1.0?':>12}")
    for k, obs in sorted(lift_crash_obs.items(), key=lambda x: -x[1]["lift"]):
        vals = [v for v in boot_crash_lifts[k] if not np.isnan(v)]
        if len(vals) > 100:
            ci_lo = float(np.percentile(vals, 2.5))
            ci_hi = float(np.percentile(vals, 97.5))
        else:
            ci_lo = ci_hi = np.nan
        excludes_1 = ci_lo > 1.0 if not np.isnan(ci_lo) else None
        results["crash_lifts"][k] = {
            "observed": round(obs["lift"], 3),
            "ci_lower": round(ci_lo, 3) if not np.isnan(ci_lo) else None,
            "ci_upper": round(ci_hi, 3) if not np.isnan(ci_hi) else None,
            "ci_excludes_1": excludes_1,
            "n_given": obs["n_given"],
        }
        excl_str = "YES" if excludes_1 else "no" if excludes_1 is not None else "N/A"
        print(f"  {k:<25} {obs['lift']:>10.3f} {ci_lo:>10.3f} {ci_hi:>10.3f} {excl_str:>12}")

    # Surge lift CIs
    print(f"\n  SURGE contagion lift ratios with 95% bootstrap CIs:")
    print(f"  {'Pair':<25} {'Observed':>10} {'CI_lower':>10} {'CI_upper':>10} {'Excl. 1.0?':>12}")
    for k, obs in sorted(lift_surge_obs.items(), key=lambda x: -x[1]["lift"]):
        vals = [v for v in boot_surge_lifts[k] if not np.isnan(v)]
        if len(vals) > 100:
            ci_lo = float(np.percentile(vals, 2.5))
            ci_hi = float(np.percentile(vals, 97.5))
        else:
            ci_lo = ci_hi = np.nan
        excludes_1 = ci_lo > 1.0 if not np.isnan(ci_lo) else None
        results["surge_lifts"][k] = {
            "observed": round(obs["lift"], 3),
            "ci_lower": round(ci_lo, 3) if not np.isnan(ci_lo) else None,
            "ci_upper": round(ci_hi, 3) if not np.isnan(ci_hi) else None,
            "ci_excludes_1": excludes_1,
            "n_given": obs["n_given"],
        }
        excl_str = "YES" if excludes_1 else "no" if excludes_1 is not None else "N/A"
        print(f"  {k:<25} {obs['lift']:>10.3f} {ci_lo:>10.3f} {ci_hi:>10.3f} {excl_str:>12}")

    # High-lift count asymmetry CI
    boot_diff = np.array(boot_n_high_crash) - np.array(boot_n_high_surge)
    ci_lo_diff = float(np.percentile(boot_diff, 2.5))
    ci_hi_diff = float(np.percentile(boot_diff, 97.5))
    obs_diff = sum(1 for v in lift_crash_obs.values() if v["lift"] > 1.5) - sum(1 for v in lift_surge_obs.values() if v["lift"] > 1.5)

    results["high_lift_count_asymmetry"] = {
        "observed_diff": obs_diff,
        "ci_lower": round(ci_lo_diff, 1),
        "ci_upper": round(ci_hi_diff, 1),
        "ci_excludes_0": ci_lo_diff > 0,
    }
    print(f"\n  High-lift pair count asymmetry (crash - surge):")
    print(f"    Observed: {obs_diff}, 95% CI: [{ci_lo_diff:.1f}, {ci_hi_diff:.1f}]")
    print(f"    CI excludes 0: {'YES' if ci_lo_diff > 0 else 'no'}")

    # Count how many crash pairs have CI excluding 1.0
    n_crash_sig = sum(1 for v in results["crash_lifts"].values() if v.get("ci_excludes_1"))
    n_surge_sig = sum(1 for v in results["surge_lifts"].values() if v.get("ci_excludes_1"))
    print(f"\n  Pairs with CI excluding 1.0: crash={n_crash_sig}/{len(results['crash_lifts'])}, surge={n_surge_sig}/{len(results['surge_lifts'])}")
    results["n_crash_sig"] = n_crash_sig
    results["n_surge_sig"] = n_surge_sig

    return results


# ============================================================================
# ANALYSIS 5: Leave-One-Participant-Out Sensitivity (Jackknife)
# ============================================================================
def analysis_5_jackknife_sensitivity(df_raw):
    """
    Remove each participant one at a time and recompute key findings.
    Shows that no single participant drives the results.
    """
    print("\n" + "=" * 72)
    print("  ANALYSIS 5: Jackknife Sensitivity (Leave-One-Participant-Out)")
    print("=" * 72)

    constructs = list(FIVE_FACTORS.keys())
    users = sorted(df_raw["user_id"].unique())

    # Full-sample results
    scores_full = compute_factor_scores(df_raw, FIVE_FACTORS)
    deltas_full = compute_deltas_df(scores_full, constructs)
    events_full = label_events_lopo(deltas_full, constructs)

    lift_crash_full = cross_session_contagion(events_full, constructs, "crash")
    lift_surge_full = cross_session_contagion(events_full, constructs, "surge")
    n_high_crash_full = sum(1 for v in lift_crash_full.values() if v["lift"] > 1.5)
    n_high_surge_full = sum(1 for v in lift_surge_full.values() if v["lift"] > 1.5)
    contagion_holds_full = n_high_crash_full > n_high_surge_full

    rec_pers_full = recovery_persistence(events_full, deltas_full, constructs)
    cr_full = [v["crash_recovery_rate"] for v in rec_pers_full.values() if v["crash_recovery_rate"] is not None]
    sp_full = [v["surge_persistence_rate"] for v in rec_pers_full.values() if v["surge_persistence_rate"] is not None]
    persist_holds_full = float(np.mean(sp_full)) > float(np.mean(cr_full))

    print(f"  Full sample: contagion asym.={'HOLDS' if contagion_holds_full else 'NO'}, "
          f"persistence asym.={'HOLDS' if persist_holds_full else 'NO'}")
    print(f"  Running {len(users)} jackknife iterations (dropping one participant each)...\n")

    jackknife_results = []
    contagion_holds_count = 0
    persistence_holds_count = 0

    for uid in users:
        df_jack = df_raw[df_raw["user_id"] != uid]
        scores = compute_factor_scores(df_jack, FIVE_FACTORS)
        deltas = compute_deltas_df(scores, constructs)
        events = label_events_lopo(deltas, constructs)

        lc = cross_session_contagion(events, constructs, "crash")
        ls = cross_session_contagion(events, constructs, "surge")
        nhc = sum(1 for v in lc.values() if v["lift"] > 1.5)
        nhs = sum(1 for v in ls.values() if v["lift"] > 1.5)
        contagion_holds = nhc > nhs

        rp = recovery_persistence(events, deltas, constructs)
        crs = [v["crash_recovery_rate"] for v in rp.values() if v["crash_recovery_rate"] is not None]
        sps = [v["surge_persistence_rate"] for v in rp.values() if v["surge_persistence_rate"] is not None]
        pers_holds = float(np.mean(sps)) > float(np.mean(crs)) if crs and sps else None

        if contagion_holds:
            contagion_holds_count += 1
        if pers_holds:
            persistence_holds_count += 1

        jackknife_results.append({
            "dropped_user": int(uid),
            "n_high_lift_crash": nhc,
            "n_high_lift_surge": nhs,
            "contagion_holds": contagion_holds,
            "avg_crash_recovery": round(float(np.mean(crs)), 3) if crs else None,
            "avg_surge_persistence": round(float(np.mean(sps)), 3) if sps else None,
            "persistence_holds": pers_holds,
        })

        status = "OK" if (contagion_holds and pers_holds) else "PARTIAL" if (contagion_holds or pers_holds) else "FLIPPED"
        print(f"  Drop P{uid}: contagion crash={nhc} surge={nhs} {'HOLDS' if contagion_holds else 'FLIP'}, "
              f"persist {'HOLDS' if pers_holds else 'FLIP'}  [{status}]")

    print(f"\n  JACKKNIFE SUMMARY:")
    print(f"    Contagion asymmetry holds: {contagion_holds_count}/{len(users)} ({contagion_holds_count/len(users):.0%})")
    print(f"    Persistence asymmetry holds: {persistence_holds_count}/{len(users)} ({persistence_holds_count/len(users):.0%})")
    print(f"    Both hold: {sum(1 for r in jackknife_results if r['contagion_holds'] and r['persistence_holds'])}/{len(users)}")

    flipped_users = [r["dropped_user"] for r in jackknife_results
                     if not (r["contagion_holds"] and r["persistence_holds"])]
    if flipped_users:
        print(f"    Participants whose removal changes a finding: {flipped_users}")
    else:
        print(f"    NO single participant's removal changes any finding!")

    results = {
        "n_participants": len(users),
        "contagion_holds_count": contagion_holds_count,
        "contagion_holds_pct": round(contagion_holds_count / len(users), 3),
        "persistence_holds_count": persistence_holds_count,
        "persistence_holds_pct": round(persistence_holds_count / len(users), 3),
        "both_hold_count": sum(1 for r in jackknife_results if r["contagion_holds"] and r["persistence_holds"]),
        "flipped_users": flipped_users,
        "per_participant": jackknife_results,
    }
    return results


# ============================================================================
# MAIN
# ============================================================================
def main():
    t_start = time.time()
    print("=" * 72)
    print("  ROBUSTNESS ANALYSES FOR ICMI '26 PAPER")
    print("  Addressing sample size (N=24) and discriminant validity concerns")
    print("=" * 72)

    df_raw = load_ratings()
    print(f"  Loaded ratings: {len(df_raw)} observations, {df_raw['user_id'].nunique()} participants")

    df_features = load_features()
    print(f"  Loaded features: {len(df_features)} rows, {len([c for c in df_features.columns if c not in NON_FEATURE_COLS])} features")

    all_results = {}

    # Analysis 1: Delta-level HTMT
    all_results["analysis_1_delta_htmt"] = analysis_1_delta_htmt(df_raw)

    # Analysis 2: 3-factor robustness (full check including detectability)
    all_results["analysis_2_three_factor"] = analysis_2_three_factor_robustness(df_raw, df_features)

    # Analysis 3: Permutation test
    all_results["analysis_3_permutation"] = analysis_3_permutation_test(df_raw, n_perm=2000)

    # Analysis 4: Bootstrap CIs on contagion lift
    all_results["analysis_4_bootstrap_contagion"] = analysis_4_bootstrap_contagion_ci(df_raw, n_boot=2000)

    # Analysis 5: Jackknife sensitivity
    all_results["analysis_5_jackknife"] = analysis_5_jackknife_sensitivity(df_raw)

    # ── Save all results ─────────────────────────────────────────────────
    output_path = OUTPUT_DIR / "robustness_analyses_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    total_time = time.time() - t_start
    print(f"\n{'=' * 72}")
    print(f"  ALL ANALYSES COMPLETE ({total_time:.0f}s / {total_time/60:.1f}min)")
    print(f"  Results saved to: {output_path}")
    print(f"{'=' * 72}")

    # ── Executive summary ────────────────────────────────────────────────
    print(f"\n  EXECUTIVE SUMMARY")
    print(f"  {'─' * 60}")

    # 1
    s1 = all_results["analysis_1_delta_htmt"]["summary"]
    print(f"  1. Delta HTMT: max drops from {s1['max_raw_htmt']:.3f} (raw) to {s1['max_delta_htmt']:.3f} (delta)")
    print(f"     Pairs > 0.90: {s1['n_above_90_raw']} (raw) -> {s1['n_above_90_delta']} (delta)")

    # 2
    holds = all_results["analysis_2_three_factor"]["all_asymmetries_hold_3factor"]
    print(f"  2. 3-Factor robustness: all asymmetries {'HOLD' if holds else 'PARTIAL'}")

    # 3
    s3 = all_results["analysis_3_permutation"]
    print(f"  3. Permutation tests: contagion p={s3['contagion']['p_value']:.4f}, "
          f"persistence p={s3['persistence']['p_value']:.4f}")

    # 4
    s4 = all_results["analysis_4_bootstrap_contagion"]
    print(f"  4. Bootstrap contagion CIs: {s4['n_crash_sig']} crash pairs exclude 1.0, "
          f"{s4['n_surge_sig']} surge pairs exclude 1.0")
    ha = s4["high_lift_count_asymmetry"]
    print(f"     High-lift count diff CI: [{ha['ci_lower']:.1f}, {ha['ci_upper']:.1f}] "
          f"{'(excludes 0)' if ha['ci_excludes_0'] else '(includes 0)'}")

    # 5
    s5 = all_results["analysis_5_jackknife"]
    print(f"  5. Jackknife: contagion holds {s5['contagion_holds_pct']:.0%}, "
          f"persistence holds {s5['persistence_holds_pct']:.0%}")
    if s5["flipped_users"]:
        print(f"     Sensitive to removing: {s5['flipped_users']}")
    else:
        print(f"     Robust to removing any single participant")


if __name__ == "__main__":
    main()
