#!/usr/bin/env python3
"""
Experiment 16: Threshold Sensitivity Analysis

Re-run event characterization and detection at multiple SD thresholds:
  0.75 SD, 1.0 SD (baseline), 1.25 SD, 1.5 SD

Shows that main findings (contagion patterns, detection feasibility) are
robust to threshold choice. Appendix material.
"""

import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, precision_recall_curve

warnings.filterwarnings("ignore")

from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "full_features_with_temporals.csv"
RESULTS_DIR = BASE / "crash_surge" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONSTRUCTS = ["familiarity", "social_penetration", "memory", "conversational", "enjoyment"]
NON_FEATURE_COLS = {"pid", "session_num", "n_prior_sessions"} | set(CONSTRUCTS)

SD_MULTIPLIERS = [0.75, 1.0, 1.25, 1.5]

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def get_all_feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def safe_auprc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_prob)


def run_experiment():
    t_total = time.time()

    print("=" * 72)
    print("EXPERIMENT 16: THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 72)

    df = pd.read_csv(DATA_PATH)
    feat_cols = get_all_feature_cols(df)
    pids = sorted(df["pid"].unique())
    n_pids = len(pids)

    print(f"Loaded {len(df)} rows, {n_pids} participants, {len(feat_cols)} features")
    print(f"Testing SD multipliers: {SD_MULTIPLIERS}")

    results = {
        "experiment": "exp16_threshold_sensitivity",
        "sd_multipliers": SD_MULTIPLIERS,
        "per_threshold": {}
    }

    for sd_mult in SD_MULTIPLIERS:
        print(f"\n{'=' * 60}")
        print(f"SD Multiplier: {sd_mult}")
        print(f"{'=' * 60}")

        threshold_results = {
            "sd_multiplier": sd_mult,
            "event_rates": {},
            "detection": {},
            "contagion_summary": {},
        }

        # 1. Event rates at this threshold
        for construct in CONSTRUCTS:
            deltas = df.groupby("pid")[construct].diff()
            sd = deltas.dropna().std()
            threshold = sd * sd_mult

            n_valid = deltas.notna().sum()
            n_crash = (deltas <= -threshold).sum()
            n_surge = (deltas >= threshold).sum()

            threshold_results["event_rates"][construct] = {
                "sd": round(float(sd), 4),
                "threshold": round(float(threshold), 4),
                "n_crash": int(n_crash),
                "crash_rate": round(float(n_crash / n_valid), 4),
                "n_surge": int(n_surge),
                "surge_rate": round(float(n_surge / n_valid), 4),
            }

            print(f"  {construct:<22} crash={n_crash} ({n_crash/n_valid:.1%})  "
                  f"surge={n_surge} ({n_surge/n_valid:.1%})  thr={threshold:.3f}")

        # 2. Contagion summary: Enjoyment-Conversational co-occurrence
        deltas_all = {}
        crash_all = {}
        for construct in CONSTRUCTS:
            d = df.groupby("pid")[construct].diff()
            sd = d.dropna().std()
            deltas_all[construct] = d
            crash_all[construct] = d <= -(sd * sd_mult)

        # Co-occurrence: P(conv crashes | enj crashes)
        enj_crash_mask = crash_all["enjoyment"] & deltas_all["enjoyment"].notna()
        if enj_crash_mask.sum() > 0:
            conv_given_enj = crash_all["conversational"][enj_crash_mask].mean()
        else:
            conv_given_enj = np.nan

        # Reverse: P(enj crashes | conv crashes)
        conv_crash_mask = crash_all["conversational"] & deltas_all["conversational"].notna()
        if conv_crash_mask.sum() > 0:
            enj_given_conv = crash_all["enjoyment"][conv_crash_mask].mean()
        else:
            enj_given_conv = np.nan

        # Recovery rate for enjoyment
        enj_crashes = []
        for pid, group in df.groupby("pid"):
            d = group["enjoyment"].diff()
            sd = d.dropna().std() if len(d.dropna()) > 1 else 1.0
            global_sd = deltas_all["enjoyment"].dropna().std()
            thr = global_sd * sd_mult

            for i in range(1, len(group)):
                if d.iloc[i] is not np.nan and d.iloc[i] <= -thr:
                    # Check recovery in remaining sessions
                    recovered = False
                    for j in range(i + 1, len(group)):
                        if d.iloc[j] is not np.nan and d.iloc[j] >= thr:
                            recovered = True
                            break
                    enj_crashes.append(recovered)

        enj_recovery = np.mean(enj_crashes) if enj_crashes else np.nan

        threshold_results["contagion_summary"] = {
            "P_conv_crash_given_enj_crash": round(float(conv_given_enj), 4) if not np.isnan(conv_given_enj) else None,
            "P_enj_crash_given_conv_crash": round(float(enj_given_conv), 4) if not np.isnan(enj_given_conv) else None,
            "enjoyment_recovery_rate": round(float(enj_recovery), 4) if not np.isnan(enj_recovery) else None,
            "n_enjoyment_crashes": len(enj_crashes),
        }
        print(f"\n  Contagion: P(conv|enj)={conv_given_enj:.2f}  P(enj|conv)={enj_given_conv:.2f}")
        print(f"  Enjoyment recovery: {enj_recovery:.1%} ({len(enj_crashes)} crashes)")

        # 3. Detection at this threshold (XGB, per-construct)
        for construct in CONSTRUCTS:
            all_y_true, all_y_score = [], []

            for test_pid in pids:
                train_pids = [p for p in pids if p != test_pid]
                train_mask = df["pid"].isin(train_pids)
                test_mask = df["pid"] == test_pid

                deltas = df.groupby("pid")[construct].diff()
                train_deltas = deltas[train_mask].dropna()
                sd = train_deltas.std()
                thr = sd * sd_mult

                binary = pd.Series(np.nan, index=df.index)
                valid = deltas.notna()
                binary[valid & (deltas <= -thr)] = 1.0
                binary[valid & (deltas > -thr)] = 0.0

                train_valid = train_mask & valid & binary.notna()
                test_valid = test_mask & valid & binary.notna()

                if test_valid.sum() == 0:
                    continue

                X_train = np.nan_to_num(df.loc[train_valid, feat_cols].values.astype(np.float32), nan=0.0)
                y_train = binary[train_valid].values.astype(int)
                X_test = np.nan_to_num(df.loc[test_valid, feat_cols].values.astype(np.float32), nan=0.0)
                y_test = binary[test_valid].values.astype(int)

                if len(np.unique(y_train)) < 2:
                    continue

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                if HAS_XGB:
                    spw = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)
                    clf = XGBClassifier(max_depth=5, n_estimators=100, learning_rate=0.1,
                                         scale_pos_weight=spw, random_state=42,
                                         eval_metric="logloss", verbosity=0, n_jobs=1)
                else:
                    clf = LogisticRegression(C=0.1, l1_ratio=0.5, penalty="elasticnet",
                                             solver="saga", class_weight="balanced",
                                             max_iter=5000, random_state=42)

                clf.fit(X_train_s, y_train)
                y_score = clf.predict_proba(X_test_s)[:, 1]
                all_y_true.extend(y_test.tolist())
                all_y_score.extend(y_score.tolist())

            all_y_true = np.array(all_y_true)
            all_y_score = np.array(all_y_score)

            auprc = safe_auprc(all_y_true, all_y_score)
            best_f1_val = None
            if auprc and not np.isnan(auprc):
                prec, rec, thresholds = precision_recall_curve(all_y_true, all_y_score)
                f1s = 2 * prec * rec / (prec + rec + 1e-10)
                best_f1_val = float(np.max(f1s))

            threshold_results["detection"][construct] = {
                "AUPRC": round(float(auprc), 4) if auprc and not np.isnan(auprc) else None,
                "best_F1": round(float(best_f1_val), 4) if best_f1_val else None,
                "crash_rate": round(float(all_y_true.mean()), 4) if len(all_y_true) > 0 else None,
            }

        results["per_threshold"][str(sd_mult)] = threshold_results

    # Summary table
    print(f"\n{'=' * 72}")
    print("SUMMARY: Detection AUPRC Across Thresholds (XGB)")
    print(f"{'=' * 72}")
    header = f"  {'Construct':<22}" + "".join(f"{sd:>8}" for sd in SD_MULTIPLIERS)
    print(header)
    print(f"  {'─' * (22 + 8 * len(SD_MULTIPLIERS))}")

    for c in CONSTRUCTS:
        row = f"  {c:<22}"
        for sd_mult in SD_MULTIPLIERS:
            det = results["per_threshold"][str(sd_mult)]["detection"].get(c, {})
            auprc_val = det.get("AUPRC", "-")
            row += f"{str(auprc_val):>8}"
        print(row)

    print(f"\nContagion stability:")
    for sd_mult in SD_MULTIPLIERS:
        cs = results["per_threshold"][str(sd_mult)]["contagion_summary"]
        print(f"  {sd_mult} SD: P(conv|enj)={cs['P_conv_crash_given_enj_crash']}  "
              f"enj_recovery={cs['enjoyment_recovery_rate']}")

    print(f"\n  Total time: {time.time()-t_total:.0f}s ({(time.time()-t_total)/60:.1f}min)")

    out_path = RESULTS_DIR / "exp16_threshold_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}")

    return results


if __name__ == "__main__":
    run_experiment()
