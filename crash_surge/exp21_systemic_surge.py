#!/usr/bin/env python3
"""
Experiment 21: Systemic Surge Detector

Parallels Exp7 (systemic crash detector) but for positive events.
Systemic surge = 2+ constructs surging simultaneously.
Binary classification: systemic surge vs not.
Models: Elastic-Net LR + XGBoost, LOPO CV.
"""

import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    f1_score, confusion_matrix
)

warnings.filterwarnings("ignore")

from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "full_features_with_temporals.csv"
RESULTS_DIR = BASE / "crash_surge" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONSTRUCTS = ["familiarity", "social_penetration", "memory", "conversational", "enjoyment"]
NON_FEATURE_COLS = {"pid", "session_num", "n_prior_sessions"} | set(CONSTRUCTS)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def get_all_feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def compute_systemic_surge_labels(df, train_pids):
    """
    Systemic surge = enjoyment surge OR 2+ constructs surge simultaneously.
    """
    surge_flags = pd.DataFrame(index=df.index)
    train_mask = df["pid"].isin(train_pids)

    for construct in CONSTRUCTS:
        deltas = df.groupby("pid")[construct].diff()
        train_deltas = deltas[train_mask].dropna()
        sd = train_deltas.std()

        surge = pd.Series(False, index=df.index)
        valid = deltas.notna()
        surge[valid] = deltas[valid] >= sd
        surge[~valid] = np.nan
        surge_flags[construct] = surge

    surge_count = surge_flags[CONSTRUCTS].sum(axis=1)
    enjoyment_surge = surge_flags["enjoyment"] == True

    # Systemic = 2+ constructs surge (no enjoyment carve-out)
    systemic = (surge_count >= 2)
    modular = (surge_count == 1)

    label = pd.Series("stable", index=df.index)
    label[modular] = "modular"
    label[systemic] = "systemic"

    any_nan = surge_flags[CONSTRUCTS].isna().any(axis=1)
    label[any_nan] = np.nan

    return label, surge_flags


def run_experiment():
    t_total = time.time()

    print("=" * 72)
    print("EXPERIMENT 21: SYSTEMIC SURGE DETECTOR")
    print("=" * 72)

    df = pd.read_csv(DATA_PATH)
    feat_cols = get_all_feature_cols(df)
    pids = sorted(df["pid"].unique())

    print(f"Loaded {len(df)} rows, {len(pids)} participants, {len(feat_cols)} features")

    # Descriptive stats (global thresholds)
    global_labels, global_surge_flags = compute_systemic_surge_labels(df, pids)
    valid = global_labels.notna()
    n_valid = int(valid.sum())
    n_systemic = int((global_labels == "systemic").sum())
    n_modular = int((global_labels == "modular").sum())
    n_stable = int((global_labels == "stable").sum())

    print(f"\n  Total valid transitions: {n_valid}")
    print(f"  Systemic surges:        {n_systemic} ({n_systemic/n_valid:.1%})")
    print(f"  Modular surges:         {n_modular} ({n_modular/n_valid:.1%})")
    print(f"  No surge (stable):      {n_stable} ({n_stable/n_valid:.1%})")

    enjoyment_surge = global_surge_flags["enjoyment"] == True
    surge_count = global_surge_flags[CONSTRUCTS].sum(axis=1)
    sys_from_enj = ((global_labels == "systemic") & enjoyment_surge).sum()
    sys_from_multi = ((global_labels == "systemic") & (surge_count >= 2)).sum()
    sys_both = ((global_labels == "systemic") & enjoyment_surge & (surge_count >= 2)).sum()

    print(f"\n  Systemic breakdown:")
    print(f"    Enjoyment surge:       {sys_from_enj}")
    print(f"    2+ constructs surge:   {sys_from_multi}")
    print(f"    Both:                  {sys_both}")

    results = {
        "experiment": "exp21_systemic_surge",
        "event_distribution": {
            "n_valid": n_valid, "n_systemic": n_systemic,
            "n_modular": n_modular, "n_stable": n_stable,
            "systemic_rate": round(n_systemic / n_valid, 4),
            "systemic_from_enjoyment": int(sys_from_enj),
            "systemic_from_multi": int(sys_from_multi),
            "systemic_both": int(sys_both),
        },
        "models": {}
    }

    # Run models
    for model_key, model_name in [("en", "Elastic-Net LR"), ("xgb", "XGBoost")]:
        if model_key == "xgb" and not HAS_XGB:
            continue

        print(f"\n{'─' * 60}")
        print(f"Model: {model_name}")
        print(f"{'─' * 60}")

        all_y_true, all_y_score = [], []
        feature_importances = np.zeros(len(feat_cols))
        n_folds = 0

        for test_pid in pids:
            train_pids = [p for p in pids if p != test_pid]
            train_mask = df["pid"].isin(train_pids)
            test_mask = df["pid"] == test_pid

            systemic_labels, _ = compute_systemic_surge_labels(df, train_pids)

            binary = pd.Series(np.nan, index=df.index)
            valid = systemic_labels.notna()
            binary[valid & (systemic_labels == "systemic")] = 1.0
            binary[valid & (systemic_labels != "systemic")] = 0.0

            train_valid = train_mask & valid
            test_valid = test_mask & valid

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

            if model_key == "en":
                clf = LogisticRegression(C=0.1, l1_ratio=0.5, penalty="elasticnet",
                                          solver="saga", class_weight="balanced",
                                          max_iter=5000, random_state=42)
                clf.fit(X_train_s, y_train)
                y_score = clf.predict_proba(X_test_s)[:, 1]
                feature_importances += np.abs(clf.coef_[0])
            else:
                spw = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)
                clf = XGBClassifier(max_depth=5, n_estimators=100, learning_rate=0.1,
                                     scale_pos_weight=spw, random_state=42,
                                     eval_metric="logloss", verbosity=0, n_jobs=1)
                clf.fit(X_train_s, y_train)
                y_score = clf.predict_proba(X_test_s)[:, 1]
                feature_importances += clf.feature_importances_

            n_folds += 1
            all_y_true.extend(y_test.tolist())
            all_y_score.extend(y_score.tolist())

        all_y_true = np.array(all_y_true)
        all_y_score = np.array(all_y_score)

        auprc = average_precision_score(all_y_true, all_y_score) if len(np.unique(all_y_true)) >= 2 else None

        best_f1 = None
        recall_at_50 = None
        macro_f1 = None
        if auprc is not None:
            prec, rec, thresholds = precision_recall_curve(all_y_true, all_y_score)
            valid_50 = prec >= 0.50
            recall_at_50 = float(rec[valid_50].max()) if valid_50.any() else 0.0
            f1s = 2 * prec * rec / (prec + rec + 1e-10)
            best_f1_idx = np.argmax(f1s)
            best_f1 = float(f1s[best_f1_idx])
            best_thresh = float(thresholds[best_f1_idx]) if best_f1_idx < len(thresholds) else 0.5
            y_pred = (all_y_score >= best_thresh).astype(int)
            macro_f1 = float(f1_score(all_y_true, y_pred, average="macro", zero_division=0))

        avg_imp = feature_importances / max(n_folds, 1)
        top_idx = np.argsort(avg_imp)[::-1][:15]
        top_feats = [(feat_cols[i], round(float(avg_imp[i]), 6)) for i in top_idx]

        print(f"  AUPRC: {auprc:.4f}" if auprc else "  AUPRC: N/A")
        print(f"  Best F1: {best_f1:.4f}" if best_f1 else "  Best F1: N/A")
        print(f"  Macro F1: {macro_f1:.4f}" if macro_f1 else "  Macro F1: N/A")
        print(f"  Recall@50%Prec: {recall_at_50:.4f}" if recall_at_50 is not None else "  Recall@50%Prec: N/A")
        print(f"  Top 5 features:")
        for fn, fv in top_feats[:5]:
            print(f"    {fn:<45} {fv:.6f}")

        results["models"][model_key] = {
            "AUPRC": round(float(auprc), 4) if auprc else None,
            "best_F1": round(float(best_f1), 4) if best_f1 else None,
            "macro_F1": round(float(macro_f1), 4) if macro_f1 else None,
            "recall_at_50pct_precision": round(float(recall_at_50), 4) if recall_at_50 is not None else None,
            "n_total": int(len(all_y_true)),
            "n_systemic": int(all_y_true.sum()),
            "systemic_rate": round(float(all_y_true.mean()), 4),
            "top_features": top_feats,
        }

    print(f"\n  Total time: {time.time()-t_total:.0f}s")

    out_path = RESULTS_DIR / "exp21_systemic_surge_2plus.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}")

    return results


if __name__ == "__main__":
    run_experiment()
