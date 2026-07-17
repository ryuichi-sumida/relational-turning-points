"""
Experiment 7: Systemic/Cascade Detector — Detect systemic crashes.

Systemic crash = 2+ constructs crash in same transition.
Modular crash = single-construct crash.

Binary classification: systemic vs not-systemic.
Models: Elastic-net LR + XGBoost
Evaluation: LOPO with AUPRC, F1, recall@50% precision.
"""

import sys
import json
from pathlib import Path
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONSTRUCTS, RESULTS_DIR
from data_loader import (
    load_features_and_labels, get_feature_columns, compute_deltas
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def compute_systemic_labels(df, train_pids):
    """
    Compute systemic crash labels using training-set SD thresholds.

    Returns:
        systemic_label: Series ('systemic', 'modular', 'stable', or NaN)
        crash_details: dict with per-construct crash counts and thresholds
    """
    crash_flags = pd.DataFrame(index=df.index)
    thresholds = {}

    train_mask = df["pid"].isin(train_pids)

    for construct in CONSTRUCTS:
        deltas = compute_deltas(df, construct)
        train_deltas = deltas[train_mask].dropna()
        sd = train_deltas.std()
        thresholds[construct] = float(sd)

        crash = pd.Series(False, index=df.index)
        valid = deltas.notna()
        crash[valid] = deltas[valid] <= -sd
        crash[~valid] = np.nan
        crash_flags[construct] = crash

    # Count crashes per transition
    crash_count = crash_flags[CONSTRUCTS].sum(axis=1)
    enjoyment_crash = crash_flags["enjoyment"] == True

    # Systemic = 2+ constructs crash (no enjoyment carve-out)
    systemic_mask = (crash_count >= 2)
    # Modular = exactly 1 construct crash
    modular_mask = (crash_count == 1)

    label = pd.Series("stable", index=df.index)
    label[modular_mask] = "modular"
    label[systemic_mask] = "systemic"

    # NaN for first sessions (any construct has NaN delta)
    any_nan = crash_flags[CONSTRUCTS].isna().any(axis=1)
    label[any_nan] = np.nan

    return label, thresholds, crash_flags


def run_experiment():
    print("=" * 72)
    print("EXPERIMENT 7: SYSTEMIC/CASCADE DETECTOR")
    print("=" * 72)

    # ---- Load data ----
    df = load_features_and_labels()
    feat_cols = get_feature_columns(df, include_session_index=False)
    unique_pids = sorted(df["pid"].unique())
    print(f"Loaded {len(df)} rows, {df['pid'].nunique()} participants, {len(feat_cols)} features")

    # ---- Descriptive: event distribution (using global thresholds) ----
    print(f"\n{'─' * 60}")
    print("Event Distribution (global thresholds)")
    print(f"{'─' * 60}")
    global_labels, _, global_crash_flags = compute_systemic_labels(df, unique_pids)
    valid = global_labels.notna()
    n_valid = valid.sum()
    n_systemic = (global_labels == "systemic").sum()
    n_modular = (global_labels == "modular").sum()
    n_stable = (global_labels == "stable").sum()
    n_no_crash = n_stable  # stable = no crash at all

    print(f"  Total valid transitions: {n_valid}")
    print(f"  Systemic crashes:        {n_systemic} ({n_systemic/n_valid:.1%})")
    print(f"  Modular crashes:         {n_modular} ({n_modular/n_valid:.1%})")
    print(f"  No crash (stable):       {n_stable} ({n_stable/n_valid:.1%})")

    # Breakdown of systemic: enjoyment crash vs 2+ constructs
    enjoyment_crash = global_crash_flags["enjoyment"] == True
    crash_count = global_crash_flags[CONSTRUCTS].sum(axis=1)
    systemic_from_enjoyment = (global_labels == "systemic") & enjoyment_crash
    systemic_from_multi = (global_labels == "systemic") & (crash_count >= 2)
    systemic_both = systemic_from_enjoyment & systemic_from_multi
    print(f"\n  Systemic breakdown:")
    print(f"    Enjoyment crash (any):       {systemic_from_enjoyment.sum()}")
    print(f"    2+ constructs crash:         {systemic_from_multi.sum()}")
    print(f"    Both (enjoyment + 2+ other): {systemic_both.sum()}")

    results = {
        "experiment": "exp7_systemic_cascade",
        "n_participants": len(unique_pids),
        "n_features": len(feat_cols),
        "event_distribution": {
            "n_valid_transitions": int(n_valid),
            "n_systemic": int(n_systemic),
            "n_modular": int(n_modular),
            "n_stable": int(n_stable),
            "systemic_rate": round(float(n_systemic / n_valid), 4),
            "modular_rate": round(float(n_modular / n_valid), 4),
            "stable_rate": round(float(n_stable / n_valid), 4),
            "systemic_from_enjoyment": int(systemic_from_enjoyment.sum()),
            "systemic_from_multi_construct": int(systemic_from_multi.sum()),
            "systemic_both": int(systemic_both.sum()),
        },
        "models": {}
    }

    # ---- Run models ----
    model_configs = {
        "elastic_net_lr": {
            "name": "Elastic-Net Logistic Regression",
            "params": {"C": 0.1, "l1_ratio": 0.5, "penalty": "elasticnet",
                       "solver": "saga", "class_weight": "balanced",
                       "max_iter": 5000, "random_state": 42}
        },
    }
    if HAS_XGB:
        model_configs["xgboost"] = {
            "name": "XGBoost",
            "params": {"max_depth": 5, "n_estimators": 100, "learning_rate": 0.1,
                       "random_state": 42, "eval_metric": "logloss"}
        }

    for model_key, model_cfg in model_configs.items():
        print(f"\n{'─' * 60}")
        print(f"Model: {model_cfg['name']}")
        print(f"{'─' * 60}")

        all_y_true = []
        all_y_score = []
        feature_importances = np.zeros(len(feat_cols))
        n_folds = 0

        for test_pid in unique_pids:
            train_pids = [p for p in unique_pids if p != test_pid]
            train_mask = df["pid"].isin(train_pids)
            test_mask = df["pid"] == test_pid

            # Compute systemic labels using training-set thresholds
            systemic_labels, thresholds, _ = compute_systemic_labels(df, train_pids)

            # Binary: systemic=1, everything else=0
            binary = pd.Series(np.nan, index=df.index)
            valid = systemic_labels.notna()
            binary[valid & (systemic_labels == "systemic")] = 1.0
            binary[valid & (systemic_labels != "systemic")] = 0.0

            # Filter valid rows
            train_valid = train_mask & valid
            test_valid = test_mask & valid

            if test_valid.sum() == 0:
                continue

            X_train = df.loc[train_valid, feat_cols].values.astype(np.float32)
            y_train = binary[train_valid].values.astype(int)
            X_test = df.loc[test_valid, feat_cols].values.astype(np.float32)
            y_test = binary[test_valid].values.astype(int)

            # Handle NaN
            X_train = np.nan_to_num(X_train, nan=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0)

            if len(np.unique(y_train)) < 2:
                continue

            # Z-score
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Train
            if model_key == "elastic_net_lr":
                clf = LogisticRegression(**model_cfg["params"])
                clf.fit(X_train, y_train)
                y_score = clf.predict_proba(X_test)[:, 1]
                feature_importances += np.abs(clf.coef_[0])
            else:
                spw = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)
                params = dict(model_cfg["params"])
                params["scale_pos_weight"] = spw
                clf = xgb.XGBClassifier(**params)
                clf.fit(X_train, y_train)
                y_score = clf.predict_proba(X_test)[:, 1]
                feature_importances += clf.feature_importances_

            n_folds += 1

            all_y_true.extend(y_test.tolist())
            all_y_score.extend(y_score.tolist())

            if n_folds % 5 == 0:
                print(f"  Completed {n_folds}/{len(unique_pids)} folds...")

        print(f"  Completed all {n_folds} folds.")

        all_y_true = np.array(all_y_true)
        all_y_score = np.array(all_y_score)

        # ---- Metrics ----
        auprc = average_precision_score(all_y_true, all_y_score) if len(np.unique(all_y_true)) >= 2 else None

        # Recall at 50% precision
        if auprc is not None:
            precision_arr, recall_arr, thresholds_arr = precision_recall_curve(all_y_true, all_y_score)
            valid_50 = precision_arr >= 0.50
            recall_at_50 = float(recall_arr[valid_50].max()) if valid_50.any() else 0.0

            # Best F1
            f1_scores = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-10)
            best_f1_idx = np.argmax(f1_scores)
            best_f1 = float(f1_scores[best_f1_idx])
            if best_f1_idx < len(thresholds_arr):
                best_threshold = float(thresholds_arr[best_f1_idx])
            else:
                best_threshold = 0.5

            # Confusion matrix at best F1 threshold
            y_pred = (all_y_score >= best_threshold).astype(int)
            cm = confusion_matrix(all_y_true, y_pred).tolist()
            macro_f1 = float(f1_score(all_y_true, y_pred, average="macro", zero_division=0))
        else:
            recall_at_50 = None
            best_f1 = None
            best_threshold = None
            cm = None
            macro_f1 = None

        print(f"  AUPRC:           {auprc:.4f}" if auprc else "  AUPRC: N/A")
        print(f"  Recall@50%Prec:  {recall_at_50:.4f}" if recall_at_50 is not None else "  Recall@50%Prec: N/A")
        print(f"  Best F1:         {best_f1:.4f}" if best_f1 is not None else "  Best F1: N/A")
        print(f"  Macro F1:        {macro_f1:.4f}" if macro_f1 is not None else "  Macro F1: N/A")

        # ---- Top features ----
        avg_importances = feature_importances / max(n_folds, 1)
        top_indices = np.argsort(avg_importances)[::-1][:20]
        top_features = [(feat_cols[i], round(float(avg_importances[i]), 6)) for i in top_indices]

        print(f"\n  Top 10 features driving systemic crashes:")
        for fname, fval in top_features[:10]:
            print(f"    {fname:<45} {fval:.6f}")

        model_results = {
            "model": model_cfg["name"],
            "auprc": round(float(auprc), 4) if auprc is not None else None,
            "recall_at_50_precision": round(float(recall_at_50), 4) if recall_at_50 is not None else None,
            "best_f1": round(float(best_f1), 4) if best_f1 is not None else None,
            "macro_f1": round(float(macro_f1), 4) if macro_f1 is not None else None,
            "best_threshold": round(float(best_threshold), 4) if best_threshold is not None else None,
            "confusion_matrix": cm,
            "n_total": int(len(all_y_true)),
            "n_systemic": int(all_y_true.sum()),
            "systemic_rate": round(float(all_y_true.mean()), 4),
            "top_features": top_features,
        }

        results["models"][model_key] = model_results

    # ---- Summary ----
    print(f"\n{'=' * 72}")
    print("SUMMARY: SYSTEMIC/CASCADE DETECTOR RESULTS")
    print(f"{'=' * 72}")
    for model_key, model_res in results["models"].items():
        print(f"\n  {model_res['model']}:")
        print(f"    AUPRC:          {model_res['auprc']}")
        print(f"    Recall@50%Prec: {model_res['recall_at_50_precision']}")
        print(f"    Best F1:        {model_res['best_f1']}")
        print(f"    Systemic rate:  {model_res['systemic_rate']}")

    # ---- Save ----
    output_path = RESULTS_DIR / "exp7_systemic_cascade_2plus.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
