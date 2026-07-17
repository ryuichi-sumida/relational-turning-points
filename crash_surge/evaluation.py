"""
Evaluation framework for breakdown detection experiments.

Metrics:
- Primary: Global AUPRC (concatenated across all LOPO folds)
- Secondary: Per-participant distribution (median/IQR)
- Recall at 50% precision
- F1-score (macro and per-class)
- Per-construct breakdown
"""

import numpy as np
import json
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    f1_score, classification_report, confusion_matrix,
)
from config import RESULTS_DIR


def compute_auprc(y_true, y_score):
    """
    Compute Area Under the Precision-Recall Curve.

    Args:
        y_true: binary labels (1 = positive class)
        y_score: predicted probabilities for positive class
    """
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_score)


def compute_recall_at_precision(y_true, y_score, target_precision=0.5):
    """
    Compute recall at a target precision level.

    Returns:
        recall: best recall achievable at >= target_precision
        threshold: corresponding threshold
    """
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # Find best recall where precision >= target
    valid = precision >= target_precision
    if not valid.any():
        return 0.0, np.nan

    best_idx = np.argmax(recall[valid])
    valid_indices = np.where(valid)[0]
    idx = valid_indices[best_idx]

    if idx < len(thresholds):
        return recall[idx], thresholds[idx]
    return recall[idx], np.nan


def compute_f1_metrics(y_true, y_pred):
    """
    Compute F1 metrics.

    Returns:
        dict with macro_f1, per_class_f1, confusion_matrix
    """
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "macro_f1": float(macro_f1),
        "per_class_f1": [float(x) for x in per_class],
        "confusion_matrix": cm.tolist(),
    }


def evaluate_detector_lopo(df, feat_cols, construct, train_predict_fn,
                           label_fn=None, include_session_index=False):
    """
    Run full LOPO evaluation for a crash detector.

    Args:
        df: Full DataFrame with features and labels
        feat_cols: Feature columns to use
        construct: Which construct to detect crashes for
        train_predict_fn: function(X_train, y_train, X_test) -> y_score (probabilities)
        label_fn: optional custom labeling function
        include_session_index: whether to include session_num as feature

    Returns:
        dict with all metrics
    """
    from data_loader import get_lopo_folds, compute_crash_surge_labels_lopo

    all_y_true = []
    all_y_score = []
    per_participant = {}

    feature_cols = list(feat_cols)
    if include_session_index and "session_num" not in feature_cols:
        feature_cols.append("session_num")

    for test_pid, train_pids, train_mask, test_mask in get_lopo_folds(df):
        # Compute labels using training-set thresholds
        threshold_sd, labels = compute_crash_surge_labels_lopo(
            df, construct, train_pids
        )

        # Binary: crash=1, not-crash=0
        binary_labels = (labels == "crash").astype(int)

        # Filter to rows with valid deltas (exclude first sessions)
        valid_mask = labels.notna()
        train_valid = train_mask & valid_mask
        test_valid = test_mask & valid_mask

        if test_valid.sum() == 0:
            continue

        X_train = df.loc[train_valid, feature_cols].values.astype(np.float32)
        y_train = binary_labels[train_valid].values
        X_test = df.loc[test_valid, feature_cols].values.astype(np.float32)
        y_test = binary_labels[test_valid].values

        # Handle NaN in features
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        if len(np.unique(y_train)) < 2:
            continue

        y_score = train_predict_fn(X_train, y_train, X_test)

        all_y_true.extend(y_test.tolist())
        all_y_score.extend(y_score.tolist())

        # Per-participant metrics
        if len(np.unique(y_test)) >= 2:
            per_participant[int(test_pid)] = {
                "auprc": float(compute_auprc(y_test, y_score)),
                "n_crashes": int(y_test.sum()),
                "n_total": int(len(y_test)),
            }

    if not all_y_true:
        return {"error": "No valid folds"}

    all_y_true = np.array(all_y_true)
    all_y_score = np.array(all_y_score)

    # Global AUPRC (primary metric)
    global_auprc = compute_auprc(all_y_true, all_y_score)

    # Recall at 50% precision
    recall_at_50, threshold_50 = compute_recall_at_precision(
        all_y_true, all_y_score, 0.5
    )

    # Recall at 70% precision
    recall_at_70, threshold_70 = compute_recall_at_precision(
        all_y_true, all_y_score, 0.7
    )

    # F1 at optimal threshold (max F1)
    precision, recall, thresholds = precision_recall_curve(all_y_true, all_y_score)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = float(f1_scores[best_f1_idx])
    if best_f1_idx < len(thresholds):
        best_threshold = float(thresholds[best_f1_idx])
    else:
        best_threshold = 0.5

    # Binary predictions at best threshold
    y_pred = (all_y_score >= best_threshold).astype(int)
    f1_metrics = compute_f1_metrics(all_y_true, y_pred)

    # Per-participant AUPRC distribution
    pp_auprcs = [v["auprc"] for v in per_participant.values()
                 if not np.isnan(v["auprc"])]

    results = {
        "construct": construct,
        "global_auprc": float(global_auprc) if not np.isnan(global_auprc) else None,
        "recall_at_50_precision": float(recall_at_50) if not np.isnan(recall_at_50) else None,
        "recall_at_70_precision": float(recall_at_70) if not np.isnan(recall_at_70) else None,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "macro_f1": f1_metrics["macro_f1"],
        "confusion_matrix": f1_metrics["confusion_matrix"],
        "n_total": int(len(all_y_true)),
        "n_crashes": int(all_y_true.sum()),
        "crash_rate": float(all_y_true.mean()),
        "per_participant_auprc_median": float(np.median(pp_auprcs)) if pp_auprcs else None,
        "per_participant_auprc_iqr": (
            float(np.percentile(pp_auprcs, 25)),
            float(np.percentile(pp_auprcs, 75))
        ) if len(pp_auprcs) >= 4 else None,
        "per_participant": per_participant,
    }

    return results


def save_results(results, filename):
    """Save results to JSON file in results directory."""
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved results to {path}")
    return path
