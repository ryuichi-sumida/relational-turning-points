"""
Data loading and crash/surge labeling for breakdown detection experiments.

Key design decisions (from Notion page):
- Thresholds computed from LOPO training split only (per fold)
- Pooled SD as main definition
- NO autoregressive features / NO past ratings
- Behavioral features only + optionally session index
"""

import numpy as np
import pandas as pd
from pathlib import Path
from config import (
    DATASET, CONSTRUCTS, CONSTRUCT_Q_MAP, META_COLS, TARGET_COLS,
    TEXT_PREFIXES, AUDIO_PREFIXES, VIDEO_PREFIXES, TEMPORAL_PREFIXES,
)


def load_features_and_labels():
    """
    Load the full feature matrix with temporal features and ground-truth labels.

    Returns:
        df: DataFrame with pid, session_num, all behavioral features, and 5 construct ratings
    """
    df = pd.read_csv(DATASET / "full_features_with_temporals.csv")

    # Rename columns for consistency
    if "pid" not in df.columns and "user_id" in df.columns:
        df = df.rename(columns={"user_id": "pid"})

    df = df.sort_values(["pid", "session_num"]).reset_index(drop=True)
    return df


def get_feature_columns(df, include_session_index=False):
    """
    Get behavioral feature columns only (no AR features, no targets, no meta).
    """
    exclude = META_COLS | TARGET_COLS | {"session", "session_idx", "n_prior_sessions"}
    if not include_session_index:
        exclude.add("session_num")

    feat_cols = [c for c in df.columns if c not in exclude]
    return feat_cols


def get_modality_features(feat_cols, modality="T+A+V"):
    """
    Filter feature columns by modality.

    Args:
        feat_cols: list of all feature column names
        modality: one of "T", "A", "V", "T+A", "T+V", "A+V", "T+A+V"
    """
    def matches_modality(col, prefixes):
        col_lower = col.lower()
        for prefix in prefixes:
            if col_lower.startswith(prefix.lower()):
                return True
            # Also check temporal variants
            for tp in ["delta_", "max_prior_", "ema_", "trend_"]:
                if col_lower.startswith(tp.lower()):
                    rest = col_lower[len(tp):]
                    if rest.startswith(prefix.lower()):
                        return True
        return False

    selected = []
    for col in feat_cols:
        include = False
        if "T" in modality and matches_modality(col, TEXT_PREFIXES):
            include = True
        if "A" in modality and matches_modality(col, AUDIO_PREFIXES):
            include = True
        if "V" in modality and matches_modality(col, VIDEO_PREFIXES):
            include = True

        # Include general temporal features that aren't modality-specific
        if col in ["session_num", "n_prior_sessions"]:
            include = True
        # cumulative/ever features are text-derived
        if col.startswith("cumulative_") or col.startswith("ever_"):
            if "T" in modality:
                include = True

        if include:
            selected.append(col)

    return selected


def get_temporal_vs_session_features(feat_cols):
    """
    Split features into session-level only vs temporal features.

    Returns:
        session_only: features from current session only
        temporal: delta, max_prior, EMA, cumulative, trend features
    """
    temporal = []
    session_only = []
    for col in feat_cols:
        is_temporal = False
        for prefix in TEMPORAL_PREFIXES:
            if col.startswith(prefix):
                is_temporal = True
                break
        if is_temporal:
            temporal.append(col)
        else:
            session_only.append(col)
    return session_only, temporal


def compute_deltas(df, construct):
    """
    Compute session-to-session deltas for a construct per participant.

    Returns:
        Series of deltas (NaN for first session of each participant)
    """
    return df.groupby("pid")[construct].diff()


def compute_crash_surge_labels_lopo(df, construct, train_pids):
    """
    Compute crash/surge labels using 1-SD threshold from training participants only.

    Args:
        df: Full DataFrame
        construct: Which construct to label
        train_pids: List of participant IDs in training set

    Returns:
        threshold_sd: the SD of deltas computed from training set
        labels: Series with values 'crash', 'surge', 'stable', or NaN
    """
    deltas = compute_deltas(df, construct)

    # Compute SD from training participants only
    train_mask = df["pid"].isin(train_pids)
    train_deltas = deltas[train_mask].dropna()
    threshold_sd = train_deltas.std()

    # Apply threshold to ALL data (including test)
    labels = pd.Series("stable", index=df.index)
    labels[deltas <= -threshold_sd] = "crash"
    labels[deltas >= threshold_sd] = "surge"
    labels[deltas.isna()] = np.nan  # First session has no delta

    return threshold_sd, labels


def compute_crash_surge_labels_global(df, construct):
    """
    Compute crash/surge labels using global 1-SD threshold (for descriptive analysis only).

    Returns:
        threshold_sd, labels
    """
    deltas = compute_deltas(df, construct)
    threshold_sd = deltas.dropna().std()

    labels = pd.Series("stable", index=df.index)
    labels[deltas <= -threshold_sd] = "crash"
    labels[deltas >= threshold_sd] = "surge"
    labels[deltas.isna()] = np.nan

    return threshold_sd, labels


def compute_all_construct_labels_global(df):
    """
    Compute crash/surge/stable labels for all 5 constructs using global thresholds.

    Returns:
        label_df: DataFrame with columns like 'label_familiarity', 'delta_familiarity', etc.
        thresholds: dict of construct -> SD threshold
    """
    label_df = pd.DataFrame(index=df.index)
    thresholds = {}

    for construct in CONSTRUCTS:
        deltas = compute_deltas(df, construct)
        sd = deltas.dropna().std()
        thresholds[construct] = sd

        label_df[f"delta_{construct}"] = deltas

        labels = pd.Series("stable", index=df.index)
        labels[deltas <= -sd] = "crash"
        labels[deltas >= sd] = "surge"
        labels[deltas.isna()] = np.nan
        label_df[f"label_{construct}"] = labels

    return label_df, thresholds


def load_human_annotator_labels():
    """
    Load human annotator labels and compute 5 constructs.
    """
    df = pd.read_csv(DATASET / "human_annotator_labels.csv")
    df = df.rename(columns={"user_id": "pid"})
    # Constructs are already computed in this file
    return df.sort_values(["pid", "session"]).reset_index(drop=True)


def load_gemini_labels(modality="text_plus_audio_plus_video"):
    """
    Load Gemini assessment labels and compute 5 constructs from Q1-Q10.

    Args:
        modality: Which Gemini modality to use (default: all three)
    """
    df = pd.read_csv(DATASET / "gemini_assessment_labels.csv")
    df = df.rename(columns={"user_id": "pid"})

    # Filter to requested modality
    df = df[df["modality"] == modality].copy()

    # Compute constructs from Q items
    for construct, q_items in CONSTRUCT_Q_MAP.items():
        df[construct] = df[q_items].mean(axis=1)

    df = df.sort_values(["pid", "session"]).reset_index(drop=True)
    return df


def compute_systemic_crash_label(df, label_df):
    """
    Compute systemic crash label:
    - Systemic = enjoyment crash OR 2+ constructs crash in same transition

    Returns:
        Series with 'systemic', 'modular', 'stable', or NaN
    """
    crash_cols = [f"label_{c}" for c in CONSTRUCTS]
    crash_count = (label_df[crash_cols] == "crash").sum(axis=1)
    enjoyment_crash = label_df["label_enjoyment"] == "crash"

    systemic = pd.Series("stable", index=df.index)
    # Modular = single-construct crash without enjoyment
    single_crash = (crash_count == 1) & ~enjoyment_crash
    systemic[single_crash] = "modular"
    # Systemic = enjoyment crash OR 2+ constructs crash
    systemic_mask = enjoyment_crash | (crash_count >= 2)
    systemic[systemic_mask] = "systemic"

    # NaN where any label is NaN (first sessions)
    any_nan = label_df[crash_cols].isna().any(axis=1)
    systemic[any_nan] = np.nan

    return systemic


def get_lopo_folds(df):
    """
    Generate Leave-One-Participant-Out folds.

    Yields:
        (test_pid, train_pids, train_mask, test_mask)
    """
    unique_pids = sorted(df["pid"].unique())
    for test_pid in unique_pids:
        train_pids = [p for p in unique_pids if p != test_pid]
        train_mask = df["pid"] != test_pid
        test_mask = df["pid"] == test_pid
        yield test_pid, train_pids, train_mask, test_mask
