
#!/usr/bin/env python3
"""
extract_face_video_features.py

Face-only video feature extractor for conversational AI experiments.

Input:
  - .mp4 (single face, webcam-ish framing)

Output:
  - One-row CSV with a compact set of *interpretable* session-level features:
    * Attention/engagement: focused_pct, switches_per_min, episode lengths, long-away episodes
    * Head pose dynamics: mean abs yaw/pitch, variability, motion energy
    * Affect (one backend):
        - MediaPipe FaceLandmarker blendshapes (default), OR
        - LibreFace AUs (optional), OR
        - OpenFace AUs (optional)

Notes:
  - Hands/torso are NOT required.
  - Designed to avoid "feature explosion" and keep interpretability.

Dependencies (core):
  pip install mediapipe opencv-python numpy

Optional:
  pip install pandas
  pip install libreface   (and its dependencies)
  OpenFace FeatureExtraction binary in PATH or provided via --openface_bin
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import urllib.request

import cv2
import numpy as np
import mediapipe as mp

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "face_landmarker.task"


def ensure_model(model_path: Optional[Path] = None) -> Path:
    """Return the model path, downloading it if necessary."""
    if model_path is not None and model_path.exists():
        return model_path
    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH
    # Download
    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading face_landmarker.task to {DEFAULT_MODEL_PATH} ...")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, str(DEFAULT_MODEL_PATH))
    print("Download complete.")
    return DEFAULT_MODEL_PATH

try:
    import pandas as pd  # optional, used for OpenFace/LibreFace parsing convenience
except Exception:
    pd = None


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    # Attention calibration: assumes user mostly looks at screen in first N seconds.
    calib_seconds: float = 5.0
    min_calib_frames: int = 15

    # Head pose thresholds (degrees) relative to baseline
    yaw_thresh_deg: float = 25.0
    pitch_thresh_deg: float = 20.0

    # Eye (iris-in-eye-box) thresholds relative to baseline (unitless ratio)
    eye_h_thresh: float = 0.18
    eye_v_thresh: float = 0.18

    # Eye openness threshold for blink/unknown
    min_eye_open_ratio: float = 0.18

    # Smoothing + debounce for attention
    smoothing_window_s: float = 0.50
    decision_margin: float = 0.60
    debounce_s: float = 0.45
    face_missing_is_away: bool = True

    # Performance
    resize_width: Optional[int] = 640

    # Affect thresholds (for baseline-corrected signals)
    smile_delta_thresh: float = 0.25
    smile_min_duration_s: float = 0.30
    tension_delta_thresh: float = 0.20  # used only for optional "high rate" if you add it

    # Blink event detection (using eye_open ratio)
    blink_close_thresh: float = 0.18
    blink_min_dur_s: float = 0.05
    blink_max_dur_s: float = 0.60

    # Away episode threshold
    long_away_thresh_s: float = 2.0


# -----------------------------
# MediaPipe FaceMesh landmark indices
# -----------------------------
IDX_NOSE_TIP = 1
IDX_CHIN = 152
IDX_LEFT_EYE_OUTER = 33
IDX_RIGHT_EYE_OUTER = 263
IDX_LEFT_MOUTH = 61
IDX_RIGHT_MOUTH = 291

IDX_LEFT_EYE_INNER = 133
IDX_RIGHT_EYE_INNER = 362

IDX_LEFT_UPPER_LID = 159
IDX_LEFT_LOWER_LID = 145
IDX_RIGHT_UPPER_LID = 386
IDX_RIGHT_LOWER_LID = 374

IDX_LEFT_IRIS = 468
IDX_RIGHT_IRIS = 473


# -----------------------------
# Utilities
# -----------------------------
def safe_div(a: float, b: float, eps: float = 1e-9) -> float:
    return a / (b + eps)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def nanmean(x: np.ndarray) -> float:
    v = float(np.nanmean(x))
    return v if np.isfinite(v) else float("nan")


def nanstd(x: np.ndarray) -> float:
    v = float(np.nanstd(x))
    return v if np.isfinite(v) else float("nan")


def rotation_matrix_to_euler_degrees(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to Euler angles in degrees.
    Returns (pitch, yaw, roll) in degrees.
    Convention is approximate; we rely on relative statistics (not absolute truth).
    """
    assert R.shape == (3, 3)
    sy = float(np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(float(R[2, 1]), float(R[2, 2]))     # pitch-like
        y = math.atan2(float(-R[2, 0]), sy)                # yaw-like
        z = math.atan2(float(R[1, 0]), float(R[0, 0]))     # roll-like
    else:
        x = math.atan2(float(-R[1, 2]), float(R[1, 1]))
        y = math.atan2(float(-R[2, 0]), sy)
        z = 0.0

    pitch = math.degrees(x)
    yaw = math.degrees(y)
    roll = math.degrees(z)
    return pitch, yaw, roll


def median_or_default(values: List[float], default: float) -> float:
    if len(values) == 0:
        return default
    return float(np.median(np.array(values, dtype=np.float64)))


# -----------------------------
# Core estimators (landmark geometry)
# -----------------------------
def compute_eye_metrics(landmarks, w: int, h: int, min_open_ratio: float) -> Dict[str, Optional[float]]:
    """
    Returns:
      - eye_h: avg horizontal iris ratio (0..1, ~0.5 center)
      - eye_v: avg vertical iris ratio (0..1, ~0.5 center)
      - eye_open: avg eye openness ratio (vertical/horizontal)

    If eye_open < min_open_ratio, returns None for eye_h/eye_v (gaze unknown).
    """
    def lm_xy(i: int) -> Tuple[float, float]:
        p = landmarks[i]
        return float(p.x * w), float(p.y * h)

    # Left eye
    l_outer = lm_xy(IDX_LEFT_EYE_OUTER)
    l_inner = lm_xy(IDX_LEFT_EYE_INNER)
    l_upper = lm_xy(IDX_LEFT_UPPER_LID)
    l_lower = lm_xy(IDX_LEFT_LOWER_LID)
    l_iris  = lm_xy(IDX_LEFT_IRIS)

    # Right eye
    r_outer = lm_xy(IDX_RIGHT_EYE_OUTER)
    r_inner = lm_xy(IDX_RIGHT_EYE_INNER)
    r_upper = lm_xy(IDX_RIGHT_UPPER_LID)
    r_lower = lm_xy(IDX_RIGHT_LOWER_LID)
    r_iris  = lm_xy(IDX_RIGHT_IRIS)

    # Eye spans
    l_eye_w = float(np.linalg.norm(np.array(l_inner) - np.array(l_outer)))
    r_eye_w = float(np.linalg.norm(np.array(r_inner) - np.array(r_outer)))

    l_eye_h = float(np.linalg.norm(np.array(l_lower) - np.array(l_upper)))
    r_eye_h = float(np.linalg.norm(np.array(r_lower) - np.array(r_upper)))

    l_open = safe_div(l_eye_h, l_eye_w)
    r_open = safe_div(r_eye_h, r_eye_w)
    eye_open = float((l_open + r_open) / 2.0)

    if eye_open < min_open_ratio:
        return {"eye_h": None, "eye_v": None, "eye_open": eye_open}

    # Iris ratios
    l_h = safe_div((l_iris[0] - l_outer[0]), (l_inner[0] - l_outer[0]))
    r_h = safe_div((r_iris[0] - r_outer[0]), (r_inner[0] - r_outer[0]))

    l_v = safe_div((l_iris[1] - l_upper[1]), (l_lower[1] - l_upper[1]))
    r_v = safe_div((r_iris[1] - r_upper[1]), (r_lower[1] - r_upper[1]))

    l_h, r_h = clamp01(float(l_h)), clamp01(float(r_h))
    l_v, r_v = clamp01(float(l_v)), clamp01(float(r_v))

    return {
        "eye_h": float((l_h + r_h) / 2.0),
        "eye_v": float((l_v + r_v) / 2.0),
        "eye_open": eye_open,
    }


def compute_head_pose(landmarks, w: int, h: int) -> Dict[str, Optional[float]]:
    """
    Head pose via solvePnP using a generic 3D face model and 2D MediaPipe points.
    Returns pitch/yaw/roll in degrees if successful.
    """
    def lm_xy(i: int) -> Tuple[float, float]:
        p = landmarks[i]
        return float(p.x * w), float(p.y * h)

    image_points = np.array([
        lm_xy(IDX_NOSE_TIP),
        lm_xy(IDX_CHIN),
        lm_xy(IDX_LEFT_EYE_OUTER),
        lm_xy(IDX_RIGHT_EYE_OUTER),
        lm_xy(IDX_LEFT_MOUTH),
        lm_xy(IDX_RIGHT_MOUTH),
    ], dtype=np.float64)

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ], dtype=np.float64)

    focal_length = float(w)
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, _tvec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return {"pitch": None, "yaw": None, "roll": None}

    R, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = rotation_matrix_to_euler_degrees(R)
    return {"pitch": float(pitch), "yaw": float(yaw), "roll": float(roll)}


def extract_blendshape_dict(res) -> Optional[Dict[str, float]]:
    """
    Extracts blendshape scores for the first face from a MediaPipe FaceLandmarkerResult.
    Returns a dict mapping blendshape name -> score, or None if not present.
    """
    if not hasattr(res, "face_blendshapes"):
        return None
    if res.face_blendshapes is None or len(res.face_blendshapes) == 0:
        return None

    # face_blendshapes[0] is a Classifications proto-like object with .categories
    cats = getattr(res.face_blendshapes[0], "categories", None)
    if cats is None:
        return None
    out: Dict[str, float] = {}
    for c in cats:
        name = getattr(c, "category_name", None)
        score = getattr(c, "score", None)
        if name is None or score is None:
            continue
        out[str(name)] = float(score)
    return out if len(out) else None


# -----------------------------
# Episode helpers (for booleans over time)
# -----------------------------
@dataclass
class EpisodeStats:
    total_true_s: float
    total_false_s: float
    mean_true_episode_s: float
    mean_false_episode_s: float
    count_true_episodes: int
    count_false_episodes: int
    long_false_episodes: int  # false episodes above threshold


def compute_episode_stats(ts: np.ndarray, state: np.ndarray, total_duration_s: float, long_false_thresh_s: float) -> EpisodeStats:
    """
    Computes run-length statistics for a boolean series sampled at timestamps ts.
    state must be shape (n,) and boolean.
    """
    if len(ts) == 0 or len(state) == 0:
        return EpisodeStats(0.0, 0.0, float("nan"), float("nan"), 0, 0, 0)

    # Ensure sorted
    order = np.argsort(ts)
    ts = ts[order]
    state = state[order]

    true_durs: List[float] = []
    false_durs: List[float] = []

    curr = bool(state[0])
    start = float(ts[0])

    for i in range(1, len(ts)):
        s = bool(state[i])
        if s != curr:
            end = float(ts[i])
            dur = max(0.0, end - start)
            if curr:
                true_durs.append(dur)
            else:
                false_durs.append(dur)
            curr = s
            start = end

    # Close final episode at total_duration_s
    end = float(total_duration_s)
    dur = max(0.0, end - start)
    if curr:
        true_durs.append(dur)
    else:
        false_durs.append(dur)

    total_true = float(np.sum(true_durs)) if len(true_durs) else 0.0
    total_false = float(np.sum(false_durs)) if len(false_durs) else 0.0
    mean_true = float(np.mean(true_durs)) if len(true_durs) else float("nan")
    mean_false = float(np.mean(false_durs)) if len(false_durs) else float("nan")
    long_false = sum(1 for d in false_durs if d >= long_false_thresh_s)

    return EpisodeStats(
        total_true_s=total_true,
        total_false_s=total_false,
        mean_true_episode_s=mean_true,
        mean_false_episode_s=mean_false,
        count_true_episodes=len(true_durs),
        count_false_episodes=len(false_durs),
        long_false_episodes=long_false
    )


# -----------------------------
# Blink detection (from eye openness)
# -----------------------------
@dataclass
class BlinkStats:
    blink_count: int
    blink_rate_per_min: float
    closed_time_s: float
    pct_time_closed: float


def compute_blink_stats(ts: np.ndarray, eye_open: np.ndarray, total_duration_s: float,
                        close_thresh: float, min_dur_s: float, max_dur_s: float) -> BlinkStats:
    """
    Detect blink events as contiguous segments where eye_open < close_thresh.
    Only count segments with duration in [min_dur_s, max_dur_s].
    Also return total time closed (all closed segments, regardless of duration).
    """
    if len(ts) == 0 or len(eye_open) == 0 or total_duration_s <= 0:
        return BlinkStats(0, 0.0, 0.0, 0.0)

    order = np.argsort(ts)
    ts = ts[order]
    eye_open = eye_open[order]

    in_closed = False
    start_t = 0.0
    blink_count = 0
    closed_time = 0.0

    def close_segment(end_t: float):
        nonlocal blink_count, closed_time, in_closed, start_t
        dur = max(0.0, end_t - start_t)
        closed_time += dur
        if min_dur_s <= dur <= max_dur_s:
            blink_count += 1
        in_closed = False

    for i in range(len(ts)):
        t = float(ts[i])
        v = float(eye_open[i])
        if not np.isfinite(v):
            # Unknown -> end any segment
            if in_closed:
                close_segment(t)
            continue

        closed = v < close_thresh
        if closed and not in_closed:
            in_closed = True
            start_t = t
        elif (not closed) and in_closed:
            close_segment(t)

    if in_closed:
        close_segment(total_duration_s)

    rate_per_min = blink_count / (total_duration_s / 60.0) if total_duration_s > 1e-9 else 0.0
    pct_closed = (closed_time / total_duration_s) * 100.0 if total_duration_s > 1e-9 else 0.0
    return BlinkStats(blink_count, float(rate_per_min), float(closed_time), float(pct_closed))


# -----------------------------
# Affect feature computation from a generic time series
# -----------------------------
@dataclass
class AffectStats:
    smile_rate_pct: float
    smile_episode_per_min: float
    mean_smile_episode_s: float
    mean_smile_intensity_when_smiling: float
    tension_mean: float
    tension_sd: float


def compute_affect_stats_from_signals(
    ts: np.ndarray,
    smile_signal: np.ndarray,
    tension_signal: np.ndarray,
    total_duration_s: float,
    calib_seconds: float,
    smile_delta_thresh: float,
    smile_min_duration_s: float,
) -> AffectStats:
    """
    Assumes smile_signal and tension_signal are continuous signals (float).
    We'll baseline-correct using median in the first calib_seconds.
    Smile episodes are counted when (smile - baseline) > smile_delta_thresh.
    """

    def baseline_from_window(sig: np.ndarray) -> float:
        mask = np.isfinite(sig) & (ts <= calib_seconds)
        vals = sig[mask]
        if vals.size < 5:
            return 0.0
        return float(np.median(vals))

    if len(ts) == 0 or total_duration_s <= 0:
        return AffectStats(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))

    ts = np.asarray(ts, dtype=np.float64)
    smile_signal = np.asarray(smile_signal, dtype=np.float64)
    tension_signal = np.asarray(tension_signal, dtype=np.float64)

    b_smile = baseline_from_window(smile_signal)
    b_tension = baseline_from_window(tension_signal)

    smile_delta = smile_signal - b_smile
    tension_delta = tension_signal - b_tension

    # Smile boolean (missing -> False)
    smile_on = (smile_delta > smile_delta_thresh) & np.isfinite(smile_delta)

    # Episode stats for smile_on (True episodes represent smiles)
    ep = compute_episode_stats(ts, smile_on.astype(bool), total_duration_s, long_false_thresh_s=float("inf"))

    # Filter to "real" smiles by min duration (we need durations of True episodes)
    # We'll recompute durations explicitly to apply min duration filter.
    order = np.argsort(ts)
    ts_o = ts[order]
    smile_o = smile_on[order]

    true_durs: List[float] = []
    curr = bool(smile_o[0])
    start = float(ts_o[0])
    for i in range(1, len(ts_o)):
        s = bool(smile_o[i])
        if s != curr:
            end = float(ts_o[i])
            dur = max(0.0, end - start)
            if curr:
                true_durs.append(dur)
            curr = s
            start = end
    # close last
    end = float(total_duration_s)
    dur = max(0.0, end - start)
    if curr:
        true_durs.append(dur)

    true_durs = [d for d in true_durs if d >= smile_min_duration_s]
    smile_time = float(np.sum(true_durs)) if true_durs else 0.0
    smile_rate_pct = (smile_time / total_duration_s) * 100.0 if total_duration_s > 1e-9 else 0.0
    smile_episode_per_min = (len(true_durs) / (total_duration_s / 60.0)) if total_duration_s > 1e-9 else 0.0
    mean_smile_episode_s = float(np.mean(true_durs)) if true_durs else 0.0

    # Intensity when smiling (delta)
    mask_smile_frames = smile_on
    if np.any(mask_smile_frames):
        mean_smile_intensity = float(np.nanmean(smile_delta[mask_smile_frames]))
    else:
        mean_smile_intensity = 0.0

    tension_mean = float(np.nanmean(tension_delta)) if np.any(np.isfinite(tension_delta)) else float("nan")
    tension_sd = float(np.nanstd(tension_delta)) if np.any(np.isfinite(tension_delta)) else float("nan")

    return AffectStats(
        smile_rate_pct=float(smile_rate_pct),
        smile_episode_per_min=float(smile_episode_per_min),
        mean_smile_episode_s=float(mean_smile_episode_s),
        mean_smile_intensity_when_smiling=float(mean_smile_intensity),
        tension_mean=float(tension_mean),
        tension_sd=float(tension_sd),
    )


# -----------------------------
# External backends: LibreFace / OpenFace
# -----------------------------
def _require_pandas() -> None:
    if pd is None:
        raise RuntimeError(
            "pandas is required for parsing LibreFace/OpenFace outputs in this script. "
            "Install it with: pip install pandas"
        )


def run_libreface(video_path: Path, output_csv: Path, device: str = "cpu") -> Path:
    """
    Runs LibreFace CLI and writes per-frame results to output_csv.
    LibreFace CLI documented to support: libreface --input_path=... --output_path=... --device=...
    """
    exe = shutil.which("libreface")
    if exe is None:
        raise RuntimeError(
            "LibreFace CLI 'libreface' not found in PATH. Install with: pip install libreface"
        )

    cmd = [exe, f'--input_path={str(video_path)}', f'--output_path={str(output_csv)}', f'--device={device}']
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "LibreFace failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )
    if not output_csv.exists():
        raise RuntimeError("LibreFace ran but did not produce the expected output CSV.")
    return output_csv


def parse_libreface_affect(libreface_csv: Path, calib_seconds: float, smile_delta_thresh: float, smile_min_duration_s: float) -> AffectStats:
    """
    Parse LibreFace per-frame CSV and compute:
      - smile: AU12 intensity (0..5) normalized to 0..1
      - tension: mean of AU4 intensity (0..5) + AU23/AU24 detection (0/1) if present, normalized
    """
    _require_pandas()
    df = pd.read_csv(libreface_csv)
    if df.empty:
        return AffectStats(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))

    # Time column
    if "frame_time_in_ms" in df.columns:
        ts = df["frame_time_in_ms"].to_numpy(dtype=np.float64) / 1000.0
    elif "timestamp" in df.columns:
        ts = df["timestamp"].to_numpy(dtype=np.float64)
    elif "frame" in df.columns:
        # fallback: assume 30fps if no timestamp
        ts = df["frame"].to_numpy(dtype=np.float64) / 30.0
    else:
        raise RuntimeError("LibreFace CSV missing a usable time column (expected frame_time_in_ms or timestamp).")

    total_duration_s = float(np.nanmax(ts)) if np.isfinite(np.nanmax(ts)) else float("nan")
    if not np.isfinite(total_duration_s) or total_duration_s <= 0:
        total_duration_s = float(ts[-1]) if len(ts) else 0.0

    # Identify AU columns robustly
    # Formats described in PyPI: au_idx and au_idx_intensity (idx is integer) 
    au_det: Dict[int, str] = {}
    au_int: Dict[int, str] = {}

    for col in df.columns:
        m = re.match(r"^au[_]?(\d+)$", col)
        if m:
            au_det[int(m.group(1))] = col
        m = re.match(r"^au[_]?(\d+)_intensity$", col)
        if m:
            au_int[int(m.group(1))] = col

    # Smile signal: AU12 intensity preferred
    if 12 in au_int:
        smile = df[au_int[12]].to_numpy(dtype=np.float64) / 5.0
    elif 12 in au_det:
        smile = df[au_det[12]].to_numpy(dtype=np.float64)  # 0/1
    else:
        smile = np.full(len(df), np.nan, dtype=np.float64)

    # Tension signal: AU4 intensity plus AU23/AU24 detections if available
    parts: List[np.ndarray] = []
    if 4 in au_int:
        parts.append(df[au_int[4]].to_numpy(dtype=np.float64) / 5.0)
    if 23 in au_det:
        parts.append(df[au_det[23]].to_numpy(dtype=np.float64))
    if 24 in au_det:
        parts.append(df[au_det[24]].to_numpy(dtype=np.float64))
    if len(parts) == 0:
        tension = np.full(len(df), np.nan, dtype=np.float64)
    else:
        tension = np.nanmean(np.vstack(parts), axis=0)

    return compute_affect_stats_from_signals(
        ts=ts,
        smile_signal=smile,
        tension_signal=tension,
        total_duration_s=total_duration_s,
        calib_seconds=calib_seconds,
        smile_delta_thresh=smile_delta_thresh,
        smile_min_duration_s=smile_min_duration_s,
    )


def run_openface(video_path: Path, out_dir: Path, openface_bin: Optional[Path]) -> Path:
    """
    Runs OpenFace FeatureExtraction on a video file, writes outputs into out_dir, and returns the CSV path.

    According to OpenFace wiki command line arguments, FeatureExtraction supports:
      FeatureExtraction.exe -f "<video>" -pose
      and -out_dir <directory> for outputs 
    """
    if openface_bin is not None:
        exe = str(openface_bin)
    else:
        # Try in PATH
        exe = shutil.which("FeatureExtraction") or shutil.which("FeatureExtraction.exe")
        if exe is None:
            raise RuntimeError(
                "OpenFace FeatureExtraction not found. Provide --openface_bin or add FeatureExtraction to PATH."
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-f", str(video_path), "-out_dir", str(out_dir), "-aus", "-pose"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "OpenFace FeatureExtraction failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )

    # OpenFace typically outputs <video_basename>.csv in out_dir
    # We'll pick the largest CSV file in the directory.
    csvs = sorted(out_dir.glob("*.csv"), key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    if not csvs:
        raise RuntimeError(f"OpenFace did not produce any CSV in {out_dir}.")
    return csvs[0]


def parse_openface_affect(openface_csv: Path, fps_fallback: float,
                          calib_seconds: float, smile_delta_thresh: float, smile_min_duration_s: float) -> AffectStats:
    """
    Compute affect features from OpenFace per-frame outputs:
      - smile: AU12_r (plus optionally AU06_r)
      - tension: mean of AU04_r, AU07_r, AU23_r, AU24_r (if present)

    OpenFace AUs are typically in columns like AU12_r, AU12_c, etc.
    """
    _require_pandas()
    df = pd.read_csv(openface_csv)
    if df.empty:
        return AffectStats(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))

    # Time
    if "timestamp" in df.columns:
        ts = df["timestamp"].to_numpy(dtype=np.float64)
    elif "frame" in df.columns:
        ts = df["frame"].to_numpy(dtype=np.float64) / float(fps_fallback or 30.0)
    else:
        # very old versions might have "frame" only; handle above
        ts = np.arange(len(df), dtype=np.float64) / float(fps_fallback or 30.0)

    total_duration_s = float(np.nanmax(ts)) if np.isfinite(np.nanmax(ts)) else float("nan")
    if not np.isfinite(total_duration_s) or total_duration_s <= 0:
        total_duration_s = float(ts[-1]) if len(ts) else 0.0

    # Smile signal
    if "AU12_r" in df.columns and "AU06_r" in df.columns:
        smile = 0.5 * (df["AU12_r"].to_numpy(dtype=np.float64) + df["AU06_r"].to_numpy(dtype=np.float64)) / 5.0
    elif "AU12_r" in df.columns:
        smile = df["AU12_r"].to_numpy(dtype=np.float64) / 5.0
    elif "AU12_c" in df.columns:
        smile = df["AU12_c"].to_numpy(dtype=np.float64)  # 0/1
    else:
        smile = np.full(len(df), np.nan, dtype=np.float64)

    # Tension signal (use available)
    tension_cols = [c for c in ["AU04_r", "AU07_r", "AU23_r", "AU24_r"] if c in df.columns]
    if tension_cols:
        tension = df[tension_cols].to_numpy(dtype=np.float64) / 5.0
        tension = np.nanmean(tension, axis=1)
    else:
        tension = np.full(len(df), np.nan, dtype=np.float64)

    return compute_affect_stats_from_signals(
        ts=ts,
        smile_signal=smile,
        tension_signal=tension,
        total_duration_s=total_duration_s,
        calib_seconds=calib_seconds,
        smile_delta_thresh=smile_delta_thresh,
        smile_min_duration_s=smile_min_duration_s,
    )


# -----------------------------
# Main extraction (MediaPipe-based attention + head pose + blink, plus chosen affect backend)
# -----------------------------
def extract_features_from_video(
    video_path: Path,
    model_path: Path,
    cfg: Config,
    affect_backend: str,
    libreface_device: str = "cpu",
    openface_bin: Optional[Path] = None,
) -> Dict[str, Any]:

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-6:
        fps = 30.0

    # MediaPipe FaceLandmarker
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=(affect_backend == "mp_blendshapes"),
        output_facial_transformation_matrixes=False,
    )
    face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    # Attention smoothing
    smooth_win_frames = max(1, int(round(cfg.smoothing_window_s * fps)))
    raw_window = deque(maxlen=smooth_win_frames)

    stable_state: Optional[bool] = None
    candidate_state: Optional[bool] = None
    candidate_start_t: Optional[float] = None
    switches = 0

    # Baseline buffers for attention (yaw/pitch/eye ratios)
    calib_yaw: List[float] = []
    calib_pitch: List[float] = []
    calib_eye_h: List[float] = []
    calib_eye_v: List[float] = []

    baseline = {
        "yaw": 0.0,
        "pitch": 0.0,
        "eye_h": 0.5,
        "eye_v": 0.5
    }
    calibrated = False
    calib_end_t = cfg.calib_seconds

    # Per-frame arrays (lightweight, single video)
    ts: List[float] = []
    stable_att: List[int] = []  # 0/1
    yaw_arr: List[float] = []
    pitch_arr: List[float] = []
    eye_open_arr: List[float] = []
    face_present_arr: List[int] = []

    # For MP blendshapes backend only
    smile_arr: List[float] = []
    tension_arr: List[float] = []
    blendshape_present_arr: List[int] = []

    # Diagnostics counts
    frames_total = 0
    frames_face = 0
    frames_pose_ok = 0
    frames_eye_known = 0
    frames_blend_ok = 0

    prev_t: Optional[float] = None
    prev_timestamp_ms = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_total += 1

        # Timestamp
        t_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        t = t_msec / 1000.0
        if prev_t is None:
            if t <= 0.0:
                t = 0.0
        else:
            if t <= prev_t or (t - prev_t) > 5.0:
                t = prev_t + (1.0 / fps)
        prev_t = t

        # Resize for speed
        if cfg.resize_width is not None:
            h0, w0 = frame.shape[:2]
            if w0 != cfg.resize_width:
                scale = cfg.resize_width / float(w0)
                frame = cv2.resize(frame, (cfg.resize_width, int(round(h0 * scale))), interpolation=cv2.INTER_LINEAR)

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(round(t * 1000))
        if timestamp_ms <= prev_timestamp_ms:
            timestamp_ms = prev_timestamp_ms + 1
        prev_timestamp_ms = timestamp_ms

        res = face_landmarker.detect_for_video(mp_image, timestamp_ms)

        face_present = len(res.face_landmarks) > 0
        face_present_arr.append(1 if face_present else 0)

        pitch = yaw = roll = None
        eye_h = eye_v = eye_open = None

        raw_attending: Optional[bool] = None
        blend_dict: Optional[Dict[str, float]] = None

        if face_present:
            frames_face += 1
            lms = res.face_landmarks[0]

            pose = compute_head_pose(lms, w, h)
            pitch, yaw, roll = pose["pitch"], pose["yaw"], pose["roll"]
            pose_ok = (pitch is not None) and (yaw is not None)
            if pose_ok:
                frames_pose_ok += 1

            eyes = compute_eye_metrics(lms, w, h, min_open_ratio=cfg.min_eye_open_ratio)
            eye_h, eye_v, eye_open = eyes["eye_h"], eyes["eye_v"], eyes["eye_open"]
            if eye_open is not None and np.isfinite(eye_open):
                # eye_open always returned; "known" means iris ratios available
                pass
            if (eye_h is not None) and (eye_v is not None):
                frames_eye_known += 1

            # Affect signals from MP blendshapes (optional)
            if affect_backend == "mp_blendshapes":
                blend_dict = extract_blendshape_dict(res)
                if blend_dict is not None:
                    frames_blend_ok += 1

            # Collect calibration (attention baseline)
            if (not calibrated) and (t <= calib_end_t) and pose_ok and (eye_h is not None) and (eye_v is not None):
                calib_pitch.append(float(pitch))
                calib_yaw.append(float(yaw))
                calib_eye_h.append(float(eye_h))
                calib_eye_v.append(float(eye_v))

            # Finish calibration when time passes
            if (not calibrated) and (t > calib_end_t):
                if len(calib_yaw) >= cfg.min_calib_frames:
                    baseline["yaw"] = median_or_default(calib_yaw, baseline["yaw"])
                    baseline["pitch"] = median_or_default(calib_pitch, baseline["pitch"])
                    baseline["eye_h"] = median_or_default(calib_eye_h, baseline["eye_h"])
                    baseline["eye_v"] = median_or_default(calib_eye_v, baseline["eye_v"])
                calibrated = True

            head_ok = False
            if pose_ok:
                head_ok = (abs(float(yaw) - baseline["yaw"]) <= cfg.yaw_thresh_deg) and \
                          (abs(float(pitch) - baseline["pitch"]) <= cfg.pitch_thresh_deg)

            eyes_ok: Optional[bool] = None
            if (eye_h is not None) and (eye_v is not None):
                eyes_ok = (abs(float(eye_h) - baseline["eye_h"]) <= cfg.eye_h_thresh) and \
                          (abs(float(eye_v) - baseline["eye_v"]) <= cfg.eye_v_thresh)

            if eyes_ok is None:
                raw_attending = head_ok
            else:
                raw_attending = bool(head_ok and eyes_ok)
        else:
            raw_attending = (False if cfg.face_missing_is_away else None)

        # Smoothing majority vote
        raw_window.append(raw_attending)
        valid = [x for x in raw_window if x is not None]
        if len(valid) == 0:
            smoothed = None
        else:
            true_frac = sum(1 for x in valid if x) / float(len(valid))
            false_frac = 1.0 - true_frac
            if true_frac >= cfg.decision_margin:
                smoothed = True
            elif false_frac >= cfg.decision_margin:
                smoothed = False
            else:
                smoothed = None

        if stable_state is None:
            stable_state = smoothed if smoothed is not None else False

        effective = smoothed if smoothed is not None else stable_state

        # Debounce
        if effective != stable_state:
            if candidate_state != effective:
                candidate_state = effective
                candidate_start_t = t
            else:
                if candidate_start_t is None:
                    candidate_start_t = t
                if (t - candidate_start_t) >= cfg.debounce_s:
                    stable_state = candidate_state
                    switches += 1
                    candidate_state = None
                    candidate_start_t = None
        else:
            candidate_state = None
            candidate_start_t = None

        # Store frame signals
        ts.append(float(t))
        stable_att.append(1 if stable_state else 0)
        yaw_arr.append(float(yaw) if yaw is not None else float("nan"))
        pitch_arr.append(float(pitch) if pitch is not None else float("nan"))
        eye_open_arr.append(float(eye_open) if eye_open is not None else float("nan"))

        if affect_backend == "mp_blendshapes":
            if blend_dict is None:
                blendshape_present_arr.append(0)
                smile_arr.append(float("nan"))
                tension_arr.append(float("nan"))
            else:
                blendshape_present_arr.append(1)

                # Smile signal: mean of mouthSmileLeft/Right
                ms_l = float(blend_dict.get("mouthSmileLeft", 0.0))
                ms_r = float(blend_dict.get("mouthSmileRight", 0.0))
                smile = 0.5 * (ms_l + ms_r)

                # Tension signal: mean of browDown + mouthPress + eyeSquint (all L/R)
                bd_l = float(blend_dict.get("browDownLeft", 0.0))
                bd_r = float(blend_dict.get("browDownRight", 0.0))
                mp_l = float(blend_dict.get("mouthPressLeft", 0.0))
                mp_r = float(blend_dict.get("mouthPressRight", 0.0))
                es_l = float(blend_dict.get("eyeSquintLeft", 0.0))
                es_r = float(blend_dict.get("eyeSquintRight", 0.0))
                tension = float(np.mean([bd_l, bd_r, mp_l, mp_r, es_l, es_r]))

                smile_arr.append(float(smile))
                tension_arr.append(float(tension))

    cap.release()
    face_landmarker.close()

    ts_np = np.asarray(ts, dtype=np.float64)
    if len(ts_np) == 0:
        raise RuntimeError(f"No frames processed for video: {video_path}")
    total_duration_s = float(np.nanmax(ts_np)) + (1.0 / fps)
    if not np.isfinite(total_duration_s) or total_duration_s <= 0:
        total_duration_s = float(len(ts_np) / fps)

    # Finalize calibration if the video was shorter than calib_seconds
    if not calibrated:
        if len(calib_yaw) >= 1:
            baseline["yaw"] = median_or_default(calib_yaw, baseline["yaw"])
            baseline["pitch"] = median_or_default(calib_pitch, baseline["pitch"])
            baseline["eye_h"] = median_or_default(calib_eye_h, baseline["eye_h"])
            baseline["eye_v"] = median_or_default(calib_eye_v, baseline["eye_v"])
        calibrated = True

    stable_np = np.asarray(stable_att, dtype=np.int32).astype(bool)
    # Attention stats
    att_ep = compute_episode_stats(ts_np, stable_np, total_duration_s, long_false_thresh_s=cfg.long_away_thresh_s)
    focused_pct = (att_ep.total_true_s / total_duration_s) * 100.0 if total_duration_s > 1e-9 else 0.0
    switches_per_min = switches / (total_duration_s / 60.0) if total_duration_s > 1e-9 else 0.0
    long_away_per_min = att_ep.long_false_episodes / (total_duration_s / 60.0) if total_duration_s > 1e-9 else 0.0

    # Head pose stats (relative to baseline)
    yaw_np = np.asarray(yaw_arr, dtype=np.float64)
    pitch_np = np.asarray(pitch_arr, dtype=np.float64)
    yaw_rel = yaw_np - float(baseline["yaw"])
    pitch_rel = pitch_np - float(baseline["pitch"])

    mean_abs_yaw = float(np.nanmean(np.abs(yaw_rel))) if np.any(np.isfinite(yaw_rel)) else float("nan")
    mean_abs_pitch = float(np.nanmean(np.abs(pitch_rel))) if np.any(np.isfinite(pitch_rel)) else float("nan")
    yaw_sd = nanstd(yaw_rel)
    pitch_sd = nanstd(pitch_rel)

    # Motion energy: average angular speed (deg/s) using finite diffs
    dt = np.diff(ts_np)
    dt[dt <= 1e-6] = 1e-6
    dy = np.diff(yaw_rel)
    dp = np.diff(pitch_rel)
    # mask where both are finite
    mask = np.isfinite(dy) & np.isfinite(dp) & np.isfinite(dt)
    if np.any(mask):
        speed = (np.abs(dy[mask]) + np.abs(dp[mask])) / dt[mask]
        head_motion_energy = float(np.nanmean(speed))
    else:
        head_motion_energy = float("nan")

    # Blink stats
    eye_open_np = np.asarray(eye_open_arr, dtype=np.float64)
    blink = compute_blink_stats(
        ts=ts_np,
        eye_open=eye_open_np,
        total_duration_s=total_duration_s,
        close_thresh=cfg.blink_close_thresh,
        min_dur_s=cfg.blink_min_dur_s,
        max_dur_s=cfg.blink_max_dur_s,
    )

    # Affect stats
    if affect_backend == "mp_blendshapes":
        smile_np = np.asarray(smile_arr, dtype=np.float64)
        tension_np = np.asarray(tension_arr, dtype=np.float64)
        # Use only frames where blendshapes present
        # (missing already NaN, handled by compute_affect_stats_from_signals)
        affect = compute_affect_stats_from_signals(
            ts=ts_np,
            smile_signal=smile_np,
            tension_signal=tension_np,
            total_duration_s=total_duration_s,
            calib_seconds=cfg.calib_seconds,
            smile_delta_thresh=cfg.smile_delta_thresh,
            smile_min_duration_s=cfg.smile_min_duration_s,
        )
        blendshape_available_pct = (frames_blend_ok / frames_total * 100.0) if frames_total > 0 else 0.0
    elif affect_backend == "libreface":
        with tempfile.TemporaryDirectory(prefix="libreface_tmp_") as td:
            tmp_csv = Path(td) / "libreface_frames.csv"
            run_libreface(video_path, tmp_csv, device=libreface_device)
            affect = parse_libreface_affect(tmp_csv, cfg.calib_seconds, cfg.smile_delta_thresh, cfg.smile_min_duration_s)
        blendshape_available_pct = float("nan")
    elif affect_backend == "openface":
        with tempfile.TemporaryDirectory(prefix="openface_tmp_") as td:
            out_dir = Path(td) / "openface_out"
            csv_path = run_openface(video_path, out_dir, openface_bin=openface_bin)
            affect = parse_openface_affect(csv_path, fps_fallback=fps,
                                           calib_seconds=cfg.calib_seconds,
                                           smile_delta_thresh=cfg.smile_delta_thresh,
                                           smile_min_duration_s=cfg.smile_min_duration_s)
        blendshape_available_pct = float("nan")
    else:
        raise ValueError(f"Unknown affect_backend: {affect_backend}")

    summary: Dict[str, Any] = {
        "video": str(video_path),
        "affect_backend": affect_backend,

        # QC
        "duration_s": float(total_duration_s),
        "fps": float(fps),
        "processed_frames": int(frames_total),
        "face_detected_pct": (frames_face / frames_total * 100.0) if frames_total > 0 else 0.0,
        "pose_available_pct": (frames_pose_ok / frames_total * 100.0) if frames_total > 0 else 0.0,
        "eye_known_pct": (frames_eye_known / frames_total * 100.0) if frames_total > 0 else 0.0,
        "blendshape_available_pct": float(blendshape_available_pct),

        # Attention / engagement
        "focused_pct": float(focused_pct),
        "switches_per_min": float(switches_per_min),

        # Head pose dynamics
        "head_motion_energy_deg_per_s": float(head_motion_energy),

        # Blink
        "blink_rate_per_min": float(blink.blink_rate_per_min),

    }

    return summary


# -----------------------------
# CSV writing
# -----------------------------
def write_single_row_csv(row: Dict[str, Any], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # stable column order
    fieldnames = list(row.keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(row)


def append_row_csv(row: Dict[str, Any], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = out_csv.exists()
    if not exists:
        write_single_row_csv(row, out_csv)
        return
    # Read existing header to keep it consistent
    with out_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
    # Merge any new columns from this row into the header
    new_cols = [k for k in row.keys() if k not in header]
    if new_cols:
        # Rewrite the file with the expanded header
        with out_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
        header = header + new_cols
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for er in existing_rows:
                w.writerow({k: er.get(k, "") for k in header})
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writerow({k: row.get(k, "") for k in header})


# -----------------------------
# Between-session delta features
# -----------------------------
DELTA_COLS = {
    "focused_pct":                  "delta_focused_pct",
    "blink_rate_per_min":           "delta_blink_rate_per_min",
    "head_motion_energy_deg_per_s": "delta_head_motion_energy",
}

MAX_PRIOR_COLS = {
    "focused_pct":    "max_prior_focused_pct",
}


def compute_video_deltas(all_sessions_csv: Path) -> Optional[Dict[str, Any]]:
    """
    Compute between-session delta and max-prior features from the
    all-sessions CSV.  Returns a list of dicts (one per session row).

    Delta = current_value - mean(all prior sessions).
    Session 1 gets 0 for deltas and NaN for max-prior.
    """
    try:
        import pandas as _pd
    except ImportError:
        print("pandas is required for delta computation. pip install pandas")
        return None

    if not all_sessions_csv.exists():
        print(f"  SKIP: {all_sessions_csv} not found")
        return None

    df = _pd.read_csv(all_sessions_csv)
    df = df.sort_values("session").reset_index(drop=True)

    results: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        rec: Dict[str, Any] = {"session": row["session"]}

        if i == 0:
            for new_name in DELTA_COLS.values():
                rec[new_name] = 0.0
            for new_name in MAX_PRIOR_COLS.values():
                rec[new_name] = float("nan")
        else:
            prior = df.iloc[:i]
            for orig_col, new_name in DELTA_COLS.items():
                rec[new_name] = float(row[orig_col]) - float(prior[orig_col].mean())
            for orig_col, new_name in MAX_PRIOR_COLS.items():
                rec[new_name] = float(prior[orig_col].max())

        results.append(rec)

    return results


# -----------------------------
# CLI
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1] / "data" / "processed_videos"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract face-only, interpretable video features (attention + head pose + blink + affect) from .mp4."
    )
    p.add_argument("input", type=str,
                   help="Path to a .mp4 file, a directory containing .mp4 files, "
                        "OR a participant ID (e.g. '20') to process all sessions under BASE_DIR/<id>/.")
    p.add_argument("--out_csv", type=str, default=None,
                   help="Output CSV path (only used for single-file / directory mode).")

    p.add_argument("--model_path", type=str, default=None,
                   help="Path to MediaPipe face_landmarker.task model. "
                        "If omitted, auto-downloads to ./models/face_landmarker.task.")

    p.add_argument(
        "--affect_backend",
        type=str,
        default="mp_blendshapes",
        choices=["mp_blendshapes", "libreface", "openface"],
        help="How to compute smile/tension: MediaPipe blendshapes (default), LibreFace AUs, or OpenFace AUs."
    )
    p.add_argument("--libreface_device", type=str, default="cpu", help='LibreFace device, e.g., "cpu" or "cuda:0".')
    p.add_argument("--openface_bin", type=str, default=None, help="Path to OpenFace FeatureExtraction binary/exe.")

    # A few knobs (keep minimal)
    p.add_argument("--calib_seconds", type=float, default=5.0)
    p.add_argument("--resize_width", type=int, default=640)
    p.add_argument("--yaw_thresh", type=float, default=25.0)
    p.add_argument("--pitch_thresh", type=float, default=20.0)

    p.add_argument("--smile_delta_thresh", type=float, default=0.25)
    p.add_argument("--smile_min_duration_s", type=float, default=0.30)

    p.add_argument("--blink_close_thresh", type=float, default=0.18)

    return p.parse_args()


def iter_videos(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    vids = sorted([p for p in input_path.rglob("*.mp4") if p.is_file()])
    return vids


def process_participant(
    pid: str,
    cfg: Config,
    model_path: Path,
    affect_backend: str,
    libreface_device: str,
    openface_bin: Optional[Path],
) -> None:
    """Process all sessions for a single participant, then compute deltas."""
    participant_dir = BASE_DIR / pid
    if not participant_dir.is_dir():
        raise FileNotFoundError(f"Participant directory not found: {participant_dir}")

    out_dir = participant_dir

    # Find session videos (video_only preferred, fall back to full mp4)
    video_only = sorted(participant_dir.glob("session_*_video_only_no_audio.mp4"))
    if not video_only:
        video_only = sorted(participant_dir.glob("session_*.mp4"))
    if not video_only:
        raise RuntimeError(f"No session mp4 files found under {participant_dir}")

    session_pattern = re.compile(r"session_(\d+)")

    all_rows: List[Dict[str, Any]] = []

    for vp in video_only:
        m = session_pattern.search(vp.stem)
        if m is None:
            continue
        sess_num = int(m.group(1))
        session_id = f"P{pid}_S{sess_num:02d}"

        print(f"\n{'='*60}")
        print(f"Processing {session_id}: {vp.name}")
        print(f"{'='*60}")

        try:
            row = extract_features_from_video(
                video_path=vp,
                model_path=model_path,
                cfg=cfg,
                affect_backend=affect_backend,
                libreface_device=libreface_device,
                openface_bin=openface_bin,
            )
        except Exception as e:
            print(f"[ERROR] {session_id}: {e}")
            continue

        row["session"] = sess_num

        # Per-session CSV
        per_session_csv = out_dir / f"{session_id}_video_features.csv"
        write_single_row_csv(row, per_session_csv)
        print(f"Wrote: {per_session_csv}")

        all_rows.append(row)

    if not all_rows:
        print("No sessions processed successfully.")
        return

    # All-sessions CSV
    all_sessions_csv = out_dir / f"P{pid}_all_video_sessions.csv"
    fieldnames = list(all_rows[0].keys())
    with all_sessions_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\nWrote all-sessions: {all_sessions_csv}")

    # Delta features
    print(f"\nComputing between-session delta features ...")
    delta_rows = compute_video_deltas(all_sessions_csv)
    if delta_rows:
        import pandas as _pd
        delta_df = _pd.DataFrame(delta_rows)
        delta_csv = out_dir / f"P{pid}_video_delta_features.csv"
        delta_df.to_csv(delta_csv, index=False)
        print(f"Wrote deltas: {delta_csv}")
        print(delta_df.to_string(index=False))


def main() -> None:
    args = parse_args()

    cfg = Config(
        calib_seconds=float(args.calib_seconds),
        resize_width=(None if int(args.resize_width) <= 0 else int(args.resize_width)),
        yaw_thresh_deg=float(args.yaw_thresh),
        pitch_thresh_deg=float(args.pitch_thresh),
        smile_delta_thresh=float(args.smile_delta_thresh),
        smile_min_duration_s=float(args.smile_min_duration_s),
        blink_close_thresh=float(args.blink_close_thresh),
    )

    model_path = ensure_model(Path(args.model_path).expanduser().resolve() if args.model_path else None)
    openface_bin = Path(args.openface_bin).expanduser().resolve() if args.openface_bin else None

    # "all" mode: process every numeric subdirectory under BASE_DIR
    if args.input.lower() == "all":
        pids = sorted(
            [d.name for d in BASE_DIR.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda x: int(x),
        )
        if not pids:
            raise RuntimeError(f"No participant directories found under {BASE_DIR}")
        print(f"Found {len(pids)} participants: {pids}")
        for pid in pids:
            print(f"\n{'#'*60}")
            print(f"# Participant {pid}")
            print(f"{'#'*60}")
            try:
                process_participant(
                    pid=pid,
                    cfg=cfg,
                    model_path=model_path,
                    affect_backend=str(args.affect_backend),
                    libreface_device=str(args.libreface_device),
                    openface_bin=openface_bin,
                )
            except Exception as e:
                print(f"[ERROR] Participant {pid}: {e}")
        return

    # Participant mode: input is a numeric ID like "20"
    if re.fullmatch(r"\d+", args.input):
        process_participant(
            pid=args.input,
            cfg=cfg,
            model_path=model_path,
            affect_backend=str(args.affect_backend),
            libreface_device=str(args.libreface_device),
            openface_bin=openface_bin,
        )
        return

    # Legacy mode: input is a file or directory
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else None
    vids = iter_videos(input_path)
    if not vids:
        raise RuntimeError(f"No .mp4 videos found under: {input_path}")

    first = True
    written_paths: List[Path] = []
    for vp in vids:
        row = extract_features_from_video(
            video_path=vp,
            model_path=model_path,
            cfg=cfg,
            affect_backend=str(args.affect_backend),
            libreface_device=str(args.libreface_device),
            openface_bin=openface_bin,
        )
        if out_csv is not None:
            if first:
                write_single_row_csv(row, out_csv)
                first = False
            else:
                append_row_csv(row, out_csv)
            if out_csv not in written_paths:
                written_paths.append(out_csv)
        else:
            per_video_csv = vp.parent / f"{vp.stem}_face_features.csv"
            write_single_row_csv(row, per_video_csv)
            written_paths.append(per_video_csv)

    for p in written_paths:
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
