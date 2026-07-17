import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import re
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import soundfile as sf

import numpy as np
import pandas as pd
import opensmile


# =========================
# Config
# =========================
FEATURE_SET = opensmile.FeatureSet.eGeMAPSv02
FEATURE_LEVEL = opensmile.FeatureLevel.Functionals

MERGE_GAP_S = 0.6
PAD_S = 0.15
MIN_SEG_DUR_S = 0.5
MAX_SEG_DUR_S = 30.0

EARLY_FRAC = 0.2
LATE_FRAC = 0.2

# =========================
# Keep only these 14 paper-friendly dimensions
# =========================
# NOTE: These are openSMILE eGeMAPSv02 functional column names as produced by opensmile.FeatureLevel.Functionals.
# If you ever switch feature_set/level, re-check column names by printing seg_feats.columns.
KEEP_SMILE_COLS = {
    # Prosody (level + variability)
    "F0semitoneFrom27.5Hz_sma3nz_amean",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "loudness_sma3_amean",
    "loudness_sma3_stddevNorm",

    # Voice quality / phonation
    "jitterLocal_sma3nz_amean",
    "shimmerLocaldB_sma3nz_amean",
    "HNRdBACF_sma3nz_amean",

    # Spectral / timbre
    "mfcc1_sma3_amean",
}


# We will compute temporal dynamics only for these base columns
DYNAMICS_BASE_COLS = {
    "F0semitoneFrom27.5Hz_sma3nz_amean",  # slope over time
    "loudness_sma3_amean",                # late-early difference
}


# =========================
# Timestamp parsing
# =========================
_TS_RE = re.compile(r"^(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})(?:\.(?P<ms>\d{1,3}))?$")

# -------------------------
# Emotion model (valence/arousal)
# -------------------------
class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits

_EMOTION_MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

def load_emotion_model(device: str):
    processor = Wav2Vec2Processor.from_pretrained(_EMOTION_MODEL_NAME)
    model = EmotionModel.from_pretrained(_EMOTION_MODEL_NAME).to(device)
    model.eval()
    return processor, model

def predict_arousal_valence(
    wav_path: str,
    processor,
    model,
    device: str,
    target_sr: int = 16000,
) -> tuple[float, float]:
    """
    Returns (arousal, valence) for a single wav file.
    Assumes your segments are already 16k mono PCM (they are).
    """
    x, sr = sf.read(wav_path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != target_sr:
        # Your pipeline already forces 16k; if not, you can resample here.
        raise ValueError(f"Expected {target_sr} Hz but got {sr} for {wav_path}")

    y = processor(x, sampling_rate=sr, return_tensors="pt")
    input_values = y["input_values"].to(device)

    with torch.no_grad():
        _, logits = model(input_values)

    # logits: [batch, 3] = [arousal, dominance, valence]
    arousal = float(logits[0, 0].detach().cpu().item())
    valence = float(logits[0, 2].detach().cpu().item())
    return arousal, valence

def ts_to_seconds(ts: str) -> float:
    ts = ts.strip()
    m = _TS_RE.match(ts)
    if not m:
        raise ValueError(f"Bad timestamp: {ts}")
    h = int(m.group("h"))
    mi = int(m.group("m"))
    s = int(m.group("s"))
    ms = m.group("ms")
    ms = int(ms.ljust(3, "0")) if ms is not None else 0
    return h * 3600 + mi * 60 + s + ms / 1000.0


# =========================
# Audio preprocessing + helpers (ffmpeg)
# =========================
def ensure_wav_16k_mono(
    input_media: str,
    wav_out: str,
    *,
    preprocess: bool = False,
    loudnorm: bool = True,
    remove_dc: bool = True,
) -> str:
    """
    Convert input audio/video to mono 16k PCM wav.
    Optionally apply light preprocessing using ffmpeg filters.

    preprocess=True enables a conservative filter chain:
      - DC removal (highpass)
      - optional loudness normalization (EBU R128)
    """
    os.makedirs(os.path.dirname(wav_out), exist_ok=True)

    filters = []
    if preprocess:
        if remove_dc:
            filters.append("highpass=f=20")
        if loudnorm:
            filters.append("loudnorm=I=-23:TP=-2:LRA=11")

    cmd = ["ffmpeg", "-y", "-i", input_media]

    if filters:
        cmd += ["-af", ",".join(filters)]

    cmd += [
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        "-c:a", "pcm_s16le",
        wav_out,
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_out

def get_wav_duration_s(wav_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        wav_path
    ]
    out = subprocess.check_output(cmd).decode("utf-8").strip()
    return float(out)

def cut_wav_segment(wav_in: str, wav_out: str, start_s: float, end_s: float) -> None:
    os.makedirs(os.path.dirname(wav_out), exist_ok=True)
    dur = max(0.0, end_s - start_s)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}",
        "-t", f"{dur:.3f}",
        "-i", wav_in,
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        wav_out,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# =========================
# Transcript loading + merging
# =========================
def load_transcript_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment="#")
    required = {"start_time", "end_time", "speaker", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")

    df["start_s"] = df["start_time"].apply(ts_to_seconds)
    df["end_s"] = df["end_time"].apply(ts_to_seconds)
    df["dur_s"] = df["end_s"] - df["start_s"]
    df = df[df["dur_s"] > 0].copy()
    df = df.sort_values("start_s").reset_index(drop=True)
    return df

def merge_consecutive_user_rows(df: pd.DataFrame, merge_gap_s: float = MERGE_GAP_S) -> pd.DataFrame:
    rows = []
    i = 0
    n = len(df)

    while i < n:
        row = df.iloc[i]
        if str(row["speaker"]).lower() != "user":
            i += 1
            continue

        start_s = float(row["start_s"])
        end_s = float(row["end_s"])
        topic = row["topic"] if "topic" in df.columns else None
        texts = [str(row["text"])]

        j = i + 1
        while j < n:
            nxt = df.iloc[j]
            if str(nxt["speaker"]).lower() != "user":
                break
            gap = float(nxt["start_s"]) - end_s
            if gap <= merge_gap_s:
                end_s = float(nxt["end_s"])
                texts.append(str(nxt["text"]))
                j += 1
            else:
                break

        rows.append({
            "speaker": "user",
            "topic": topic,
            "start_s": start_s,
            "end_s": end_s,
            "dur_s": end_s - start_s,
            "text": " ".join(t.strip() for t in texts if t and t.strip()),
            "n_utterances_merged": (j - i),
        })
        i = j

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values("start_s").reset_index(drop=True)


# =========================
# Feature extraction (openSMILE)
# =========================
def extract_segment_features(smile: opensmile.Smile, wav_path: str) -> pd.DataFrame:
    feats = smile.process_file(wav_path)
    return feats.reset_index(drop=True)

def weighted_mean(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    X = np.asarray(X, dtype=float)
    denom = np.sum(w)
    if denom <= 0:
        return np.nan * np.ones(X.shape[1], dtype=float)
    return (w[:, None] * X).sum(axis=0) / denom

def weighted_std(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    X = np.asarray(X, dtype=float)
    mu = weighted_mean(X, w)
    denom = np.sum(w)
    if denom <= 0:
        return np.nan * np.ones(X.shape[1], dtype=float)
    var = (w[:, None] * (X - mu) ** 2).sum(axis=0) / denom
    return np.sqrt(np.maximum(var, 0.0))

def slope_over_time(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(t) < 2:
        return np.zeros(X.shape[1], dtype=float)
    t0 = t - t.mean()
    denom = np.sum(t0 ** 2)
    if denom < 1e-12:
        return np.zeros(X.shape[1], dtype=float)
    return (t0[:, None] * (X - X.mean(axis=0))).sum(axis=0) / denom

def late_minus_early(
    X: np.ndarray,
    t_mid: np.ndarray,
    w: np.ndarray,
    session_start: float,
    session_end: float,
    early_frac: float = EARLY_FRAC,
    late_frac: float = LATE_FRAC,
) -> np.ndarray:
    total = session_end - session_start
    if total <= 0:
        return np.zeros(X.shape[1], dtype=float)

    early_end = session_start + early_frac * total
    late_start = session_end - late_frac * total

    early_mask = t_mid <= early_end
    late_mask = t_mid >= late_start

    if early_mask.sum() == 0 or late_mask.sum() == 0:
        return np.zeros(X.shape[1], dtype=float)

    early_mean = weighted_mean(X[early_mask], w[early_mask])
    late_mean = weighted_mean(X[late_mask], w[late_mask])
    return late_mean - early_mean


# =========================
# Turn-taking features (from TSV)
# =========================
def compute_turntaking_features(df_full: pd.DataFrame) -> Dict[str, float]:
    df = df_full.sort_values("start_s").reset_index(drop=True)
    user = df[df["speaker"].str.lower() == "user"].copy()
    system = df[df["speaker"].str.lower() != "user"].copy()

    out: Dict[str, float] = {}
    out["user_total_speech_s"] = float(user["dur_s"].sum()) if len(user) else 0.0
    out["system_total_speech_s"] = float(system["dur_s"].sum()) if len(system) else 0.0
    u = out["user_total_speech_s"]
    b = out["system_total_speech_s"]
    out["user_to_bot_speech_ratio"] = float(u / (b + 1e-9))  # avoid divide-by-zero
    out["n_user_turns_raw"] = int(len(user))
    out["n_system_turns_raw"] = int(len(system))
    out["user_backchannel_ratio"] = float((user["dur_s"] < 0.7).mean()) if len(user) else 0.0

    gaps = []
    overlaps = []
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        cur = df.iloc[i]
        if str(cur["speaker"]).lower() == "user" and str(prev["speaker"]).lower() != "user":
            gap = float(cur["start_s"]) - float(prev["end_s"])
            gaps.append(gap)
            if gap < 0:
                overlaps.append(-gap)

    if gaps:
        gaps_np = np.array(gaps, dtype=float)
        out["gap_mean_s"] = float(gaps_np.mean())
        out["gap_std_s"] = float(gaps_np.std(ddof=0))
        out["gap_p50_s"] = float(np.median(gaps_np))
        out["gap_p90_s"] = float(np.percentile(gaps_np, 90))
        out["overlap_rate_after_system"] = float((gaps_np < 0).mean())
        out["overlap_mean_s"] = float(np.mean(overlaps)) if overlaps else 0.0
    else:
        out.update({
            "gap_mean_s": 0.0, "gap_std_s": 0.0, "gap_p50_s": 0.0, "gap_p90_s": 0.0,
            "overlap_rate_after_system": 0.0, "overlap_mean_s": 0.0
        })

    return out

def report_nan_segments(
    segment_df: pd.DataFrame,
    out_dir: str,
    session_id: str,
    *,
    check_cols=None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Identify segments where openSMILE returned NaNs for important features.

    Returns a DataFrame with one row per problematic segment and writes a CSV.
    """
    if check_cols is None:
        check_cols = [
            "F0semitoneFrom27.5Hz_sma3nz_amean",
            "loudness_sma3_amean",
            "jitterLocal_sma3nz_amean",
            "shimmerLocaldB_sma3nz_amean",
            "HNRdBACF_sma3nz_amean",
            "mfcc1_sma3_amean",
        ]


    # Keep only columns that actually exist
    present_cols = [c for c in check_cols if c in segment_df.columns]
    if not present_cols:
        if verbose:
            print("No requested openSMILE columns found in segment_df.")
        return pd.DataFrame()

    nan_mask = segment_df[present_cols].isna()
    bad_any = nan_mask.any(axis=1)

    if not bad_any.any():
        if verbose:
            print("No NaNs found in key openSMILE features.")
        return pd.DataFrame()

    report = segment_df.loc[bad_any, [
        "participant_id", "session_id", "seg_idx",
        "start_s", "end_s", "dur_s", "mid_s",
        "n_utterances_merged", "seg_wav", "text"
    ]].copy()

    report["nan_features"] = nan_mask.loc[bad_any].apply(
        lambda r: ",".join(c for c, v in r.items() if v),
        axis=1
    )

    report["n_nan"] = nan_mask.loc[bad_any].sum(axis=1).values

    # Optional: include actual feature values for inspection
    for c in present_cols:
        report[c] = segment_df.loc[bad_any, c].values

    report = report.sort_values(
        ["n_nan", "dur_s"],
        ascending=[False, True]
    ).reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{session_id}_segments_with_nan_features.csv")
    report.to_csv(out_csv, index=False)

    if verbose:
        print(f"\n=== Segments with NaNs ({len(report)}) ===")
        print(
            report[[
                "seg_idx", "start_s", "end_s", "dur_s",
                "n_nan", "nan_features", "seg_wav"
            ]].to_string(index=False)
        )
        print(f"Wrote: {out_csv}")

    return report



# =========================
# Main: process one session
# =========================
def process_session(
    transcript_tsv: str,
    wav_16k_mono: str,
    out_dir: str,
    session_id: str,
    participant_id: str,
    *,
    emo_processor=None,
    emo_model=None,
    emo_device: str = "cuda",
    feature_set: Optional[opensmile.FeatureSet] = None,
    feature_level: Optional[opensmile.FeatureLevel] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (segment_df, session_df). Writes CSVs to out_dir.

    wav_16k_mono: path to an already-converted 16kHz mono WAV file.

    Segment CSV still contains full eGeMAPSv02 functionals (handy for debugging/future work).
    Session CSV is trimmed to a compact, paper-friendly set:
      - 10 acoustic + dynamics features (from eGeMAPS)
      - 4 turn-taking features (from transcript)
    """
    os.makedirs(out_dir, exist_ok=True)

    wav_path = wav_16k_mono
    audio_dur = get_wav_duration_s(wav_path)

    # 2) Load transcript
    df_full = load_transcript_tsv(transcript_tsv)
    df_full = df_full[(df_full["start_s"] < audio_dur + 0.01)].copy()

    # 3) Merge user rows
    df_user_merged = merge_consecutive_user_rows(df_full, MERGE_GAP_S)
    if len(df_user_merged) == 0:
        raise RuntimeError("No user segments after merging.")

    # 4) openSMILE extractor
    smile = opensmile.Smile(
        feature_set=feature_set or FEATURE_SET,
        feature_level=feature_level or FEATURE_LEVEL,
    )

    seg_rows = []
    feature_frames = []

    session_start = float(df_full["start_s"].min())
    session_end = float(df_full["end_s"].max())

    # 5) Cut segments + extract features
    for idx, r in df_user_merged.iterrows():
        start = float(r["start_s"]) - PAD_S
        end = float(r["end_s"]) + PAD_S

        start = max(0.0, start)
        end = min(audio_dur, end)

        dur = end - start
        if dur < MIN_SEG_DUR_S:
            continue
        if dur > MAX_SEG_DUR_S:
            end = start + MAX_SEG_DUR_S
            dur = MAX_SEG_DUR_S

        seg_wav = os.path.join(out_dir, "segments", f"{session_id}_userseg_{idx:04d}.wav")
        cut_wav_segment(wav_path, seg_wav, start, end)

        feats = extract_segment_features(smile, seg_wav)
        feature_frames.append(feats)

        seg_rows.append({
            "participant_id": participant_id,
            "session_id": session_id,
            "seg_idx": idx,
            "start_s": start,
            "end_s": end,
            "dur_s": dur,
            "mid_s": 0.5 * (start + end),
            "topic": r.get("topic", None),
            "text": r.get("text", ""),
            "n_utterances_merged": int(r.get("n_utterances_merged", 1)),
            "seg_wav": seg_wav,
        })

    if len(seg_rows) == 0:
        raise RuntimeError("All segments dropped due to MIN_SEG_DUR_S/MAX_SEG_DUR_S.")

    seg_meta = pd.DataFrame(seg_rows).reset_index(drop=True)
    seg_feats = pd.concat(feature_frames, axis=0).reset_index(drop=True)

    # Segment-level output (full eGeMAPS)
    segment_df = pd.concat([seg_meta, seg_feats], axis=1)

    # -------------------------
    # Emotion features per segment
    # -------------------------
    arousals = []
    valences = []
    for p in segment_df["seg_wav"].tolist():
        try:
            a, v = predict_arousal_valence(p, emo_processor, emo_model, device=emo_device)
        except Exception as e:
            # don't crash whole session on one bad segment
            a, v = np.nan, np.nan
            print(f"[warn] emotion inference failed for {p}: {e}")
        arousals.append(a)
        valences.append(v)

    segment_df["arousal"] = arousals
    segment_df["valence"] = valences

    # report weird segments with NaNs
    _ = report_nan_segments(
    segment_df,
    out_dir=out_dir,
    session_id=session_id,
    verbose=True,
    )

    # 6) Session aggregation (compute, but only keep a trimmed set)
    feat_cols = list(seg_feats.columns)
    X = segment_df[feat_cols].to_numpy(dtype=float)
    w = segment_df["dur_s"].to_numpy(dtype=float)
    t_mid = segment_df["mid_s"].to_numpy(dtype=float)

    mu = weighted_mean(X, w)
    slope = slope_over_time(X, t_mid)
    lme = late_minus_early(X, t_mid, w, session_start, session_end)

    feat_idx = {c: j for j, c in enumerate(feat_cols)}

    def get_mu(name: str) -> float:
        return float(mu[feat_idx[name]]) if name in feat_idx else np.nan

    def get_slope(name: str) -> float:
        return float(slope[feat_idx[name]]) if name in feat_idx else np.nan

    def get_lme(name: str) -> float:
        return float(lme[feat_idx[name]]) if name in feat_idx else np.nan

    tt = compute_turntaking_features(df_full)

    max_arousal = float(np.nanmax(segment_df["arousal"].to_numpy(dtype=float)))
    max_valence = float(np.nanmax(segment_df["valence"].to_numpy(dtype=float)))


    # Build compact, paper-friendly session row
    session_row = {
        # Turn-taking: keep only the 4 core features
        "user_to_bot_speech_ratio": float(tt.get("user_to_bot_speech_ratio", 0.0)),
        "user_backchannel_ratio": float(tt.get("user_backchannel_ratio", 0.0)),
        "gap_mean_s": float(tt.get("gap_mean_s", 0.0)),
        "overlap_rate_after_system": float(tt.get("overlap_rate_after_system", 0.0)),

        # Acoustic: 14 total including dynamics
        # Prosody
        "F0_mean": get_mu("F0semitoneFrom27.5Hz_sma3nz_amean"),
        "loudness_mean": get_mu("loudness_sma3_amean"),
        "F0_std": get_mu("F0semitoneFrom27.5Hz_sma3nz_stddevNorm"),
        "loudness_std": get_mu("loudness_sma3_stddevNorm"),

        # Voice quality
        "jitter_mean": get_mu("jitterLocal_sma3nz_amean"),
        "shimmer_mean": get_mu("shimmerLocaldB_sma3nz_amean"),
        "hnr_mean": get_mu("HNRdBACF_sma3nz_amean"),

        # Spectral
        "mfcc1_mean": get_mu("mfcc1_sma3_amean"),

        # Dynamics
        "F0_slope": get_slope("F0semitoneFrom27.5Hz_sma3nz_amean"),
        "loudness_late_minus_early": get_lme("loudness_sma3_amean"),

        # NEW (2 more features)
        "arousal_max": max_arousal,
        "valence_max": max_valence,

    }

    session_df = pd.DataFrame([session_row])

    seg_out = os.path.join(out_dir, f"{session_id}_segment_features.csv")
    ses_out = os.path.join(out_dir, f"{session_id}_session_features.csv")
    segment_df.to_csv(seg_out, index=False)
    session_df.to_csv(ses_out, index=False)

    return segment_df, session_df


# =========================
# Between-session delta features
# =========================
DELTA_COLS = {
    "user_to_bot_speech_ratio": "delta_user_speech_time",
    "gap_mean_s":               "delta_gap_mean_s",
    "F0_std":                   "delta_F0_std",
    "valence_max":              "delta_valence_max",
}

MAX_PRIOR_COLS = {
    "valence_max": "max_prior_valence_max",
    "arousal_max": "max_prior_arousal_max",
}


def compute_delta_for_participant(pid: int, base: str) -> Optional[pd.DataFrame]:
    """
    Compute between-session (delta) audio features for one participant.

    Output columns per session:
      - delta_user_speech_time        (current - mean of prior sessions)
      - delta_gap_mean_s              (current - mean of prior sessions)
      - delta_F0_std                  (current - mean of prior sessions)
      - delta_valence_max             (current - mean of prior sessions)
      - max_prior_valence_max         (max valence_max from all prior sessions)
      - max_prior_arousal_max         (max arousal_max from all prior sessions)

    Session 1 gets 0 for deltas and NaN for max-prior.
    """
    base_path = Path(base)
    csv_path = base_path / str(pid) / "actual_features" / f"P{pid}_all_sessions.csv"
    if not csv_path.exists():
        print(f"  SKIP P{pid}: {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)
    df = df.sort_values("session").reset_index(drop=True)

    results = []
    for i, row in df.iterrows():
        rec = {"session": row["session"]}

        if i == 0:
            for new_name in DELTA_COLS.values():
                rec[new_name] = 0.0
            for new_name in MAX_PRIOR_COLS.values():
                rec[new_name] = np.nan
        else:
            prior = df.iloc[:i]
            for orig_col, new_name in DELTA_COLS.items():
                rec[new_name] = row[orig_col] - prior[orig_col].mean()
            for orig_col, new_name in MAX_PRIOR_COLS.items():
                rec[new_name] = prior[orig_col].max()

        results.append(rec)

    return pd.DataFrame(results)


def compute_all_deltas(base: str, participant_ids: list[int]) -> None:
    """Compute and save delta features for all participants."""
    base_path = Path(base)
    for pid in participant_ids:
        print(f"Processing P{pid} deltas...")
        out = compute_delta_for_participant(pid, base)
        if out is None:
            continue

        out_dir = base_path / str(pid) / "actual_features"
        out_path = out_dir / f"P{pid}_delta_features.csv"
        out.to_csv(out_path, index=False)
        print(f"  Saved {out_path}")
        print(out.to_string(index=False))
        print()


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    import glob

    base = str(Path(__file__).resolve().parents[1] / "data" / "processed_videos")

    # Load emotion model once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emo_processor, emo_model = load_emotion_model(device)

    user_dirs = sorted(
        [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))],
        key=lambda x: int(x),
    )

    for user_id_str in user_dirs:
        core_path = os.path.join(base, user_id_str)
        user_id = int(user_id_str)

        # Find all session transcripts for this user
        tsv_files = sorted(glob.glob(os.path.join(core_path, "session_*_transcript.tsv")))

        for tsv_path in tsv_files:
            # Extract session number from filename
            fname = os.path.basename(tsv_path)  # e.g. session_03_transcript.tsv
            sess_num = int(fname.split("_")[1])

            media_path = os.path.join(core_path, f"session_{sess_num:02d}_audio_only_16k_mono.wav")
            if not os.path.exists(media_path):
                print(f"[skip] missing audio: {media_path}")
                continue

            out_dir = os.path.join(core_path, "actual_features")
            participant_id = f"P{user_id:02d}"
            session_id = f"P{user_id:02d}_S{sess_num:02d}"

            print(f"\n{'='*60}")
            print(f"Processing {session_id} ...")
            print(f"{'='*60}")

            try:
                seg_df, ses_df = process_session(
                    transcript_tsv=tsv_path,
                    wav_16k_mono=media_path,
                    out_dir=out_dir,
                    session_id=session_id,
                    participant_id=participant_id,
                    emo_processor=emo_processor,
                    emo_model=emo_model,
                    emo_device=device,
                )
                print(f"[done] {session_id}: {len(seg_df)} segments")
            except Exception as e:
                print(f"[ERROR] {session_id}: {e}")
                continue

    # After all sessions are processed, compute between-session deltas
    participant_ids = [int(d) for d in user_dirs]
    compute_all_deltas(base, participant_ids)
