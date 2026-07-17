import os
import re
import subprocess
from typing import Dict, Tuple, Optional

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
# Timestamp parsing
# =========================
_TS_RE = re.compile(r"^(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})(?:\.(?P<ms>\d{1,3}))?$")

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
            # Remove DC + very low rumble (very gentle)
            filters.append("highpass=f=20")
        if loudnorm:
            # Loudness normalization; good for across-session comparability.
            # This is 2-pass capable, but 1-pass is often fine for research pipelines.
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


# =========================
# Main: process one session
# =========================
def process_session(
    transcript_tsv: str,
    audio_wav_or_media: str,
    out_dir: str,
    session_id: str,
    participant_id: str,
    *,
    preprocess_audio: bool = True,
    feature_set: Optional[opensmile.FeatureSet] = None,
    feature_level: Optional[opensmile.FeatureLevel] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (segment_df, session_df). Writes CSVs to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Preprocess / standardize audio (this replaces the MoviePy step)
    wav_path = os.path.join(out_dir, f"{session_id}.mono16k.wav")
    ensure_wav_16k_mono(
        audio_wav_or_media,
        wav_path,
        preprocess=preprocess_audio,
        loudnorm=True,
        remove_dc=True,
    )

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
    segment_df = pd.concat([seg_meta, seg_feats], axis=1)

    # 6) Session aggregation
    feat_cols = list(seg_feats.columns)
    X = segment_df[feat_cols].to_numpy(dtype=float)
    w = segment_df["dur_s"].to_numpy(dtype=float)
    t_mid = segment_df["mid_s"].to_numpy(dtype=float)

    mu = weighted_mean(X, w)
    sd = weighted_std(X, w)
    slope = slope_over_time(X, t_mid)
    lme = late_minus_early(X, t_mid, w, session_start, session_end)

    tt = compute_turntaking_features(df_full)

    session_row = {
        "participant_id": participant_id,
        "session_id": session_id,
        "n_user_segments": int(len(segment_df)),
        "audio_dur_s": float(audio_dur),
        "session_start_s": float(session_start),
        "session_end_s": float(session_end),
        **tt
    }

    for j, c in enumerate(feat_cols):
        session_row[f"smile_wmean__{c}"] = float(mu[j])
        session_row[f"smile_wstd__{c}"] = float(sd[j])
        session_row[f"smile_slope__{c}"] = float(slope[j])
        session_row[f"smile_late_minus_early__{c}"] = float(lme[j])

    session_df = pd.DataFrame([session_row])

    seg_out = os.path.join(out_dir, f"{session_id}_segment_features.csv")
    ses_out = os.path.join(out_dir, f"{session_id}_session_features.csv")
    segment_df.to_csv(seg_out, index=False)
    session_df.to_csv(ses_out, index=False)

    return segment_df, session_df


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    core_path = "data/example_session"  # directory containing transcript.tsv and user.mp4
    transcript_tsv = f"{core_path}/transcript.tsv"
    media_path = f"{core_path}/user.mp4"   # can also be wav/mp3/etc
    out_dir = f"{core_path}/features"
    participant_id = "P01"
    session_id = "P01_S03"

    # If you want ComParE instead of eGeMAPS, pass it here:
    # feature_set = opensmile.FeatureSet.ComParE_2016
    # feature_level = opensmile.FeatureLevel.Functionals

    seg_df, ses_df = process_session(
        transcript_tsv=transcript_tsv,
        audio_wav_or_media=media_path,
        out_dir=out_dir,
        session_id=session_id,
        participant_id=participant_id,
        preprocess_audio=True,
        # feature_set=feature_set,
        # feature_level=feature_level,
    )

    print("Segment features:", seg_df.shape)
    print("Session features:", ses_df.shape)
    print(ses_df.iloc[0][["participant_id", "session_id", "n_user_segments", "user_total_speech_s", "gap_mean_s"]])
