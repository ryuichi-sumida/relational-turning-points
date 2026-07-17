"""
Compute between-session (delta) audio features for each participant.

Output columns per session:
  - delta_user_speech_time        (current - mean of prior sessions)
  - delta_gap_mean_s              (current - mean of prior sessions)
  - delta_F0_std                  (current - mean of prior sessions)
  - delta_valence_max             (current - mean of prior sessions)
  - max_prior_valence_max         (max valence_max from all prior sessions)
  - max_prior_arousal_max         (max arousal_max from all prior sessions)

Session 1 gets 0 for deltas and NaN for max-prior.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "data" / "processed_videos"

PARTICIPANT_IDS = [20,21,22,24,25,26,27,28,32,33,34,35,36,38,41,45,46,47,48,50,52,53,55,58]

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


def compute_for_participant(pid: int) -> pd.DataFrame | None:
    csv_path = BASE / str(pid) / "actual_features" / f"P{pid}_all_sessions.csv"
    if not csv_path.exists():
        print(f"  SKIP P{pid}: {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)
    # Sort by session number
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


def main():
    for pid in PARTICIPANT_IDS:
        print(f"Processing P{pid}...")
        out = compute_for_participant(pid)
        if out is None:
            continue

        out_dir = BASE / str(pid) / "actual_features"
        out_path = out_dir / f"P{pid}_delta_features.csv"
        out.to_csv(out_path, index=False)
        print(f"  Saved {out_path}")
        print(out.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
