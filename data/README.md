# Data directory

The study data is **not distributed with this repository**. It comes from a
longitudinal human-subjects study (24 participants × 10 sessions, with audio
and video recordings) and cannot be shared publicly under the conditions
approved by the Ethics Review Committee on Research with Human Subjects of
Waseda University. De-identified derived data may be available from the
corresponding author on reasonable request.

This page documents the layout and file formats the scripts expect, so you
can (a) understand exactly what each pipeline stage consumes and produces, or
(b) run the pipeline on your own recordings of a comparable study.
Everything in this directory except this README is gitignored.

## Directory layout

```
data/
├── processed_videos/            # raw per-participant session recordings (pipeline INPUT)
│   └── <pid>/                   #   one directory per participant, e.g. 20/
│       ├── session_01.mp4                     # full recording (webcam video + audio)
│       ├── session_01_audio_only_16k_mono.wav # extracted audio, 16 kHz mono
│       ├── session_01_video_only_no_audio.mp4 # video stream only
│       ├── session_01_transcript.tsv          # time-aligned ASR transcript (format below)
│       └── ...                                # session_02 … session_10
├── names.txt                    # participant name variants (optional; format below)
├── user_assessment_labels.csv   # post-session questionnaire ratings (format below)
├── excel_files/                 # raw questionnaire / annotation exports
│   ├── annotation_final.xlsx
│   └── user_assessment_final.xlsx
├── all_text_features.csv        # OUTPUT of features/text.py
├── all_audio_features.csv       # OUTPUT of features/audio.py
├── all_video_features.csv       # OUTPUT of features/extract_face_video_features.py
├── text_cache/                  # LLM annotation cache (created automatically)
└── full_features_with_temporals.csv  # merged feature matrix (input to crash_surge/ and
                                       # analysis/robustness_analyses.py)
```

## Session recordings (`processed_videos/<pid>/`)

Each participant directory is named by a numeric participant id (`20/`,
`21/`, …) and holds ten sessions named `session_01` … `session_10`. Each
session is a ~5-minute open-domain conversation with the agent, recorded via
the participant's webcam. The feature extractors read:

- `session_NN.mp4` — used by the video pipeline (face/attention features)
- `session_NN_audio_only_16k_mono.wav` — used by the audio pipeline
  (openSMILE eGeMAPS, wav2vec valence/arousal, turn-taking)
- `session_NN_transcript.tsv` — used by the text pipeline

The extractors also write their per-session intermediate CSVs back into the
participant directory (e.g. `P20_S01_video_features.csv`,
`actual_features/P20_S01_session_features.csv`).

## ASR transcript format (`session_NN_transcript.tsv`)

Tab-separated, one row per speaker turn. Lines starting with `#` are
comments; a header row is optional (recognized by `start_time` in the first
column). Columns:

| column       | example        | notes                              |
|--------------|----------------|------------------------------------|
| `start_time` | `00:00:02.844` | turn start, `HH:MM:SS.mmm`         |
| `end_time`   | `00:00:07.500` | turn end                           |
| `topic`      | `introduction` | dialogue-phase tag (not required by the parser) |
| `speaker`    | `user`         | `user` or `system`                 |
| `text`       | `Hi, nice to see you again!` | the transcribed utterance |

Example:

```
# format: intella-transcript-v2.0
start_time	end_time	topic	speaker	text
00:00:02.844	00:00:07.500	introduction	system	Welcome back! Last time you mentioned your exam — how did it go?
00:00:07.900	00:00:12.156	introduction	user	It went pretty well, thanks for remembering!
```

Only `start_time`, `end_time`, `speaker`, and `text` are used by the code
(see `parse_tsv` in `features/text.py`); any transcript source that produces
these fields will work.

## Self-report labels (`user_assessment_labels.csv`)

One row per (participant, session): the ten 7-point Likert items from the
post-session questionnaire plus the five construct scores (each the mean of
two items, see Table 1 in the paper).

| column | meaning |
|--------|---------|
| `user_id` | participant id (matches `processed_videos/<pid>/`) |
| `session` | session number, 1–10 |
| `Q1`–`Q10` | Likert ratings, 1–7 |
| `familiarity` | mean(Q1, Q2) |
| `social_penetration` | mean(Q3, Q4) — self-disclosure comfort |
| `memory` | mean(Q5, Q6) — perceived memory |
| `conversational` | mean(Q7, Q8) — conversational quality |
| `enjoyment` | mean(Q9, Q10) |

Example row (illustrative values):

```csv
user_id,session,Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,familiarity,social_penetration,memory,conversational,enjoyment
12,3,6.0,5.0,4.0,5.0,6.0,5.0,7.0,6.0,6.0,7.0,5.5,4.5,5.5,6.5,6.5
```

## Merged feature matrix (`full_features_with_temporals.csv`)

The input to all `crash_surge/` experiments: one row per (participant,
session) with the behavioral features from all three modalities —
session-level, temporal (cross-session deltas/trends), and person-normalized
— plus the five construct ratings as target columns. Meta columns are `pid`
and `session_num`; every other non-target column is treated as a feature
(see `crash_surge/config.py` for the modality prefix groupings and exclusion
rules).

## Participant names (`names.txt`, optional)

Used only by `features/text.py` to compute name-usage metrics (does the
agent address the user by name?). Plain text containing
`(<user_id>, "<name variant>")` tuples, one or more per participant:

```
(20, "Alice") (20, "Ali") (24, "Bob")
```

If the file is absent, name-usage metrics are simply skipped — everything
else runs normally.

## Running on your own data

1. Arrange recordings and transcripts under `data/processed_videos/<pid>/`
   as above.
2. Collect post-session ratings into `data/user_assessment_labels.csv`.
3. Run the extractors in `features/` to produce the per-modality CSVs, and
   merge them (with temporal/person-normalized derivatives) into
   `data/full_features_with_temporals.csv`.
4. Run the `crash_surge/` experiments and `analysis/` scripts.
