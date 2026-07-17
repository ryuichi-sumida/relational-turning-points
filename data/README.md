# Data directory

The study data is **not distributed with this repository**. It comes from a
longitudinal human-subjects study (24 participants × 10 sessions, with audio
and video recordings) and cannot be shared publicly under the conditions
approved by the Ethics Review Committee on Research with Human Subjects of
Waseda University. De-identified derived data may be available from the
corresponding author on reasonable request.

The scripts in this repository expect the following files here:

```
data/
├── user_assessment_labels.csv        # per-session Q1–Q10 self-report ratings
├── full_features_with_temporals.csv  # 351 multimodal features + labels, one row per (pid, session)
├── all_text_features.csv             # output of features/text.py
├── all_audio_features.csv            # output of features/audio.py
├── all_video_features.csv            # output of features/extract_face_video_features.py
├── excel_files/                      # raw questionnaire / annotation exports
│   ├── annotation_final.xlsx
│   └── user_assessment_final.xlsx
├── processed_videos/<pid>/           # per-participant session recordings + transcripts
└── names.txt                         # participant name variants (used only for name-usage text metrics)
```

Everything in this directory except this README is gitignored.
