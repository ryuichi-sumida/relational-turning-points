# Memory-Driven Self-Disclosure and Relational Turning Points

Analysis code for the ICMI '26 paper:

> Ryuichi Sumida, Mao Saeki, Masaki Eguchi, Sadahiro Yoshikawa, Koji Inoue,
> Tatsuya Kawahara, and Yoichi Matsuyama. 2026. **Memory-Driven Self-Disclosure
> and Relational Turning Points: A Longitudinal Multimodal Study of Human-AI
> Interaction.** In *International Conference on Multimodal Interaction
> (ICMI '26), October 05–09, 2026, Napoli, Italy.* ACM.
> https://doi.org/10.1145/3776574.3831135

Preprint: [arXiv:2607.14593](https://arxiv.org/abs/2607.14593)

We study how human-AI relationships develop over repeated interaction: 24
participants talked with a memory-augmented conversational agent (InteLLA)
across 10 daily sessions and rated five relational constructs after each
session (Familiarity, Social Penetration / self-disclosure comfort, Perceived
Memory, Conversational Quality, Enjoyment). The code here covers multimodal
feature extraction (text / audio / video), cross-session temporal modeling of
the constructs, and event-based detection and forecasting of relational
*crashes* and *surges*.

## Repository structure

```
features/      Multimodal feature extraction (351 features)
  text.py                        GPT-based turn-level text annotation → text features
  semantic_judge.py              LLM-as-judge semantic features (memory accuracy, attunement, …)
  audio.py                       openSMILE eGeMAPS + wav2vec valence/arousal + turn-taking
  opensmile_final.py             openSMILE extraction runner
  preprocessing_opensmile_full.py  audio preprocessing / segmentation
  compute_delta_features.py      temporal / person-normalized deltas
  extract_face_video_features.py MediaPipe-based face & attention features
  topic_counter.py               topic diversity / novelty features
  annotation_validation.py       LLM-vs-human annotation agreement (Cohen's κ)

analysis/      Construct-level statistical analyses
  cfa_analysis.py                5-factor CFA + HTMT discriminant validity
  riclpm_analysis.py             Random-intercept cross-lagged panel model
  rerun_clpm_and_path_diagram.py 5×5 cross-lagged panel model + path diagram
  concurrent_regression_all.py   Same-session construct regressions
  growth_curve_summary.py        Within-person growth curves
  mediation_memory_disclosure_enjoyment.py  Memory → Disclosure → Enjoyment mediation
  robustness_analyses.py         Permutation tests, jackknife LOPO, bootstrap CIs
  reduced_factor_robustness.py   Reduced-factor robustness of crash–surge asymmetry
  compare_annotations.py         Self-report ↔ annotator comparison
  fig_mediation_diagram.py       Mediation figure

crash_surge/   Turning-point (crash / surge) detection & forecasting
  config.py, data_loader.py, evaluation.py   Shared config, data loading, LOPO evaluation
  exp1_event_counts.py           Event counts per construct
  exp2 / exp20                   Contagion, recovery, persistence
  exp7 / exp21 / exp22           Systemic (multi-construct) events
  exp16 / exp23                  Threshold sensitivity analyses
  exp26 / exp27 / exp29          Paired-bootstrap significance tests
  exp28b_en_stp_bootstrap_correct.py  Primary ENLR+STP detection/forecast AUPRC
  exp_ablation_detection.py      Feature-set ablation
  figures/                       Paper figure generation

data/          Study data (NOT distributed — see data/README.md)
```

## Setup

Uses [uv](https://docs.astral.sh/uv/) with Python 3.11:

```bash
uv sync
```

LLM-based feature extraction needs API keys in the environment (or a `.env`
file): `OPENAI_API_KEY` for `features/text.py` / `semantic_judge.py`, and
`GEMINI_API_KEY` for `features/annotation_validation.py`. The statistical
analyses and crash/surge experiments need no API access.

## Running

The study data itself is not public (see below), so the pipeline is runnable
only with access to the data files described in `data/README.md`. With those
in place:

```bash
# 1. Feature extraction (raw recordings → feature CSVs)
uv run python features/text.py
uv run python features/audio.py
uv run python features/extract_face_video_features.py

# 2. Construct-level analyses
uv run python analysis/cfa_analysis.py
uv run python analysis/riclpm_analysis.py
uv run python analysis/robustness_analyses.py

# 3. Crash/surge experiments (write JSON/CSV to crash_surge/results/)
uv run python crash_surge/exp1_event_counts.py
uv run python crash_surge/exp28b_en_stp_bootstrap_correct.py
uv run python crash_surge/exp_ablation_detection.py

# 4. Figures
uv run python crash_surge/figures/fig5_detection_auprc.py
```

## Data availability

The underlying study data (recordings, transcripts, per-participant ratings)
comes from a human-subjects study approved by the Ethics Review Committee on
Research with Human Subjects of Waseda University and cannot be shared
publicly. De-identified derived data may be available from the corresponding
author on reasonable request: Ryuichi Sumida, sumida.ryuichi.65m@st.kyoto-u.ac.jp.

## Citation

```bibtex
@inproceedings{sumida2026memory,
  author    = {Sumida, Ryuichi and Saeki, Mao and Eguchi, Masaki and
               Yoshikawa, Sadahiro and Inoue, Koji and Kawahara, Tatsuya and
               Matsuyama, Yoichi},
  title     = {Memory-Driven Self-Disclosure and Relational Turning Points:
               A Longitudinal Multimodal Study of Human-AI Interaction},
  booktitle = {International Conference on Multimodal Interaction (ICMI '26)},
  year      = {2026},
  publisher = {ACM},
  address   = {Napoli, Italy},
  doi       = {10.1145/3776574.3831135}
}
```

## License

Code is released under the [MIT License](LICENSE). The paper is published
under CC-BY 4.0.
