"""Figure 6: Stable Feature Signatures for Crash and Surge Detection.

Updated to use ablation results: for each construct, loads the best-performing
feature-set condition and extracts top features from EN and XGB models.
A feature is "stable" for a construct if it appears in the top 15 of the
best model and was selected in >= 18 of 24 LOPO folds.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
ABLATION_DIR = Path(__file__).resolve().parents[1] / "results" / "ablation"
SUMMARY_CSV = ABLATION_DIR / "summary.csv"
CONSTRUCTS = ["familiarity", "social_penetration", "memory", "conversational", "enjoyment"]
CONSTRUCTS_WITH_SYS = CONSTRUCTS + ["systemic"]

STABILITY_THRESHOLD = 18  # >= 75% of 24 folds

# ── Find best feature set per construct × event ──────────────────────────
def get_best_conditions(event_type, constructs):
    """Return dict: construct -> (feature_set, json_path) for the best ablation condition."""
    df = pd.read_csv(SUMMARY_CSV)
    sub = df[df["event"] == event_type]
    best = {}
    for con in constructs:
        s = sub[sub["construct"] == con]
        if s.empty:
            continue
        row = s.loc[s["best_AUPRC"].idxmax()]
        fs = row["feature_set"]
        json_path = ABLATION_DIR / f"{fs}_{event_type}_{con}.json"
        best[con] = (fs, json_path)
    return best

# ── Load top features from ablation JSONs ────────────────────────────────
def load_ablation_stability(event_type, constructs):
    """Load top features from the best ablation condition per construct.

    For each construct, load both EN and XGB top_features from the best condition.
    A feature is counted as "stable" for a construct if n_folds >= threshold
    in either model's top 15.

    Returns: dict of feature -> list of (construct, max_n_folds)
    """
    best_conditions = get_best_conditions(event_type, constructs)
    all_counts = {}

    for con, (fs, json_path) in best_conditions.items():
        with open(json_path) as f:
            d = json.load(f)

        # Collect best n_folds per feature across both models
        feat_folds = {}
        for model_key in ["elastic_net", "xgboost"]:
            for entry in d.get(model_key, {}).get("top_features", []):
                feat = entry["feature"]
                nf = entry["n_folds"]
                feat_folds[feat] = max(feat_folds.get(feat, 0), nf)

        for feat, nf in feat_folds.items():
            if feat not in all_counts:
                all_counts[feat] = []
            all_counts[feat].append((con, nf))

    return all_counts

crash_counts = load_ablation_stability("crash", CONSTRUCTS_WITH_SYS)
surge_counts = load_ablation_stability("surge", CONSTRUCTS)

# ── Compute aggregate stability score ─────────────────────────────────────
def aggregate_stability(all_counts):
    """Return dict: feature -> (n_constructs_stable, max_fold_count, mean_fold_count)."""
    result = {}
    for feat, entries in all_counts.items():
        n_stable = sum(1 for _, c in entries if c >= STABILITY_THRESHOLD)
        max_c = max(c for _, c in entries)
        mean_c = np.mean([c for _, c in entries])
        result[feat] = (n_stable, max_c, mean_c)
    return result

crash_agg = aggregate_stability(crash_counts)
surge_agg = aggregate_stability(surge_counts)

# ── Classify feature type by modality ─────────────────────────────────────
PERSON_NORM_PREFIXES = ("dvol_", "pdev_", "pz_")
TEMPORAL_PREFIXES = ("delta_", "max_prior_", "EMA_", "trend_", "ever_", "cumulative_")

AUDIO_FEATURES = {
    "F0_mean", "F0_std", "F0_slope", "jitter_mean", "shimmer_mean",
    "hnr_mean", "loudness_mean", "loudness_std", "loudness_late_minus_early",
    "mfcc1_mean", "speech_rate", "overlap_rate_after_system",
    "gap_mean_s", "user_words_per_minute", "switches_per_min",
}

VIDEO_FEATURES = {
    "head_motion_energy_deg_per_s", "arousal_max", "valence_max",
    "focused_pct", "au_brow_raise_mean", "au_lip_corner_pull_mean",
    "gaze_pitch_std", "gaze_yaw_std",
}

def classify_feature(feat):
    """Return (base_name, modality) for a feature."""
    for pfx in PERSON_NORM_PREFIXES:
        if feat.startswith(pfx):
            base = feat[len(pfx):]
            return base, "person_norm"
    for pfx in TEMPORAL_PREFIXES:
        if feat.startswith(pfx):
            base = feat[len(pfx):]
            return base, "temporal"
    if feat in AUDIO_FEATURES:
        return feat, "audio"
    if feat in VIDEO_FEATURES:
        return feat, "video"
    return feat, "text"

# ── Select top features ──────────────────────────────────────────────────
def get_top_features(agg, n=10):
    """Sort by (n_constructs_stable DESC, max_fold DESC, mean_fold DESC) and return top n."""
    ranked = sorted(agg.items(), key=lambda x: (-x[1][0], -x[1][1], -x[1][2]))
    ranked = [(f, s) for f, s in ranked if s[0] >= 1]
    return ranked[:n]

crash_top = get_top_features(crash_agg, n=10)
surge_top = get_top_features(surge_agg, n=10)

# Print for debugging
print("CRASH top features:")
for f, (ns, mf, meanf) in crash_top:
    _, mod = classify_feature(f)
    print(f"  {f:45s} stable_in={ns} max_folds={mf} modality={mod}")

print("\nSURGE top features:")
for f, (ns, mf, meanf) in surge_top:
    _, mod = classify_feature(f)
    print(f"  {f:45s} stable_in={ns} max_folds={mf} modality={mod}")

# ── Clean feature names for display ───────────────────────────────────────
NICE_NAMES = {
    "question_about_bot": "Question about bot",
    "bot_question_dominance": "Bot question dominance",
    "focused_pct": "Focused gaze %",
    "hedging_rate": "Hedging rate",
    "user_name_used_by_bot": "User name used by bot",
    "bot_self_disclosure_rate": "Bot self-disclosure rate",
    "sem_conversational_depth": "Conversational depth (sem)",
    "sem_response_relevance": "Response relevance (sem)",
    "sem_emotional_attunement": "Emotional attunement (sem)",
    "user_initiated_memory_test": "User-initiated memory test",
    "frustration_with_bot_memory": "Frustration w/ bot memory",
    "praises_the_bot": "Praises the bot",
    "user_self_disclosure_rate": "User self-disclosure rate",
    "user_backchannel_ratio": "User backchannel ratio",
    "user_question_rate": "User question rate",
    "user_emotion_expression_rate": "User emotion expression rate",
    "deep_followup_rate": "Deep follow-up rate",
    "emotional_support_validation_rate": "Emotional support/validation",
    "cross_session_question_repetition_rate": "Cross-session Q repetition",
    "cross_session_content_repetition_rate": "Cross-session content repeat",
    "session_topic_novelty_score": "Session topic novelty",
    "bot_content_novelty_score": "Bot content novelty",
    "user_topic_novelty_score": "User topic novelty",
    "keyword_memory_ref_rate": "Keyword memory ref. rate",
    "memory_reference_rate": "Memory reference rate",
    "memory_accuracy_rate": "Memory accuracy rate",
    "sem_memory_accuracy": "Memory accuracy (sem)",
    "actionability_marker_rate": "Actionability marker rate",
    "alignment_variance": "Alignment variance",
    "user_bot_semantic_alignment": "User-bot semantic alignment",
    "max_user_vulnerability_level": "Max vulnerability level",
    "personalized_opening_presence": "Personalized opening",
    "overlap_rate_after_system": "Overlap rate (after system)",
    "head_motion_energy_deg_per_s": "Head motion energy",
    "arousal_max": "Arousal (max)",
    "valence_max": "Valence (max)",
    "F0_slope": "F0 slope",
    "F0_mean": "F0 mean",
    "F0_std": "F0 std",
    "jitter_mean": "Jitter",
    "shimmer_mean": "Shimmer",
    "hnr_mean": "HNR",
    "loudness_mean": "Loudness mean",
    "loudness_std": "Loudness std",
    "loudness_late_minus_early": "Loudness late-early diff",
    "mfcc1_mean": "MFCC1",
    "user_words_per_minute": "User words/min",
    "switches_per_min": "Turn switches/min",
    "sem_felt_understanding": "Felt understanding (sem)",
    "n_prior_sessions": "N prior sessions",
    "response_specificity_score": "Response specificity",
    "speech_rate": "Speech rate",
    "gap_mean_s": "Response gap (mean)",
}

def clean_name(feat):
    """Remove prefix, then apply nice name mapping."""
    base, modality = classify_feature(feat)
    prefix_label = ""
    if modality == "person_norm":
        for pfx in PERSON_NORM_PREFIXES:
            if feat.startswith(pfx):
                prefix_label = pfx.rstrip("_").upper()
                break
    elif modality == "temporal":
        for pfx in TEMPORAL_PREFIXES:
            if feat.startswith(pfx):
                prefix_label = pfx.rstrip("_")
                break
    nice = NICE_NAMES.get(base, base.replace("_", " ").title())
    if prefix_label:
        nice = f"{nice} ({prefix_label})"
    return nice

# ── Colors ────────────────────────────────────────────────────────────────
MODALITY_COLORS = {
    "text": "#4878CF",        # blue
    "audio": "#EE854A",       # orange
    "video": "#6ACC64",       # green
    "person_norm": "#9B59B6", # purple
    "temporal": "#E74C3C",    # red
}

MODALITY_LABELS = {
    "text": "Session-level text",
    "audio": "Session-level audio",
    "video": "Session-level video",
    "person_norm": "Person-norm. (dvol/pdev/pz)",
    "temporal": "Temporal (delta/EMA/trend)",
}

# ── Plot ──────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 5.0), sharey=False)

def plot_panel(ax, top_features, title):
    """Plot horizontal bar chart for one panel."""
    top_features = list(reversed(top_features))

    names = []
    scores = []
    colors = []
    for feat, (n_stable, max_fold, mean_fold) in top_features:
        names.append(clean_name(feat))
        scores.append(n_stable)
        _, modality = classify_feature(feat)
        colors.append(MODALITY_COLORS[modality])

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, scores, color=colors, edgecolor="white", linewidth=0.5, height=0.7)

    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2,
                str(score), va="center", ha="left", fontsize=7, color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    n_total = 6 if "Crash" in title else 5
    ax.set_xlabel(f"No. constructs where stable (of {n_total})")
    ax.set_title(title, fontweight="bold", pad=6)
    ax.set_xlim(0, max(scores) + 0.8 if scores else 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plot_panel(ax1, crash_top, "Crash Detection")
plot_panel(ax2, surge_top, "Surge Detection")

# Shared legend — only include modalities that appear
used_modalities = set()
for feat, _ in crash_top + surge_top:
    _, mod = classify_feature(feat)
    used_modalities.add(mod)

handles = [mpl.patches.Patch(facecolor=MODALITY_COLORS[m], edgecolor="white",
           label=MODALITY_LABELS[m]) for m in ["text", "audio", "video", "person_norm", "temporal"]
           if m in used_modalities]
fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), frameon=False,
           bbox_to_anchor=(0.65, -0.01))

plt.subplots_adjust(left=0.35, right=0.95, bottom=0.10, top=0.95, hspace=0.45)

out = Path(__file__).resolve().parents[2] / "figures" / "fig_stable_features.pdf"
out.parent.mkdir(exist_ok=True)
fig.savefig(out)
print(f"\nSaved to {out}")

fig.savefig(out.with_suffix(".png"))
print(f"Saved PNG preview to {out.with_suffix('.png')}")
