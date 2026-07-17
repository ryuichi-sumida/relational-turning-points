"""
Shared configuration for breakdown detection experiments.
"""
from pathlib import Path

# Paths
BASE = Path(__file__).resolve().parents[1]
DATASET = BASE / "data"
RESULTS_DIR = BASE / "crash_surge" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Constructs and their Q-item mappings
CONSTRUCTS = ["familiarity", "social_penetration", "memory", "conversational", "enjoyment"]

CONSTRUCT_Q_MAP = {
    "familiarity": ["Q1"],
    "social_penetration": ["Q3", "Q4"],
    "memory": ["Q5", "Q6"],
    "conversational": ["Q7", "Q8"],
    "enjoyment": ["Q9", "Q10"],
}

# Feature columns to EXCLUDE from model input (meta + targets + AR features)
META_COLS = {"pid", "session_num", "session"}
TARGET_COLS = set(CONSTRUCTS)
# Hard constraint: NO autoregressive features (past ratings, lagged ratings, etc.)
AR_COLS = set()  # None should exist in behavioral features, but guard against it

# Modality-to-feature prefix mappings (for ablation)
# These are approximate groupings based on feature naming conventions
TEXT_PREFIXES = [
    "user_emotion_expression", "emotional_support", "user_self_disclosure",
    "max_user_vulnerability", "bot_self_disclosure", "mean_bot_self_disclosure",
    "deep_followup", "bot_question_dominance", "user_question_rate",
    "user_topic_initiation", "memory_reference", "memory_accuracy",
    "frustration_with_bot_memory", "cross_session_question_repetition",
    "cross_session_content_repetition", "bot_content_novelty",
    "user_initiated_memory_test", "actionability_marker", "hedging_rate",
    "response_specificity", "personalized_opening", "positive_closing",
    "delta_user_vulnerability", "max_prior_vulnerability",
    "question_about_bot", "praises_the_bot", "user_name_used_by_bot",
    "user_words_per_minute", "keyword_memory_ref",
    "user_bot_semantic_alignment", "alignment_variance",
    "user_topic_novelty", "session_topic_novelty", "topic_diversity",
    "fact_based_memory", "cumulative_fact_count",
    "sem_memory_accuracy", "sem_emotional_attunement", "sem_response_relevance",
    "sem_topic_novelty", "sem_conversational_depth", "sem_felt_understanding",
]

AUDIO_PREFIXES = [
    "user_to_bot_speech_ratio", "user_backchannel", "gap_mean_s",
    "overlap_rate", "F0_", "loudness_", "jitter_", "shimmer_", "hnr_",
    "mfcc", "arousal_", "valence_", "delta_user_speech_time",
    "delta_gap_mean_s",
]

VIDEO_PREFIXES = [
    "focused_pct", "switches_per_min", "head_motion_energy",
    "blink_rate",
]

# Session-level only features (no delta/EMA/max_prior/trend prefix)
TEMPORAL_PREFIXES = ["delta_", "max_prior_", "EMA_", "ever_", "cumulative_",
                     "n_prior_sessions", "trend_"]
