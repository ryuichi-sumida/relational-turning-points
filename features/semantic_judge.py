#!/usr/bin/env python3
"""
Tier 1: LLM-as-Judge Semantic Features.

For each session, feed ALL prior session transcripts + current session to the LLM.
Evaluate on 6 dimensions (1-7 scale matching user self-report):
  - sem_memory_accuracy: Did bot correctly recall facts from prior sessions?
  - sem_emotional_attunement: Did bot respond appropriately to user's emotional state?
  - sem_response_relevance: Were responses on-topic and substantive?
  - sem_topic_novelty: How much new ground covered vs. rehashing?
  - sem_conversational_depth: Did follow-ups go deeper or stay surface?
  - sem_felt_understanding: Would a reader feel the user was "heard"?

Usage:
    uv run python features/semantic_judge.py --user_id 20

    uv run python features/semantic_judge.py
"""

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import shared utilities from text.py (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from text import (
    parse_tsv,
    call_llm_json,
    print_cost_summary,
    BASE_DIR,
    DATASET_DIR,
)

# ── Output paths ─────────────────────────────────────────────────────────────
SEMANTIC_OUTPUT_DIR = BASE_DIR / "data"

# ── Dimensions ────────────────────────────────────────────────────────────────
SEMANTIC_DIMENSIONS = [
    "sem_memory_accuracy",
    "sem_emotional_attunement",
    "sem_response_relevance",
    "sem_topic_novelty",
    "sem_conversational_depth",
    "sem_felt_understanding",
]

# ── Judge prompt ──────────────────────────────────────────────────────────────
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of chatbot conversation quality. You will receive:
1. Condensed transcripts of PRIOR sessions between the same user and the chatbot "Intella"
2. The CURRENT session transcript in full

Rate the CURRENT session on these 6 dimensions using a 1-7 scale (matching the user's self-report scale):

1. **memory_accuracy** (1-7): How accurately does the bot recall and reference facts, preferences, and experiences from prior sessions? Consider: Are memory references correct? Does the bot confuse or fabricate details?
   - 1 = Frequent incorrect or fabricated memories
   - 4 = Some correct, some incorrect or no memory references
   - 7 = All memory references are accurate and specific

2. **emotional_attunement** (1-7): How well does the bot respond to the user's emotional state? Consider: Does it acknowledge emotions? Is the tone appropriate? Does it validate feelings?
   - 1 = Completely misreads or ignores emotions
   - 4 = Sometimes acknowledges emotions but inconsistently
   - 7 = Consistently sensitive and appropriate emotional responses

3. **response_relevance** (1-7): How on-topic and substantive are the bot's responses? Consider: Does it address what the user actually said? Are responses meaningful or generic?
   - 1 = Mostly off-topic or generic filler
   - 4 = Generally on-topic but sometimes superficial
   - 7 = Consistently relevant, substantive, and thoughtful

4. **topic_novelty** (1-7): How much new conversational ground is covered? Consider: Does the session explore new topics or rehash old ones? Is there variety?
   - 1 = Entirely rehashes prior conversations
   - 4 = Mix of new and repeated topics
   - 7 = Explores entirely new topics and angles

5. **conversational_depth** (1-7): Do follow-up questions and responses go deeper? Consider: Does the bot ask meaningful follow-ups? Does the conversation build on itself?
   - 1 = Stays entirely at surface level
   - 4 = Some depth but inconsistent
   - 7 = Consistently builds depth with thoughtful follow-ups

6. **felt_understanding** (1-7): Would a reader feel the user was truly "heard" and understood? Consider: Does the bot demonstrate genuine comprehension of the user's perspective?
   - 1 = User clearly not understood at all
   - 4 = Partially understood
   - 7 = User would feel deeply heard and understood

Return a JSON object with these exact keys:
{
    "memory_accuracy": <int 1-7>,
    "emotional_attunement": <int 1-7>,
    "response_relevance": <int 1-7>,
    "topic_novelty": <int 1-7>,
    "conversational_depth": <int 1-7>,
    "felt_understanding": <int 1-7>
}

Return ONLY the JSON object."""


# ── Core functions ────────────────────────────────────────────────────────────

def load_all_sessions_for_user(user_dir: Path) -> list[dict]:
    """Load all sessions for a user in chronological order.

    Returns:
        List of dicts with keys: session_id, session_num, turns, path
    """
    tsv_files = sorted(glob.glob(str(user_dir / "session_*_transcript.tsv")))
    sessions = []
    for tsv_path in tsv_files:
        session_id = Path(tsv_path).stem.replace("_transcript", "")
        match = re.search(r"(\d+)", session_id)
        session_num = int(match.group(1)) if match else 0
        turns = parse_tsv(Path(tsv_path))
        if turns:
            sessions.append({
                "session_id": session_id,
                "session_num": session_num,
                "turns": turns,
                "path": tsv_path,
            })
    return sessions


def format_prior_sessions(sessions: list[dict], current_idx: int) -> str:
    """Format prior sessions as condensed transcripts for context.

    Args:
        sessions: All sessions for this user
        current_idx: Index of the current session (0-based)

    Returns:
        String with condensed prior session transcripts
    """
    if current_idx == 0:
        return "(No prior sessions)"

    parts = []
    for i in range(current_idx):
        sess = sessions[i]
        parts.append(f"--- Prior Session {sess['session_num']} ---")
        for t in sess["turns"]:
            parts.append(f"[{t['speaker']}]: {t['text']}")
        parts.append("")

    return "\n".join(parts)


def format_current_session(session: dict) -> str:
    """Format current session transcript."""
    lines = [f"--- Current Session {session['session_num']} ---"]
    for t in session["turns"]:
        lines.append(f"[{t['speaker']}]: {t['text']}")
    return "\n".join(lines)


def evaluate_session(
    prior_context: str,
    current_transcript: str,
    session_num: int,
    user_id: str,
    session_id: str,
) -> dict:
    """Evaluate a single session using the LLM judge.

    Args:
        prior_context: Formatted prior sessions
        current_transcript: Formatted current session
        session_num: 1-based session number
        user_id: User identifier
        session_id: Session identifier

    Returns:
        Dict with 6 dimension scores
    """
    user_prompt = f"{prior_context}\n\n{current_transcript}"

    result = call_llm_json(
        JUDGE_SYSTEM_PROMPT,
        user_prompt,
        cache_prefix=f"semjudge_{user_id}_{session_id}",
    )

    # Extract scores with defaults
    scores = {}
    dim_keys = [
        "memory_accuracy", "emotional_attunement", "response_relevance",
        "topic_novelty", "conversational_depth", "felt_understanding",
    ]
    for key in dim_keys:
        val = result.get(key, 4)
        # Clamp to 1-7 range
        try:
            val = max(1, min(7, int(val)))
        except (TypeError, ValueError):
            val = 4
        scores[f"sem_{key}"] = val

    # Session 1 overrides: no prior sessions to judge memory against
    if session_num == 1:
        scores["sem_memory_accuracy"] = 4  # neutral (no prior sessions)
        scores["sem_topic_novelty"] = 7    # all content is novel

    return scores


def process_user_semantic(user_dir: Path) -> list[dict]:
    """Process all sessions for one user with semantic judge.

    Returns:
        List of metric dicts (one per session)
    """
    user_id = user_dir.name
    print(f"[Semantic Judge] Processing user: {user_id}")

    sessions = load_all_sessions_for_user(user_dir)
    if not sessions:
        print(f"  No sessions found for {user_id}")
        return []

    all_metrics = []
    for idx, session in enumerate(sessions):
        print(f"  Session {session['session_num']} ({idx + 1}/{len(sessions)})")

        prior_context = format_prior_sessions(sessions, idx)
        current_transcript = format_current_session(session)

        scores = evaluate_session(
            prior_context,
            current_transcript,
            session["session_num"],
            user_id,
            session["session_id"],
        )

        row = {"session_id": session["session_id"]}
        row.update(scores)
        all_metrics.append(row)

    # Write per-user CSV
    if all_metrics:
        output_path = user_dir / "semantic_judge_metrics.csv"
        fieldnames = list(all_metrics[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_metrics:
                writer.writerow(row)
        print(f"  Wrote {output_path} ({len(all_metrics)} rows)")

    return all_metrics


def aggregate_semantic_features():
    """Aggregate all per-user semantic_judge_metrics.csv into one file."""
    all_dfs = []
    for d in sorted(DATASET_DIR.iterdir()):
        if d.is_dir() and re.match(r"^\d+$", d.name):
            csv_path = d / "semantic_judge_metrics.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df.insert(0, "participant_id", int(d.name))
                all_dfs.append(df)

    if not all_dfs:
        print("No per-user semantic CSVs found for aggregation.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = SEMANTIC_OUTPUT_DIR / "all_semantic_features.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n[Semantic Judge] Aggregated {len(all_dfs)} users → {out_path} "
          f"({combined.shape[0]} rows × {combined.shape[1]} cols)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tier 1: LLM-as-Judge Semantic Features"
    )
    parser.add_argument(
        "--user_id", type=str, default=None,
        help="Process only this user ID. If not set, process all.",
    )
    parser.add_argument(
        "--aggregate_only", action="store_true",
        help="Only aggregate existing CSVs without reprocessing.",
    )
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate_semantic_features()
        return

    if args.user_id:
        user_dir = DATASET_DIR / args.user_id
        if not user_dir.is_dir():
            print(f"User directory not found: {user_dir}")
            return
        process_user_semantic(user_dir)
    else:
        for d in sorted(DATASET_DIR.iterdir()):
            if d.is_dir() and re.match(r"^\d+$", d.name):
                process_user_semantic(d)

    aggregate_semantic_features()
    print_cost_summary()


if __name__ == "__main__":
    main()
