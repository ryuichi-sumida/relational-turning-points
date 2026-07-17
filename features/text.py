#!/usr/bin/env python3
"""Extract text-based metrics from conversation transcripts using OpenAI API."""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

import numpy as np
from openai import OpenAI

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]  # repo root
DATASET_DIR = BASE_DIR / "data" / "processed_videos"
NAMES_FILE = BASE_DIR / "data" / "names.txt"
CACHE_DIR = BASE_DIR / "data" / "text_cache"

# ── OpenAI client ──────────────────────────────────────────────────────────────
load_dotenv()  # load .env if present
client = OpenAI()  # uses OPENAI_API_KEY env var
LLM_MODEL = "gpt-4.1-2025-04-14"
EMBED_MODEL = "text-embedding-3-small"

# ── Pricing (USD per 1M tokens, as of 2025) ─────────────────────────────────
PRICING = {
    "gpt-4.1-2025-04-14": {"input": 2, "output": 8},
    "text-embedding-3-small": {"input": 0.020},
}

# ── Token usage tracker ──────────────────────────────────────────────────────
_token_usage = {
    "llm_input_tokens": 0,
    "llm_output_tokens": 0,
    "llm_cached_input_tokens": 0,
    "embed_input_tokens": 0,
    "llm_calls": 0,
    "llm_cache_hits": 0,
    "embed_calls": 0,
}


def print_cost_summary():
    """Print total token usage and estimated cost."""
    llm_price = PRICING.get(LLM_MODEL, {"input": 0, "output": 0})
    embed_price = PRICING.get(EMBED_MODEL, {"input": 0})

    llm_input_cost = _token_usage["llm_input_tokens"] / 1_000_000 * llm_price["input"]
    llm_output_cost = _token_usage["llm_output_tokens"] / 1_000_000 * llm_price.get("output", 0)
    embed_cost = _token_usage["embed_input_tokens"] / 1_000_000 * embed_price["input"]
    total = llm_input_cost + llm_output_cost + embed_cost

    print("\n" + "=" * 60)
    print("COST SUMMARY")
    print("=" * 60)
    print(f"LLM ({LLM_MODEL}):")
    print(f"  Calls: {_token_usage['llm_calls']} (cache hits: {_token_usage['llm_cache_hits']})")
    print(f"  Input tokens:  {_token_usage['llm_input_tokens']:>10,}  ${llm_input_cost:.4f}")
    print(f"  Output tokens: {_token_usage['llm_output_tokens']:>10,}  ${llm_output_cost:.4f}")
    print(f"Embeddings ({EMBED_MODEL}):")
    print(f"  Calls: {_token_usage['embed_calls']}")
    print(f"  Input tokens:  {_token_usage['embed_input_tokens']:>10,}  ${embed_cost:.4f}")
    print("-" * 60)
    print(f"TOTAL ESTIMATED COST: ${total:.4f}")
    print("=" * 60)

# ── Name mapping ──────────────────────────────────────────────────────────────
def load_names() -> dict[str, list[str]]:
    """Load user_id -> list of name variants from names.txt."""
    mapping: dict[str, list[str]] = {}
    if not NAMES_FILE.exists():
        return mapping
    text = NAMES_FILE.read_text(encoding="utf-8")
    for m in re.finditer(r'\((\d+),\s*"([^"]+)"\)', text):
        uid, name = m.group(1), m.group(2)
        mapping.setdefault(uid, []).append(name)
    return mapping


USER_NAMES = load_names()


# ── Cache helpers ─────────────────────────────────────────────────────────────
def _cache_key(prefix: str, *parts: str) -> Path:
    """Deterministic cache path from prefix + content parts."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]
    return CACHE_DIR / f"{prefix}_{h}.json"


def _cache_get(key: Path) -> dict | None:
    if key.exists():
        try:
            return json.loads(key.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def _cache_set(key: Path, data: dict):
    key.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ts_to_seconds(ts: str) -> float:
    """Convert HH:MM:SS.mmm to seconds."""
    parts = ts.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def parse_tsv(path: Path) -> list[dict]:
    """Parse a transcript TSV into list of turn dicts with timestamps."""
    turns = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if parts[0] == "start_time":
                continue  # header
            if len(parts) < 5:
                continue
            turns.append({
                "speaker": parts[3],
                "text": parts[4],
                "start_sec": _ts_to_seconds(parts[0]),
                "end_sec": _ts_to_seconds(parts[1]),
            })
    return turns


def call_llm_json(system_prompt: str, user_prompt: str, max_retries: int = 3,
                   cache_prefix: str = "llm") -> dict:
    """Call LLM with JSON output, returning parsed dict. Results are cached."""
    key = _cache_key(cache_prefix, LLM_MODEL, system_prompt, user_prompt)
    cached = _cache_get(key)
    if cached is not None:
        _token_usage["llm_cache_hits"] += 1
        return cached

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=16384,
            )
            if resp.usage:
                _token_usage["llm_input_tokens"] += resp.usage.prompt_tokens
                _token_usage["llm_output_tokens"] += resp.usage.completion_tokens
            _token_usage["llm_calls"] += 1
            result = json.loads(resp.choices[0].message.content)
            _cache_set(key, result)
            return result
        except Exception as e:
            print(f"  LLM call attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return {}


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings for a list of texts using text-embedding-3-small."""
    if not texts:
        return []
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    if resp.usage:
        _token_usage["embed_input_tokens"] += resp.usage.prompt_tokens
    _token_usage["embed_calls"] += 1
    return [d.embedding for d in resp.data]


def cosine_sim(a: list[float], b: list[float]) -> float:
    a_np, b_np = np.array(a), np.array(b)
    denom = np.linalg.norm(a_np) * np.linalg.norm(b_np)
    if denom == 0:
        return 0.0
    return float(np.dot(a_np, b_np) / denom)


def safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else float("nan")


# ── A1. Keyword-based memory reference detection ────────────────────────────
_MEMORY_REF_PATTERNS = [
    re.compile(r"\bremember\b", re.IGNORECASE),
    re.compile(r"\blast time\b", re.IGNORECASE),
    re.compile(r"\byou mentioned\b", re.IGNORECASE),
    re.compile(r"\byou said\b", re.IGNORECASE),
    re.compile(r"\bwe talked about\b", re.IGNORECASE),
    re.compile(r"\bwe discussed\b", re.IGNORECASE),
    re.compile(r"\bas I (said|told|mentioned)\b", re.IGNORECASE),
    re.compile(r"\bI told you\b", re.IGNORECASE),
    re.compile(r"\bI mentioned\b", re.IGNORECASE),
    re.compile(r"\byou asked me\b", re.IGNORECASE),
    re.compile(r"\bpreviously\b", re.IGNORECASE),
    re.compile(r"\bbefore\b.{0,15}\b(session|conversation)\b", re.IGNORECASE),
    re.compile(r"\blast (session|conversation|week)\b", re.IGNORECASE),
    re.compile(r"\bdo you recall\b", re.IGNORECASE),
]


def compute_keyword_memory_refs(turns: list[dict]) -> tuple[int, float]:
    """Count keyword-based memory references across all turns.

    Returns:
        (count, rate_per_turn)
    """
    count = 0
    for t in turns:
        text = t["text"]
        for pat in _MEMORY_REF_PATTERNS:
            if pat.search(text):
                count += 1
                break  # one match per turn is enough
    rate = safe_div(count, len(turns))
    return count, rate


# ── A2. Within-session semantic alignment ────────────────────────────────────

def compute_user_bot_alignment(turns: list[dict]) -> tuple[float, float]:
    """Cosine similarity between user utterances and subsequent bot responses.

    Returns:
        (mean_alignment, alignment_variance)
    """
    pairs = []
    for i, t in enumerate(turns):
        if t["speaker"] == "user":
            # Find next bot turn
            for j in range(i + 1, len(turns)):
                if turns[j]["speaker"] == "system":
                    pairs.append((t["text"], turns[j]["text"]))
                    break

    if not pairs:
        return float("nan"), float("nan")

    user_texts = [p[0] for p in pairs]
    bot_texts = [p[1] for p in pairs]
    user_embs = get_embeddings(user_texts)
    bot_embs = get_embeddings(bot_texts)

    sims = [cosine_sim(u, b) for u, b in zip(user_embs, bot_embs)]
    mean_align = float(np.mean(sims))
    var_align = float(np.var(sims)) if len(sims) > 1 else 0.0
    return mean_align, var_align


# ── A3. User-side topic novelty ──────────────────────────────────────────────

def compute_user_topic_novelty(
    turns: list[dict],
    prior_all_embeddings: list[list[float]],
) -> tuple[float, float, float, list[list[float]]]:
    """Embed user utterances and compare to all prior session embeddings.

    Args:
        turns: Current session turns
        prior_all_embeddings: Accumulated embeddings from all prior sessions

    Returns:
        (user_topic_novelty_score, session_topic_novelty_score,
         topic_diversity_within_session, current_embeddings)
    """
    user_texts = [t["text"] for t in turns if t["speaker"] == "user" and len(t["text"].split()) >= 3]

    if not user_texts:
        return float("nan"), float("nan"), float("nan"), []

    current_embs = get_embeddings(user_texts)

    # User topic novelty: how novel are user utterances vs. all prior sessions
    if prior_all_embeddings:
        max_sims = []
        for cemb in current_embs:
            sims = [cosine_sim(cemb, pemb) for pemb in prior_all_embeddings]
            max_sims.append(max(sims))
        user_topic_novelty = float(np.mean([1.0 - s for s in max_sims]))
    else:
        user_topic_novelty = 1.0  # first session: all novel

    # Session topic novelty: combine user + bot texts
    all_session_texts = [t["text"] for t in turns if len(t["text"].split()) >= 3]
    if all_session_texts and prior_all_embeddings:
        all_session_embs = get_embeddings(all_session_texts)
        max_sims = []
        for cemb in all_session_embs:
            sims = [cosine_sim(cemb, pemb) for pemb in prior_all_embeddings]
            max_sims.append(max(sims))
        session_topic_novelty = float(np.mean([1.0 - s for s in max_sims]))
    elif not prior_all_embeddings:
        session_topic_novelty = 1.0
    else:
        session_topic_novelty = float("nan")

    # Within-session diversity: mean pairwise distance among user utterances
    if len(current_embs) >= 2:
        pairwise_sims = []
        for i in range(len(current_embs)):
            for j in range(i + 1, len(current_embs)):
                pairwise_sims.append(cosine_sim(current_embs[i], current_embs[j]))
        topic_diversity = 1.0 - float(np.mean(pairwise_sims))
    else:
        topic_diversity = 0.0

    return user_topic_novelty, session_topic_novelty, topic_diversity, current_embs


# ── A4. Fact extraction + retrieval-based memory accuracy ────────────────────

FACT_EXTRACTION_SYSTEM = """You are extracting factual claims from a conversation between a user and a chatbot named Intella. Extract concrete, verifiable facts about the user that were shared in this conversation.

Return a JSON object with key "facts" containing an array of strings. Each string should be a single factual claim.
Focus on:
- Personal details (name, age, occupation, hobbies, family)
- Preferences (likes, dislikes, favorites)
- Experiences (events, travel, education)
- Plans and goals
- Opinions on specific topics

Do NOT include:
- Vague statements or greetings
- Bot's own statements that aren't user facts
- Duplicate/redundant facts

Return ONLY the JSON object."""


def extract_facts_from_session(turns: list[dict], session_id: str, user_id: str) -> list[str]:
    """Extract factual claims from a session transcript via LLM."""
    user_turns_text = "\n".join(
        f"[{t['speaker']}]: {t['text']}" for t in turns
    )
    result = call_llm_json(
        FACT_EXTRACTION_SYSTEM,
        user_turns_text,
        cache_prefix=f"facts_{user_id}_{session_id}",
    )
    return result.get("facts", [])


def compute_fact_based_memory_accuracy(
    turns: list[dict],
    prior_facts: list[str],
    prior_fact_embeddings: list[list[float]],
) -> tuple[float, int, int]:
    """Check bot memory claims against accumulated fact database.

    Args:
        turns: Current session turns
        prior_facts: List of factual claims from prior sessions
        prior_fact_embeddings: Embeddings of prior facts

    Returns:
        (accuracy, n_verified, n_contradicted)
    """
    if not prior_facts or not prior_fact_embeddings:
        return float("nan"), 0, 0

    # Find bot turns that look like memory references
    bot_memory_texts = []
    for t in turns:
        if t["speaker"] == "system":
            text = t["text"].lower()
            memory_cues = ["remember", "last time", "you mentioned", "you said",
                          "you told", "earlier", "before", "previous"]
            if any(cue in text for cue in memory_cues):
                bot_memory_texts.append(t["text"])

    if not bot_memory_texts:
        return float("nan"), 0, 0

    bot_mem_embs = get_embeddings(bot_memory_texts)

    n_verified = 0
    n_contradicted = 0
    for bemb in bot_mem_embs:
        sims = [cosine_sim(bemb, femb) for femb in prior_fact_embeddings]
        max_sim = max(sims)
        if max_sim > 0.70:
            n_verified += 1
        elif max_sim < 0.40:
            n_contradicted += 1

    total = n_verified + n_contradicted
    accuracy = safe_div(n_verified, total) if total > 0 else float("nan")
    return accuracy, n_verified, n_contradicted


# ── LLM annotation prompts ────────────────────────────────────────────────────

TURN_ANNOTATION_SYSTEM = """You are an expert conversation analyst. You will receive a conversation transcript between a user and a system (chatbot named Intella). Annotate EVERY turn.

Return a JSON object with key "turns" containing an array. Each element corresponds to one turn (in order) and must have these fields:

For ALL turns:
- "is_question": bool — does this turn contain a question?

For USER turns only (set null for system turns):
- "emotion_expressed": bool — does the user express emotion?
- "emotion_valence": float from -1 (very negative) to 1 (very positive), or null if no emotion
- "user_vulnerability_level": int 0-3 (0=none, 1=mild personal info, 2=moderate personal struggles, 3=deep vulnerability)
- "is_topic_initiation": bool — does the user introduce a new topic?
- "user_question_about_bot": bool — does the user ask a question about the bot itself? (e.g., "What do you like?", "Do you have hobbies?")
- "user_praise_bot": bool — does the user praise or compliment the bot? (e.g., "You ask good questions", "It's nice talking to you")
- "user_memory_test": bool — does the user test the bot's memory? (e.g., "Do you remember…?", "Last time I said…", "As I mentioned…")
- "user_memory_frustration": bool — does the user express frustration about the bot's memory? (e.g., "We already discussed this", "I told you before")

For SYSTEM turns only (set null for user turns):
- "bot_emotion_acknowledgment": bool — does the bot acknowledge user's emotion from a prior turn?
- "bot_validation": bool — does the bot validate the user's feelings or experience?
- "bot_supportive": bool — is this a supportive/encouraging response?
- "bot_self_disclosure_level": int 0-3 (0=none, 1=mild opinion, 2=personal preference/experience, 3=deep self-disclosure)
- "bot_disclosure_reciprocity": bool — if the prior user turn had vulnerability >= 2, does the bot reciprocate with self-disclosure?
- "bot_open_ended_followup": bool — if the prior user turn had vulnerability >= 2, does the bot ask an open-ended follow-up that invites elaboration (vs. topic shift)?
- "bot_answers_user_question": bool — if a prior user turn asked a question, does this turn answer it?
- "has_example_marker": bool — does the response contain a concrete example?
- "has_actionability_marker": bool — does the response contain actionable advice?
- "has_hedging": bool — does the response hedge (e.g., "I think", "maybe", "perhaps")?
- "has_apology_repair": bool — does the bot apologize or attempt repair (e.g., "sorry", "I apologize", "let me correct")?
- "has_contradiction": bool — does the bot contradict something it said earlier in the conversation?
- "has_calibrated_uncertainty": bool — does the bot express appropriate uncertainty (e.g., "I'm not sure but...")?
- "is_content_providing": bool — does the bot provide substantive content (vs only asking a question)?
- "memory_reference": bool — does the bot reference something from a previous conversation/session?
- "memory_reference_correct": bool or null — if memory_reference is true, is it accurate?
- "memory_reference_specificity": int 0-3 or null — if memory_reference, how specific? (0=vague, 3=very specific)
- "personalized_opening": bool — (first system turn only) is the opening personalized to this user?

Return ONLY the JSON object with "turns" array. Each array element must have all applicable fields."""


# ── Core processing ────────────────────────────────────────────────────────────

def annotate_turns(turns: list[dict]) -> list[dict]:
    """Send transcript to LLM for turn-level annotation, chunking if needed."""
    CHUNK_SIZE = 30  # max turns per LLM call to avoid JSON truncation

    all_annotations = []
    for chunk_start in range(0, len(turns), CHUNK_SIZE):
        chunk = turns[chunk_start:chunk_start + CHUNK_SIZE]
        # Include a few prior turns as context (but only annotate the chunk)
        context_start = max(0, chunk_start - 3)
        context_turns = turns[context_start:chunk_start]

        transcript_lines = []
        for i, t in enumerate(context_turns):
            transcript_lines.append(f"[Context] Turn {context_start + i} [{t['speaker']}]: {t['text']}")
        for i, t in enumerate(chunk):
            transcript_lines.append(f"Turn {chunk_start + i} [{t['speaker']}]: {t['text']}")
        transcript_text = "\n".join(transcript_lines)

        if context_turns:
            transcript_text += f"\n\nNote: Only annotate turns {chunk_start} through {chunk_start + len(chunk) - 1}. The [Context] turns are for reference only."

        result = call_llm_json(TURN_ANNOTATION_SYSTEM, transcript_text)
        chunk_annotations = result.get("turns", [])

        # Pad or truncate to match chunk size
        while len(chunk_annotations) < len(chunk):
            chunk_annotations.append({})
        all_annotations.extend(chunk_annotations[:len(chunk)])

    return all_annotations[:len(turns)]



def compute_metrics(
    turns: list[dict],
    annotations: list[dict],
    session_id: str,
    user_id: str,
    prior_bot_question_embeddings: list[list[float]],
    prior_vuln_means: list[float],
    prior_max_vuln: list[int],
    prior_bot_content_embeddings: list[list[float]] | None = None,
) -> dict:
    """Compute all ~41 metrics from annotations."""
    user_turns = [(i, t, annotations[i]) for i, t in enumerate(turns) if t["speaker"] == "user"]
    sys_turns = [(i, t, annotations[i]) for i, t in enumerate(turns) if t["speaker"] == "system"]

    n_user = len(user_turns)
    n_sys = len(sys_turns)

    # ── A. Emotional & Empathic ──
    user_emotional = [(i, t, a) for i, t, a in user_turns if a.get("emotion_expressed")]
    n_emotional = len(user_emotional)

    user_emotion_expression_rate = safe_div(n_emotional, n_user)

    # Emotional support & validation: bot acknowledges, validates, or supports after emotional user turn
    support_count = 0
    for i, t, a in user_emotional:
        for j, t2, a2 in sys_turns:
            if j > i:
                if a2.get("bot_emotion_acknowledgment") or a2.get("bot_validation") or a2.get("bot_supportive"):
                    support_count += 1
                break
    emotional_support_validation_rate = safe_div(support_count, n_emotional)

    # ── B. Social Penetration ──
    vuln_levels = [a.get("user_vulnerability_level", 0) or 0 for _, _, a in user_turns]
    user_self_disclosure_rate = safe_div(sum(1 for v in vuln_levels if v >= 1), n_user)
    max_user_vulnerability_level = max(vuln_levels) if vuln_levels else 0
    bot_disc_levels = [a.get("bot_self_disclosure_level", 0) or 0 for _, _, a in sys_turns]
    bot_self_disclosure_rate = safe_div(sum(1 for v in bot_disc_levels if v >= 1), n_sys)
    mean_bot_self_disclosure_level = float(np.mean(bot_disc_levels)) if bot_disc_levels else float("nan")

    high_vuln_turns = [(i, t, a) for i, t, a in user_turns if (a.get("user_vulnerability_level", 0) or 0) >= 2]

    # ── C. Conversational Balance ──
    bot_questions = sum(1 for _, _, a in sys_turns if a.get("is_question"))
    bot_content = sum(1 for _, _, a in sys_turns if a.get("is_content_providing"))
    bot_question_dominance = safe_div(bot_questions, bot_content) if bot_content > 0 else safe_div(bot_questions, n_sys)

    user_questions = sum(1 for _, _, a in user_turns if a.get("is_question"))
    user_question_rate = safe_div(user_questions, n_user)

    topic_init = sum(1 for _, _, a in user_turns if a.get("is_topic_initiation"))
    user_topic_initiation_rate = safe_div(topic_init, n_user)

    # ── D. Memory & Continuity ──
    mem_refs = [(i, t, a) for i, t, a in sys_turns if a.get("memory_reference")]
    memory_reference_rate = safe_div(len(mem_refs), n_sys)

    # Cross-session question repetition (using embeddings)
    current_bot_questions_text = [t["text"] for _, t, a in sys_turns if a.get("is_question")]
    cross_session_question_repetition_rate = float("nan")
    max_semantic_similarity_to_prior_questions = float("nan")

    if current_bot_questions_text and prior_bot_question_embeddings:
        current_q_embs = get_embeddings(current_bot_questions_text)
        max_sims = []
        for cemb in current_q_embs:
            sims = [cosine_sim(cemb, pemb) for pemb in prior_bot_question_embeddings]
            max_sims.append(max(sims))
        # A question is "repeated" if max similarity > 0.85
        repeated = sum(1 for s in max_sims if s > 0.85)
        cross_session_question_repetition_rate = safe_div(repeated, len(current_bot_questions_text))
        max_semantic_similarity_to_prior_questions = max(max_sims) if max_sims else float("nan")
    elif not prior_bot_question_embeddings:
        cross_session_question_repetition_rate = 0.0
        max_semantic_similarity_to_prior_questions = 0.0

    # ── D2. Memory Accuracy (from existing annotations) ──
    correct_refs = sum(1 for _, _, a in sys_turns if a.get("memory_reference") and a.get("memory_reference_correct"))
    total_refs = len(mem_refs)
    memory_accuracy_rate = safe_div(correct_refs, total_refs)  # NaN if no refs

    # ── D3. Cross-session content repetition & novelty (new embeddings) ──
    if prior_bot_content_embeddings is None:
        prior_bot_content_embeddings = []
    current_bot_content_texts = [t["text"] for _, t, a in sys_turns if a.get("is_content_providing")]
    cross_session_content_repetition_rate = float("nan")
    bot_content_novelty_score = float("nan")

    if current_bot_content_texts and prior_bot_content_embeddings:
        current_content_embs = get_embeddings(current_bot_content_texts)
        max_sims = []
        for cemb in current_content_embs:
            sims = [cosine_sim(cemb, pemb) for pemb in prior_bot_content_embeddings]
            max_sims.append(max(sims))
        # Content is "repeated" if max similarity > 0.80 (lower than question threshold)
        repeated_content = sum(1 for s in max_sims if s > 0.80)
        cross_session_content_repetition_rate = safe_div(repeated_content, len(current_bot_content_texts))
        # Novelty: mean of (1 - max_sim) — higher = more novel
        bot_content_novelty_score = float(np.mean([1.0 - s for s in max_sims]))
    elif not prior_bot_content_embeddings:
        cross_session_content_repetition_rate = 0.0
        bot_content_novelty_score = 1.0  # first session: all content is novel

    # ── E. Answer Quality ──
    action_count = sum(1 for _, _, a in sys_turns if a.get("has_actionability_marker"))
    actionability_marker_rate = safe_div(action_count, n_sys)
    hedge_count = sum(1 for _, _, a in sys_turns if a.get("has_hedging"))
    hedging_rate = safe_div(hedge_count, n_sys)

    # Response specificity: combine example, actionability, and calibrated uncertainty markers
    example_count = sum(1 for _, _, a in sys_turns if a.get("has_example_marker"))
    calibrated_count = sum(1 for _, _, a in sys_turns if a.get("has_calibrated_uncertainty"))
    specificity_numerator = example_count + action_count + calibrated_count
    response_specificity_score = safe_div(specificity_numerator, n_sys)

    # ── F. Openings & Closings ──
    personalized_opening_presence = 0
    if sys_turns:
        personalized_opening_presence = 1 if sys_turns[0][2].get("personalized_opening") else 0

    # Positive closing indicator: check if last user turn is positive
    positive_closing_indicator = float("nan")
    if user_turns:
        last_text = user_turns[-1][1]["text"]
        closing_result = call_llm_json(
            "Rate the sentiment of this final user message. Return JSON: {\"sentiment\": \"positive\"|\"neutral\"|\"negative\"}",
            last_text,
        )
        sent = closing_result.get("sentiment", "")
        positive_closing_indicator = 1 if sent == "positive" else 0

    # ── G. Longitudinal ──
    current_mean_vuln = float(np.mean(vuln_levels)) if vuln_levels else float("nan")
    prior_vuln_mean = float(np.mean(prior_vuln_means)) if prior_vuln_means else float("nan")
    delta_user_vulnerability = current_mean_vuln - prior_vuln_mean if not (math.isnan(current_mean_vuln) or math.isnan(prior_vuln_mean)) else float("nan")

    # ── I. Additional metrics ──
    # B5: Follow-up depth after disclosure
    followup_count = 0
    for i, t, a in high_vuln_turns:
        for j, t2, a2 in sys_turns:
            if j > i:
                if a2.get("bot_open_ended_followup"):
                    followup_count += 1
                break
    deep_followup_rate = safe_div(followup_count, len(high_vuln_turns))

    # D2: User memory frustration (binary)
    frustration_with_bot_memory = 1 if any(a.get("user_memory_frustration") for _, _, a in user_turns) else 0

    # D4: User-initiated memory tests (binary)
    user_initiated_memory_test = 1 if any(a.get("user_memory_test") for _, _, a in user_turns) else 0

    # I1: Question about bot (binary)
    question_about_bot = 1 if any(a.get("user_question_about_bot") for _, _, a in user_turns) else 0

    # I2: Praises the bot (binary)
    praises_the_bot = 1 if any(a.get("user_praise_bot") for _, _, a in user_turns) else 0

    # I3: Name/identity use
    user_name_variants = [n.lower() for n in USER_NAMES.get(user_id, [])]
    user_name_used_by_bot = 0
    for _, t, _ in sys_turns:
        text_lower = t["text"].lower()
        for name in user_name_variants:
            user_name_used_by_bot += text_lower.count(name)

    # I4: User words per minute
    user_word_counts = [len(t["text"].split()) for _, t, _ in user_turns]
    total_user_words = sum(user_word_counts)
    if turns:
        session_duration_min = (turns[-1].get("end_sec", 0) - turns[0].get("start_sec", 0)) / 60.0
    else:
        session_duration_min = 0.0
    user_words_per_minute = safe_div(total_user_words, session_duration_min)

    # H1: Max prior vulnerability
    max_prior_vulnerability = max(prior_max_vuln) if prior_max_vuln else 0

    # ── Build output ──
    metrics = {
        "session_id": session_id,
        # A. Emotional & Empathic Responsiveness
        "user_emotion_expression_rate": user_emotion_expression_rate,
        "emotional_support_validation_rate": emotional_support_validation_rate,
        # B. Social Penetration & Disclosure Dynamics
        "user_self_disclosure_rate": user_self_disclosure_rate,
        "max_user_vulnerability_level": max_user_vulnerability_level,
        "bot_self_disclosure_rate": bot_self_disclosure_rate,
        "mean_bot_self_disclosure_level": mean_bot_self_disclosure_level,
        "deep_followup_rate": deep_followup_rate,
        # C. Conversational Balance & User Agency
        "bot_question_dominance": bot_question_dominance,
        "user_question_rate": user_question_rate,
        "user_topic_initiation_rate": user_topic_initiation_rate,
        # D. Memory & Conversational Continuity
        "memory_reference_rate": memory_reference_rate,
        "memory_accuracy_rate": memory_accuracy_rate,
        "frustration_with_bot_memory": frustration_with_bot_memory,
        "cross_session_question_repetition_rate": cross_session_question_repetition_rate,
        "cross_session_content_repetition_rate": cross_session_content_repetition_rate,
        "bot_content_novelty_score": bot_content_novelty_score,
        "user_initiated_memory_test": user_initiated_memory_test,
        # E. Answer Quality & Specificity
        "actionability_marker_rate": actionability_marker_rate,
        "hedging_rate": hedging_rate,
        "response_specificity_score": response_specificity_score,
        # F. Openings, Closings, and Recency Effects
        "personalized_opening_presence": personalized_opening_presence,
        "positive_closing_indicator": positive_closing_indicator,
        # G. Longitudinal Change
        "delta_user_vulnerability": delta_user_vulnerability,
        # H. Scores From Prior Sessions
        "max_prior_vulnerability": max_prior_vulnerability,
        # I. Others
        "question_about_bot": question_about_bot,
        "praises_the_bot": praises_the_bot,
        "user_name_used_by_bot": user_name_used_by_bot,
        "user_words_per_minute": user_words_per_minute,
    }
    return metrics, current_mean_vuln, current_bot_questions_text, max_user_vulnerability_level, current_bot_content_texts


def process_user(user_dir: Path):
    """Process all sessions for one user and write CSV."""
    user_id = user_dir.name
    print(f"Processing user: {user_id}")

    # Find and sort transcript files
    tsv_files = sorted(glob.glob(str(user_dir / "session_*_transcript.tsv")))
    if not tsv_files:
        print(f"  No transcripts found for {user_id}, skipping.")
        return

    # Cross-session state
    prior_bot_question_embeddings: list[list[float]] = []
    prior_bot_content_embeddings: list[list[float]] = []
    prior_vuln_means: list[float] = []
    prior_max_vuln: list[int] = []
    prior_all_embeddings: list[list[float]] = []
    prior_facts: list[str] = []
    prior_fact_embeddings: list[list[float]] = []

    all_metrics = []

    for tsv_path in tsv_files:
        session_id = Path(tsv_path).stem.replace("_transcript", "")
        print(f"  Session: {session_id}")

        turns = parse_tsv(Path(tsv_path))
        if not turns:
            print(f"    Empty transcript, skipping.")
            continue

        # Call 1: turn-level annotation
        annotations = annotate_turns(turns)

        # Compute metrics
        metrics, current_vuln_mean, current_bot_q_texts, session_max_vuln, current_bot_content_texts = compute_metrics(
            turns, annotations, session_id, user_id,
            prior_bot_question_embeddings,
            prior_vuln_means,
            prior_max_vuln,
            prior_bot_content_embeddings,
        )

        # ── Tier 2 features ──
        # A1: Keyword-based memory references
        kw_mem_count, kw_mem_rate = compute_keyword_memory_refs(turns)
        metrics["keyword_memory_ref_count"] = kw_mem_count
        metrics["keyword_memory_ref_rate"] = kw_mem_rate

        # A2: Within-session semantic alignment
        align_mean, align_var = compute_user_bot_alignment(turns)
        metrics["user_bot_semantic_alignment"] = align_mean
        metrics["alignment_variance"] = align_var

        # A3: User-side topic novelty
        user_novelty, sess_novelty, topic_div, current_user_embs = compute_user_topic_novelty(
            turns, prior_all_embeddings,
        )
        metrics["user_topic_novelty_score"] = user_novelty
        metrics["session_topic_novelty_score"] = sess_novelty
        metrics["topic_diversity_within_session"] = topic_div

        # A4: Fact extraction + retrieval-based memory accuracy
        session_facts = extract_facts_from_session(turns, session_id, user_id)
        fact_acc, n_verified, n_contradicted = compute_fact_based_memory_accuracy(
            turns, prior_facts, prior_fact_embeddings,
        )
        metrics["fact_based_memory_accuracy"] = fact_acc
        metrics["fact_based_n_verified"] = n_verified
        metrics["fact_based_n_contradicted"] = n_contradicted
        metrics["cumulative_fact_count"] = len(prior_facts)

        all_metrics.append(metrics)

        # Update cross-session state
        if not math.isnan(current_vuln_mean):
            prior_vuln_means.append(current_vuln_mean)
        prior_max_vuln.append(session_max_vuln)

        # Accumulate bot question embeddings for future sessions
        if current_bot_q_texts:
            new_embs = get_embeddings(current_bot_q_texts)
            prior_bot_question_embeddings.extend(new_embs)

        # Accumulate bot content embeddings for future sessions
        if current_bot_content_texts:
            new_content_embs = get_embeddings(current_bot_content_texts)
            prior_bot_content_embeddings.extend(new_content_embs)

        # Accumulate all embeddings for topic novelty (A3)
        if current_user_embs:
            prior_all_embeddings.extend(current_user_embs)

        # Accumulate facts for memory accuracy (A4)
        if session_facts:
            prior_facts.extend(session_facts)
            fact_embs = get_embeddings(session_facts)
            prior_fact_embeddings.extend(fact_embs)

    if not all_metrics:
        print(f"  No metrics computed for {user_id}.")
        return

    # Write CSV
    output_path = user_dir / "text_metrics_claude.csv"
    fieldnames = list(all_metrics[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_metrics:
            writer.writerow(row)
    print(f"  Wrote {output_path} ({len(all_metrics)} rows × {len(fieldnames)} cols)")


def aggregate_all_users():
    """Read all per-user text_metrics_claude.csv files and produce all_text_features.csv."""
    import pandas as pd

    all_dfs = []
    for d in sorted(DATASET_DIR.iterdir()):
        if d.is_dir() and re.match(r"^\d+$", d.name):
            csv_path = d / "text_metrics_claude.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df.insert(0, "participant_id", int(d.name))
                all_dfs.append(df)

    if not all_dfs:
        print("No per-user CSVs found for aggregation.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = BASE_DIR / "data" / "all_text_features.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nAggregated {len(all_dfs)} users → {out_path} ({combined.shape[0]} rows × {combined.shape[1]} cols)")


def main():
    parser = argparse.ArgumentParser(description="Extract text-based metrics from conversation transcripts.")
    parser.add_argument("--user_id", type=str, default=None, help="Process only this user ID (directory name). If not set, process all.")
    parser.add_argument("--aggregate_only", action="store_true", help="Only aggregate existing CSVs without reprocessing.")
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate_all_users()
        return

    if args.user_id:
        user_dir = DATASET_DIR / args.user_id
        if not user_dir.is_dir():
            print(f"User directory not found: {user_dir}")
            return
        process_user(user_dir)
    else:
        # Process all numeric-only user directories
        for d in sorted(DATASET_DIR.iterdir()):
            if d.is_dir() and re.match(r"^\d+$", d.name):
                process_user(d)

    # Aggregate all per-user CSVs into one file
    aggregate_all_users()

    print_cost_summary()


if __name__ == "__main__":
    main()
