#!/usr/bin/env python3
"""
Annotation Validation: Inter-Annotator Agreement (IAA) across 3 LLMs.

Re-annotates a stratified random sample of user turns, bot turns, and
user-bot pairs using two alternative LLMs (GPT-5.4-nano and Gemini-3.1-
Flash-Lite) and computes IAA metrics against the original GPT-4.1
annotations.

Metrics:
  - Binary fields: pairwise Cohen's kappa, 3-way Fleiss' kappa,
    percent agreement.
  - Ordinal fields: pairwise quadratic-weighted Cohen's kappa,
    3-way Krippendorff's alpha (ordinal distance).

Usage:
    uv run --directory .../Equ python code/text/annotation_validation.py
"""

import os
import re
import json
import time
import random
import hashlib
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment
load_dotenv()
load_dotenv()  # also load local .env if present

from openai import OpenAI
from google import genai
from sklearn.metrics import cohen_kappa_score
import krippendorff

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

DATASET_ROOT = str(Path(__file__).resolve().parents[1] / "data" / "processed_videos")
CACHE_DB_PATH = os.path.join(DATASET_ROOT, "cache_chatgpt.db")
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "validation_results"
)
SEED = 42

# Model identifiers
ORIGINAL_MODEL = "gpt-4.1-2025-04-14"
ALT_MODEL_OPENAI = "gpt-5.4-nano"
ALT_MODEL_GEMINI = "gemini-3.1-flash-lite-preview"

# Sample sizes
N_USER_TURNS = 150
N_BOT_TURNS = 150
N_PAIRS = 100

# Field definitions
USER_FIELDS = [
    "emotion_present", "emotion_valence", "vuln_level",
    "praise_bot", "question_about_bot", "memory_test", "memory_frustration",
]
BOT_FIELDS = [
    "is_question", "self_disclosure_level", "informative",
    "actionable", "hedge_strength", "memory_reference",
]
PAIR_FIELDS = [
    "ack_emotion", "validate_emotion", "offer_support",
    "open_ended_followup", "topic_shift",
]

BINARY_USER = ["emotion_present", "praise_bot", "question_about_bot", "memory_test", "memory_frustration"]
ORDINAL_USER = ["emotion_valence", "vuln_level"]

BINARY_BOT = ["is_question", "informative", "actionable", "memory_reference"]
ORDINAL_BOT = ["self_disclosure_level", "hedge_strength"]

BINARY_PAIR = PAIR_FIELDS  # all binary

# Prompts — exact copies from chatgpt_pro_text.py
USER_TURN_SYSTEM_PROMPT = """You are a careful behavioral annotation assistant.
You will receive a JSON array of user utterances (single turns). For each item, output exactly one JSON object with:

- id: same as input
- emotion_present: 0/1  (does the user express an emotional state explicitly or implicitly)
- emotion_valence: -1/0/+1  (-1 negative, 0 neutral/mixed, +1 positive)
- vuln_level: 0/1/2/3 using this rubric:
  0 = No personal info (facts unrelated to self)
  1 = Personal facts/preferences (likes, habits, non-sensitive)
  2 = Personal experiences or emotions (tired, stressed, struggled)
  3 = Deeply personal/sensitive worries, loneliness, trauma, strong negative emotions, etc.
- praise_bot: 0/1  (user compliments/praises the bot)
- question_about_bot: 0/1  (user asks about the bot itself: identity, preferences, feelings)
- memory_test: 0/1  (user asks if bot remembers: "do you remember", "last time I said")
- memory_frustration: 0/1 (user indicates bot forgot/repeated: "we already talked about this")

Output ONLY valid JSON: a single JSON array of objects. No extra text.
"""

BOT_TURN_SYSTEM_PROMPT = """You are a careful behavioral annotation assistant.
You will receive a JSON array of bot utterances (single turns). For each item, output exactly one JSON object with:

- id: same as input
- is_question: 0/1 (the bot asks the user something or prompts them to answer; includes "Tell me more about X.")
- self_disclosure_level: 0/1/2/3:
  0 = no self-disclosure
  1 = mild preference/habit ("I like helping people")
  2 = experiential framing ("I've talked to others who...")
  3 = values/opinions/perspective ("I believe reflection is important")
- informative: 0/1 (bot provides explanations/advice/substantive content; not just backchannels)
- actionable: 0/1 (contains concrete advice / steps / specific suggestions the user can do)
- hedge_strength: 0/1/2 where:
  0 = not hedged
  1 = mildly hedged ("might", "maybe", "could")
  2 = strongly hedged/generic ("it depends", "in general", "some people say", very vague)
- memory_reference: 0/1 (bot refers to prior sessions or user-specific past info: "last time you mentioned...")

Output ONLY valid JSON: a single JSON array of objects. No extra text.
"""

PAIR_SYSTEM_PROMPT = """You are a careful conversational annotator.
You will receive a JSON array of items, each containing:
- id
- user_text
- user_emotion_present (0/1)
- user_vuln_level (0-3)
- bot_text

For each item, output one JSON object with:
- id
- ack_emotion: 0/1 (bot acknowledges user's emotion)
- validate_emotion: 0/1 (bot validates emotion as reasonable/understandable)
- offer_support: 0/1 (bot offers comfort/reassurance/emotional support)
- open_ended_followup: 0/1 (bot invites elaboration with an open-ended follow-up, e.g., "How did that feel?", "Tell me more")
- topic_shift: 0/1 (bot shifts away from the user's disclosed topic immediately)

If user_emotion_present=0, then ack_emotion/validate_emotion/offer_support should typically be 0 unless bot still responds emotionally.
If user_vuln_level<2, open_ended_followup can still be 1 if it genuinely invites elaboration.

Output ONLY valid JSON: a single JSON array of objects. No extra text.
"""

# ──────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def parse_time_to_seconds(t: str) -> float:
    if pd.isna(t):
        return np.nan
    t = str(t).strip().replace(",", ".")
    parts = t.split(":")
    if len(parts) != 3:
        raise ValueError(f"Unexpected time format: {t}")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


# ──────────────────────────────────────────────────────────────────────
# Data loading  (mirrors chatgpt_pro_text.py)
# ──────────────────────────────────────────────────────────────────────

def load_all_turns(dataset_root: str) -> pd.DataFrame:
    """Load turns from all participants via TSV transcripts."""
    import glob as _glob

    pids = sorted(
        int(d) for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d)) and d.isdigit()
    )
    all_rows = []
    for pid in pids:
        pdir = os.path.join(dataset_root, str(pid))
        tsvs = sorted(_glob.glob(os.path.join(pdir, "session_*_transcript.tsv")))
        for tsv in tsvs:
            m = re.search(r"session_(\d+)_transcript\.tsv$", tsv)
            if not m:
                continue
            snum = int(m.group(1))
            df = pd.read_csv(tsv, sep="\t", comment="#")
            df.columns = [c.strip() for c in df.columns]
            df["text"] = df["text"].astype(str).map(normalize_text)
            df["speaker"] = df["speaker"].astype(str).str.lower().str.strip()
            df["speaker"] = df["speaker"].replace({"system": "bot", "assistant": "bot"})
            df = df[df["text"].map(lambda x: isinstance(x, str) and len(x) > 0)].copy()
            df["start_sec"] = df["start_time"].map(parse_time_to_seconds)
            df["end_sec"] = df["end_time"].map(parse_time_to_seconds)
            df = df.sort_values(["start_sec", "end_sec"]).reset_index(drop=True)
            df["turn_index"] = np.arange(len(df), dtype=int)
            df["participant_id"] = pid
            df["session_num"] = snum
            df["session_id"] = f"{pid:02d}_s{snum:02d}"
            df["turn_id"] = df.apply(
                lambda r: f"p{pid:02d}_s{snum:02d}_t{int(r['turn_index']):04d}", axis=1
            )
            df["n_words"] = df["text"].map(lambda s: len(s.split()) if s else 0)
            all_rows.append(df)

    big = pd.concat(all_rows, ignore_index=True)
    big["text_norm"] = big["text"].map(normalize_text)
    big["text_hash"] = big["text_norm"].map(sha1_text)
    return big


# ──────────────────────────────────────────────────────────────────────
# SQLite cache reader (read-only for original GPT-4.1 annotations)
# ──────────────────────────────────────────────────────────────────────

class CacheReader:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path)

    def get(self, k: str) -> Optional[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT v FROM kv WHERE k=?", (k,))
        row = cur.fetchone()
        return row[0] if row else None

    def set(self, k: str, v: str):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO kv(k,v,created_at) VALUES(?,?,?)",
            (k, v, time.time()),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()


# ──────────────────────────────────────────────────────────────────────
# LLM callers
# ──────────────────────────────────────────────────────────────────────

def call_openai_json(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 3,
) -> Any:
    """Call OpenAI chat completion expecting JSON output."""
    client = OpenAI(timeout=180.0)  # generous timeout for batched JSON
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content.strip()
            m = re.search(r"(\[.*\]|\{.*\})", text, flags=re.DOTALL)
            if m:
                text = m.group(1)
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                for val in parsed.values():
                    if isinstance(val, list):
                        return val
            return parsed
        except Exception as e:
            last_err = e
            print(f"      Attempt {attempt+1}/{max_retries} failed: {type(e).__name__}: {str(e)[:100]}")
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"OpenAI JSON parse failed after {max_retries} attempts: {last_err}")


def call_gemini_json(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 3,
) -> Any:
    """Call Gemini via google-genai expecting JSON output."""
    client = genai.Client()
    combined_prompt = (
        f"[SYSTEM INSTRUCTIONS]\n{system_prompt}\n\n"
        f"[INPUT]\n{user_prompt}"
    )
    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=combined_prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0,
                },
            )
            text = response.text.strip()
            m = re.search(r"(\[.*\]|\{.*\})", text, flags=re.DOTALL)
            if m:
                text = m.group(1)
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                for val in parsed.values():
                    if isinstance(val, list):
                        return val
            return parsed
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Gemini JSON parse failed after {max_retries} attempts: {last_err}")


# ──────────────────────────────────────────────────────────────────────
# Annotation logic
# ──────────────────────────────────────────────────────────────────────

def annotate_batch(
    items: List[Dict],
    system_prompt: str,
    model_name: str,
    provider: str,   # "openai" or "gemini"
    fields: List[str],
    batch_size: int = 1,  # default to single-item to avoid proxy timeouts
    cache: Optional[CacheReader] = None,
    cache_prefix: str = "",
) -> Dict[str, Dict[str, int]]:
    """
    Annotate a list of items and return {id: {field: value, ...}}.
    Uses cache if available (for reading/writing intermediate results).
    Sends one item at a time to avoid proxy buffering timeouts.
    """
    results: Dict[str, Dict[str, int]] = {}

    # Check cache for already-done items
    to_do = []
    for item in items:
        item_id = item["id"]
        if cache and cache_prefix:
            ck = f"val|{cache_prefix}|{model_name}|{item_id}"
            cached = cache.get(ck)
            if cached is not None:
                results[item_id] = json.loads(cached)
                continue
        to_do.append(item)

    if not to_do:
        return results

    n_cached = len(items) - len(to_do)
    print(f"    {len(to_do)} items to annotate ({n_cached} from cache)")
    caller = call_openai_json if provider == "openai" else call_gemini_json

    for idx, item in enumerate(tqdm(to_do, desc=f"  {model_name}", leave=False)):
        item_id = item["id"]
        payload = json.dumps([item], ensure_ascii=False)

        annotation = {f: 0 for f in fields}
        try:
            out = caller(model_name, system_prompt, payload)
            # Parse response: could be list or dict
            if isinstance(out, dict) and "id" in out:
                obj = out
            elif isinstance(out, list) and len(out) >= 1:
                obj = out[0] if isinstance(out[0], dict) else {}
            elif isinstance(out, dict):
                # Might be wrapped: {"result": [{...}]}
                obj = out
            else:
                obj = {}
            annotation = {f: safe_int(obj.get(f, 0)) for f in fields}
        except Exception as e:
            if idx < 3:  # only print first few warnings
                print(f"      WARNING: item {item_id} failed ({type(e).__name__}), using defaults")

        results[item_id] = annotation
        # Cache immediately
        if cache and cache_prefix:
            ck = f"val|{cache_prefix}|{model_name}|{item_id}"
            cache.set(ck, json.dumps(annotation))

        # Small delay to be polite to APIs
        time.sleep(0.05)

    return results


# ──────────────────────────────────────────────────────────────────────
# IAA computation
# ──────────────────────────────────────────────────────────────────────

def compute_fleiss_kappa(ratings_matrix: np.ndarray, n_categories: int) -> float:
    """
    Compute Fleiss' kappa.
    ratings_matrix: (N_items, N_raters) with integer category labels.
    """
    N, n_raters = ratings_matrix.shape
    # Build the count matrix: (N, n_categories)
    counts = np.zeros((N, n_categories), dtype=float)
    for cat in range(n_categories):
        counts[:, cat] = (ratings_matrix == cat).sum(axis=1)

    # Overall proportion per category
    p_j = counts.sum(axis=0) / (N * n_raters)

    # P_i for each item
    P_i = (np.sum(counts ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))

    P_bar = P_i.mean()
    P_e = np.sum(p_j ** 2)

    if abs(1 - P_e) < 1e-10:
        return 1.0 if abs(P_bar - 1.0) < 1e-10 else 0.0

    kappa = (P_bar - P_e) / (1 - P_e)
    return float(kappa)


def pairwise_cohens_kappa(
    raters: Dict[str, List[int]],
    weights: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute pairwise Cohen's kappa between all pairs of raters.
    raters: {rater_name: [ratings for each item]}
    weights: None for binary, "quadratic" for ordinal.
    Returns {pair_name: kappa}.
    """
    names = sorted(raters.keys())
    results = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = np.array(raters[names[i]])
            b = np.array(raters[names[j]])
            # Filter out any NaN positions
            mask = ~(np.isnan(a) | np.isnan(b))
            if mask.sum() < 5:
                results[f"{names[i]} vs {names[j]}"] = float("nan")
                continue
            try:
                k = cohen_kappa_score(
                    a[mask].astype(int),
                    b[mask].astype(int),
                    weights=weights,
                )
                results[f"{names[i]} vs {names[j]}"] = float(k)
            except Exception:
                results[f"{names[i]} vs {names[j]}"] = float("nan")
    return results


def pairwise_percent_agreement(raters: Dict[str, List[int]]) -> Dict[str, float]:
    """Percent agreement for all rater pairs."""
    names = sorted(raters.keys())
    results = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = np.array(raters[names[i]])
            b = np.array(raters[names[j]])
            mask = ~(np.isnan(a) | np.isnan(b))
            if mask.sum() == 0:
                results[f"{names[i]} vs {names[j]}"] = float("nan")
                continue
            results[f"{names[i]} vs {names[j]}"] = float(
                (a[mask] == b[mask]).mean()
            )
    return results


def compute_iaa_for_field(
    annotations: Dict[str, Dict[str, Dict[str, int]]],
    field: str,
    is_ordinal: bool,
    item_ids: List[str],
    model_names: List[str],
) -> Dict[str, Any]:
    """
    Compute IAA metrics for a single field.
    annotations: {model_name: {item_id: {field: value}}}
    """
    # Build rater vectors
    raters = {}
    for model in model_names:
        vals = []
        for iid in item_ids:
            v = annotations.get(model, {}).get(iid, {}).get(field, None)
            vals.append(float(v) if v is not None else float("nan"))
        raters[model] = vals

    # Pairwise Cohen's kappa
    weights = "quadratic" if is_ordinal else None
    pw_kappa = pairwise_cohens_kappa(raters, weights=weights)

    # Percent agreement
    pw_agree = pairwise_percent_agreement(raters)

    # 3-way metrics
    # Build matrix (N, 3)
    matrix = np.column_stack([np.array(raters[m]) for m in model_names])

    # Remove items where any rater has NaN
    valid_mask = ~np.isnan(matrix).any(axis=1)
    matrix_clean = matrix[valid_mask].astype(int)

    fleiss_k = float("nan")
    kripp_alpha = float("nan")

    if len(matrix_clean) >= 5:
        if is_ordinal:
            # Krippendorff's alpha with ordinal level
            try:
                # krippendorff expects (raters x items) matrix
                kripp_alpha = krippendorff.alpha(
                    reliability_data=matrix_clean.T,
                    level_of_measurement="ordinal",
                )
            except Exception as e:
                print(f"    Krippendorff alpha failed for {field}: {e}")
        else:
            # Fleiss' kappa
            unique_cats = sorted(set(matrix_clean.flatten()))
            n_cats = max(unique_cats) + 1 if unique_cats else 2
            # Ensure at least 2 categories
            n_cats = max(n_cats, 2)
            try:
                fleiss_k = compute_fleiss_kappa(matrix_clean, n_cats)
            except Exception as e:
                print(f"    Fleiss kappa failed for {field}: {e}")

            # Also compute Krippendorff alpha (nominal) for binary
            try:
                kripp_alpha = krippendorff.alpha(
                    reliability_data=matrix_clean.T,
                    level_of_measurement="nominal",
                )
            except Exception:
                pass

    kappa_vals = [v for v in pw_kappa.values() if not np.isnan(v)]
    agree_vals = [v for v in pw_agree.values() if not np.isnan(v)]

    result = {
        "field": field,
        "type": "ordinal" if is_ordinal else "binary",
        "n_items": int(valid_mask.sum()),
        "pairwise_kappa": pw_kappa,
        "mean_pairwise_kappa": float(np.mean(kappa_vals)) if kappa_vals else float("nan"),
        "min_pairwise_kappa": float(np.min(kappa_vals)) if kappa_vals else float("nan"),
        "max_pairwise_kappa": float(np.max(kappa_vals)) if kappa_vals else float("nan"),
        "pairwise_agreement": pw_agree,
        "mean_agreement": float(np.mean(agree_vals)) if agree_vals else float("nan"),
    }

    if is_ordinal:
        result["krippendorff_alpha_ordinal"] = float(kripp_alpha) if not np.isnan(kripp_alpha) else None
    else:
        result["fleiss_kappa"] = float(fleiss_k) if not np.isnan(fleiss_k) else None
        result["krippendorff_alpha_nominal"] = float(kripp_alpha) if not np.isnan(kripp_alpha) else None

    return result


# ──────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────

def stratified_sample(
    df: pd.DataFrame,
    n: int,
    group_col: str = "participant_id",
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Stratified sampling across participants, roughly equal per group."""
    if rng is None:
        rng = np.random.default_rng(SEED)

    groups = sorted(df[group_col].unique())
    per_group = max(1, n // len(groups))
    remainder = n - per_group * len(groups)

    samples = []
    for g in groups:
        gdf = df[df[group_col] == g]
        take = min(per_group, len(gdf))
        samples.append(gdf.sample(n=take, random_state=int(rng.integers(1e9))))

    combined = pd.concat(samples, ignore_index=True)

    # If we need more due to rounding, sample from the rest
    if len(combined) < n:
        remaining = df[~df.index.isin(combined.index)]
        extra = min(n - len(combined), len(remaining))
        if extra > 0:
            combined = pd.concat(
                [combined, remaining.sample(n=extra, random_state=int(rng.integers(1e9)))],
                ignore_index=True,
            )

    return combined.head(n)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("=" * 60)
    print("Annotation Validation: 3-LLM Inter-Annotator Agreement")
    print("=" * 60)

    # ── Load all turns ──────────────────────────────────────────────
    print("\n[1/6] Loading conversation data...")
    all_turns = load_all_turns(DATASET_ROOT)
    user_turns = all_turns[all_turns["speaker"] == "user"].copy()
    bot_turns = all_turns[all_turns["speaker"] == "bot"].copy()
    print(f"  Total: {len(user_turns)} user turns, {len(bot_turns)} bot turns")
    print(f"  Participants: {sorted(all_turns['participant_id'].unique())}")

    # ── Open cache for original annotations ─────────────────────────
    cache = CacheReader(CACHE_DB_PATH)

    # ── Sample user turns ───────────────────────────────────────────
    print(f"\n[2/6] Sampling {N_USER_TURNS} user turns (stratified by participant)...")
    user_sample = stratified_sample(user_turns, N_USER_TURNS, rng=rng)
    print(f"  Sampled {len(user_sample)} user turns across {user_sample['participant_id'].nunique()} participants")

    # ── Sample bot turns ────────────────────────────────────────────
    print(f"  Sampling {N_BOT_TURNS} bot turns (stratified by participant)...")
    bot_sample = stratified_sample(bot_turns, N_BOT_TURNS, rng=rng)
    print(f"  Sampled {len(bot_sample)} bot turns across {bot_sample['participant_id'].nunique()} participants")

    # ── Retrieve original GPT-4.1 annotations from cache ───────────
    print("\n[3/6] Retrieving / generating original GPT-4.1 annotations...")

    # For user turns: try to retrieve from cache, otherwise annotate
    orig_user: Dict[str, Dict[str, int]] = {}
    user_need_annotation = []
    for _, row in user_sample.iterrows():
        tid = row["turn_id"]
        thash = row["text_hash"]
        key = f"turn_user|{ORIGINAL_MODEL}|v1|{tid}|{thash}"
        cached = cache.get(key)
        if cached is not None:
            orig_user[tid] = json.loads(cached)
        else:
            user_need_annotation.append(row)

    print(f"  User turns: {len(orig_user)} from cache, {len(user_need_annotation)} need annotation")

    if user_need_annotation:
        print("  Annotating missing user turns with GPT-4.1...")
        items = [{"id": r["turn_id"], "text": r["text_norm"]}
                 for _, r in pd.DataFrame(user_need_annotation).iterrows()]
        result = annotate_batch(
            items, USER_TURN_SYSTEM_PROMPT, ORIGINAL_MODEL,
            "openai", USER_FIELDS, batch_size=15, cache=cache,
            cache_prefix="turn_user_orig",
        )
        # Also store in the main cache format for future use
        for item_row in user_need_annotation:
            tid = item_row["turn_id"]
            thash = item_row["text_hash"]
            if tid in result:
                key = f"turn_user|{ORIGINAL_MODEL}|v1|{tid}|{thash}"
                cache.set(key, json.dumps(result[tid]))
                orig_user[tid] = result[tid]

    # For bot turns
    orig_bot: Dict[str, Dict[str, int]] = {}
    bot_need_annotation = []
    for _, row in bot_sample.iterrows():
        tid = row["turn_id"]
        thash = row["text_hash"]
        key = f"turn_bot|{ORIGINAL_MODEL}|v1|{tid}|{thash}"
        cached = cache.get(key)
        if cached is not None:
            orig_bot[tid] = json.loads(cached)
        else:
            bot_need_annotation.append(row)

    print(f"  Bot turns: {len(orig_bot)} from cache, {len(bot_need_annotation)} need annotation")

    if bot_need_annotation:
        print("  Annotating missing bot turns with GPT-4.1...")
        items = [{"id": r["turn_id"], "text": r["text_norm"]}
                 for _, r in pd.DataFrame(bot_need_annotation).iterrows()]
        result = annotate_batch(
            items, BOT_TURN_SYSTEM_PROMPT, ORIGINAL_MODEL,
            "openai", BOT_FIELDS, batch_size=15, cache=cache,
            cache_prefix="turn_bot_orig",
        )
        for item_row in bot_need_annotation:
            tid = item_row["turn_id"]
            thash = item_row["text_hash"]
            if tid in result:
                key = f"turn_bot|{ORIGINAL_MODEL}|v1|{tid}|{thash}"
                cache.set(key, json.dumps(result[tid]))
                orig_bot[tid] = result[tid]

    # ── Build pair sample ───────────────────────────────────────────
    # We need user turns that have emotion_present=1 or vuln_level>=2,
    # and have a subsequent bot turn.
    print("\n  Building pair sample (emotion_present=1 or vuln_level>=2)...")

    # Get original annotations for ALL user turns to identify eligible pairs
    # We'll use the sampled + any needed extras
    pair_candidates = []
    for _, row in all_turns[all_turns["speaker"] == "user"].iterrows():
        tid = row["turn_id"]
        thash = row["text_hash"]
        key = f"turn_user|{ORIGINAL_MODEL}|v1|{tid}|{thash}"
        cached_val = cache.get(key)
        if cached_val is not None:
            ann = json.loads(cached_val)
            if ann.get("emotion_present", 0) == 1 or ann.get("vuln_level", 0) >= 2:
                pair_candidates.append({
                    "participant_id": row["participant_id"],
                    "session_num": row["session_num"],
                    "turn_id": tid,
                    "text": row["text_norm"],
                    "text_hash": thash,
                    "user_emotion_present": ann.get("emotion_present", 0),
                    "user_vuln_level": ann.get("vuln_level", 0),
                })

    # If not enough from cache, also check sampled user turns we just annotated
    for tid, ann in orig_user.items():
        if ann.get("emotion_present", 0) == 1 or ann.get("vuln_level", 0) >= 2:
            row_match = user_sample[user_sample["turn_id"] == tid]
            if not row_match.empty and tid not in [c["turn_id"] for c in pair_candidates]:
                r = row_match.iloc[0]
                pair_candidates.append({
                    "participant_id": r["participant_id"],
                    "session_num": r["session_num"],
                    "turn_id": tid,
                    "text": r["text_norm"],
                    "text_hash": r["text_hash"],
                    "user_emotion_present": ann.get("emotion_present", 0),
                    "user_vuln_level": ann.get("vuln_level", 0),
                })

    print(f"  Found {len(pair_candidates)} eligible pair candidates")

    # Now find next bot turn for each candidate
    pair_items = []
    for cand in pair_candidates:
        pid = cand["participant_id"]
        snum = cand["session_num"]
        utid = cand["turn_id"]

        # Find this turn in all_turns and get next bot
        sess_turns = all_turns[
            (all_turns["participant_id"] == pid) &
            (all_turns["session_num"] == snum)
        ].sort_values("turn_index")

        # Find user turn index
        user_row = sess_turns[sess_turns["turn_id"] == utid]
        if user_row.empty:
            continue
        uidx = user_row.iloc[0]["turn_index"]

        # Find next bot turn
        next_bots = sess_turns[
            (sess_turns["turn_index"] > uidx) &
            (sess_turns["speaker"] == "bot")
        ]
        if next_bots.empty:
            continue

        bot_row = next_bots.iloc[0]
        pair_id = f"{utid}__{bot_row['turn_id']}"
        pair_items.append({
            "pair_id": pair_id,
            "participant_id": pid,
            "user_turn_id": utid,
            "bot_turn_id": bot_row["turn_id"],
            "user_text": cand["text"],
            "bot_text": normalize_text(bot_row["text"]),
            "user_emotion_present": cand["user_emotion_present"],
            "user_vuln_level": cand["user_vuln_level"],
        })

    pair_df = pd.DataFrame(pair_items)
    if len(pair_df) > N_PAIRS:
        pair_df = pair_df.sample(n=N_PAIRS, random_state=int(rng.integers(1e9)))
    print(f"  Final pair sample: {len(pair_df)} pairs")

    # Get original pair annotations
    orig_pair: Dict[str, Dict[str, int]] = {}
    pair_need_annotation = []
    for _, row in pair_df.iterrows():
        pid = row["pair_id"]
        uhash = sha1_text(row["user_text"])
        bhash = sha1_text(row["bot_text"])
        key = f"pair|{ORIGINAL_MODEL}|v1|{pid}|{uhash}|{bhash}"
        cached_val = cache.get(key)
        if cached_val is not None:
            orig_pair[pid] = json.loads(cached_val)
        else:
            pair_need_annotation.append(row)

    print(f"  Pair annotations: {len(orig_pair)} from cache, {len(pair_need_annotation)} need annotation")

    if pair_need_annotation:
        print("  Annotating missing pairs with GPT-4.1...")
        items = []
        for _, r in pd.DataFrame(pair_need_annotation).iterrows():
            items.append({
                "id": r["pair_id"],
                "user_text": r["user_text"],
                "user_emotion_present": int(r["user_emotion_present"]),
                "user_vuln_level": int(r["user_vuln_level"]),
                "bot_text": r["bot_text"],
            })
        result = annotate_batch(
            items, PAIR_SYSTEM_PROMPT, ORIGINAL_MODEL,
            "openai", PAIR_FIELDS, batch_size=15, cache=cache,
            cache_prefix="pair_orig",
        )
        for _, r in pd.DataFrame(pair_need_annotation).iterrows():
            pid = r["pair_id"]
            if pid in result:
                uhash = sha1_text(r["user_text"])
                bhash = sha1_text(r["bot_text"])
                key = f"pair|{ORIGINAL_MODEL}|v1|{pid}|{uhash}|{bhash}"
                cache.set(key, json.dumps(result[pid]))
                orig_pair[pid] = result[pid]

    # ── Annotate with alternative LLMs ──────────────────────────────
    print("\n[4/6] Annotating samples with alternative LLMs...")

    # -- GPT-5.4-nano (user turns) --
    print(f"\n  === {ALT_MODEL_OPENAI} ===")
    print("  User turns...")
    nano_user_items = [{"id": r["turn_id"], "text": r["text_norm"]}
                       for _, r in user_sample.iterrows()]
    nano_user = annotate_batch(
        nano_user_items, USER_TURN_SYSTEM_PROMPT, ALT_MODEL_OPENAI,
        "openai", USER_FIELDS, batch_size=15, cache=cache,
        cache_prefix="val_user",
    )

    print("  Bot turns...")
    nano_bot_items = [{"id": r["turn_id"], "text": r["text_norm"]}
                      for _, r in bot_sample.iterrows()]
    nano_bot = annotate_batch(
        nano_bot_items, BOT_TURN_SYSTEM_PROMPT, ALT_MODEL_OPENAI,
        "openai", BOT_FIELDS, batch_size=15, cache=cache,
        cache_prefix="val_bot",
    )

    print("  Pairs...")
    nano_pair_items = []
    for _, r in pair_df.iterrows():
        nano_pair_items.append({
            "id": r["pair_id"],
            "user_text": r["user_text"],
            "user_emotion_present": int(r["user_emotion_present"]),
            "user_vuln_level": int(r["user_vuln_level"]),
            "bot_text": r["bot_text"],
        })
    nano_pair = annotate_batch(
        nano_pair_items, PAIR_SYSTEM_PROMPT, ALT_MODEL_OPENAI,
        "openai", PAIR_FIELDS, batch_size=15, cache=cache,
        cache_prefix="val_pair",
    )

    # -- Gemini (user turns) --
    print(f"\n  === {ALT_MODEL_GEMINI} ===")
    print("  User turns...")
    gemini_user_items = [{"id": r["turn_id"], "text": r["text_norm"]}
                         for _, r in user_sample.iterrows()]
    gemini_user = annotate_batch(
        gemini_user_items, USER_TURN_SYSTEM_PROMPT, ALT_MODEL_GEMINI,
        "gemini", USER_FIELDS, batch_size=15, cache=cache,
        cache_prefix="val_user",
    )

    print("  Bot turns...")
    gemini_bot_items = [{"id": r["turn_id"], "text": r["text_norm"]}
                        for _, r in bot_sample.iterrows()]
    gemini_bot = annotate_batch(
        gemini_bot_items, BOT_TURN_SYSTEM_PROMPT, ALT_MODEL_GEMINI,
        "gemini", BOT_FIELDS, batch_size=15, cache=cache,
        cache_prefix="val_bot",
    )

    print("  Pairs...")
    gemini_pair_items = []
    for _, r in pair_df.iterrows():
        gemini_pair_items.append({
            "id": r["pair_id"],
            "user_text": r["user_text"],
            "user_emotion_present": int(r["user_emotion_present"]),
            "user_vuln_level": int(r["user_vuln_level"]),
            "bot_text": r["bot_text"],
        })
    gemini_pair = annotate_batch(
        gemini_pair_items, PAIR_SYSTEM_PROMPT, ALT_MODEL_GEMINI,
        "gemini", PAIR_FIELDS, batch_size=15, cache=cache,
        cache_prefix="val_pair",
    )

    # ── Compute IAA ─────────────────────────────────────────────────
    print("\n[5/6] Computing IAA metrics...")

    model_names = [ORIGINAL_MODEL, ALT_MODEL_OPENAI, ALT_MODEL_GEMINI]

    all_results = {
        "user_turns": [],
        "bot_turns": [],
        "pairs": [],
    }

    # --- User turn fields ---
    user_ids = [r["turn_id"] for _, r in user_sample.iterrows()]
    user_annotations = {
        ORIGINAL_MODEL: orig_user,
        ALT_MODEL_OPENAI: nano_user,
        ALT_MODEL_GEMINI: gemini_user,
    }

    print("\n  User turn fields:")
    for field in USER_FIELDS:
        is_ord = field in ORDINAL_USER
        result = compute_iaa_for_field(
            user_annotations, field, is_ord, user_ids, model_names
        )
        all_results["user_turns"].append(result)
        ktype = "QW-kappa" if is_ord else "kappa"
        three_way = result.get("krippendorff_alpha_ordinal") or result.get("fleiss_kappa")
        three_way_label = "Kripp-alpha" if is_ord else "Fleiss-kappa"
        print(f"    {field:25s}  mean {ktype}={result['mean_pairwise_kappa']:.3f}  "
              f"{three_way_label}={three_way:.3f}" if three_way is not None else
              f"    {field:25s}  mean {ktype}={result['mean_pairwise_kappa']:.3f}  "
              f"{three_way_label}=N/A")

    # --- Bot turn fields ---
    bot_ids = [r["turn_id"] for _, r in bot_sample.iterrows()]
    bot_annotations = {
        ORIGINAL_MODEL: orig_bot,
        ALT_MODEL_OPENAI: nano_bot,
        ALT_MODEL_GEMINI: gemini_bot,
    }

    print("\n  Bot turn fields:")
    for field in BOT_FIELDS:
        is_ord = field in ORDINAL_BOT
        result = compute_iaa_for_field(
            bot_annotations, field, is_ord, bot_ids, model_names
        )
        all_results["bot_turns"].append(result)
        ktype = "QW-kappa" if is_ord else "kappa"
        three_way = result.get("krippendorff_alpha_ordinal") or result.get("fleiss_kappa")
        three_way_label = "Kripp-alpha" if is_ord else "Fleiss-kappa"
        print(f"    {field:25s}  mean {ktype}={result['mean_pairwise_kappa']:.3f}  "
              f"{three_way_label}={three_way:.3f}" if three_way is not None else
              f"    {field:25s}  mean {ktype}={result['mean_pairwise_kappa']:.3f}  "
              f"{three_way_label}=N/A")

    # --- Pair fields ---
    pair_ids = pair_df["pair_id"].tolist()
    pair_annotations = {
        ORIGINAL_MODEL: orig_pair,
        ALT_MODEL_OPENAI: nano_pair,
        ALT_MODEL_GEMINI: gemini_pair,
    }

    print("\n  Pair fields:")
    for field in PAIR_FIELDS:
        result = compute_iaa_for_field(
            pair_annotations, field, False, pair_ids, model_names
        )
        all_results["pairs"].append(result)
        three_way = result.get("fleiss_kappa")
        print(f"    {field:25s}  mean kappa={result['mean_pairwise_kappa']:.3f}  "
              f"Fleiss-kappa={three_way:.3f}" if three_way is not None else
              f"    {field:25s}  mean kappa={result['mean_pairwise_kappa']:.3f}  "
              f"Fleiss-kappa=N/A")

    # ── Aggregate summaries ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("AGGREGATE SUMMARY")
    print("=" * 60)

    # Binary fields
    binary_kappas = []
    binary_fleiss = []
    binary_agree = []
    for section in ["user_turns", "bot_turns", "pairs"]:
        for r in all_results[section]:
            if r["type"] == "binary":
                if not np.isnan(r["mean_pairwise_kappa"]):
                    binary_kappas.append(r["mean_pairwise_kappa"])
                fk = r.get("fleiss_kappa")
                if fk is not None and not np.isnan(fk):
                    binary_fleiss.append(fk)
                if not np.isnan(r["mean_agreement"]):
                    binary_agree.append(r["mean_agreement"])

    print(f"\nBinary annotations ({len(binary_kappas)} fields):")
    if binary_kappas:
        print(f"  Mean pairwise Cohen's kappa: {np.mean(binary_kappas):.3f} "
              f"(range: {np.min(binary_kappas):.3f} -- {np.max(binary_kappas):.3f})")
    if binary_fleiss:
        print(f"  Mean Fleiss' kappa:          {np.mean(binary_fleiss):.3f} "
              f"(range: {np.min(binary_fleiss):.3f} -- {np.max(binary_fleiss):.3f})")
    if binary_agree:
        print(f"  Mean percent agreement:      {np.mean(binary_agree):.3f}")

    # Ordinal fields
    ordinal_kappas = []
    ordinal_kripp = []
    for section in ["user_turns", "bot_turns"]:
        for r in all_results[section]:
            if r["type"] == "ordinal":
                if not np.isnan(r["mean_pairwise_kappa"]):
                    ordinal_kappas.append(r["mean_pairwise_kappa"])
                ka = r.get("krippendorff_alpha_ordinal")
                if ka is not None and not np.isnan(ka):
                    ordinal_kripp.append(ka)

    print(f"\nOrdinal annotations ({len(ordinal_kappas)} fields):")
    if ordinal_kappas:
        print(f"  Mean pairwise QW-kappa:       {np.mean(ordinal_kappas):.3f} "
              f"(range: {np.min(ordinal_kappas):.3f} -- {np.max(ordinal_kappas):.3f})")
    if ordinal_kripp:
        print(f"  Mean Krippendorff's alpha:    {np.mean(ordinal_kripp):.3f} "
              f"(range: {np.min(ordinal_kripp):.3f} -- {np.max(ordinal_kripp):.3f})")

    # ── Save results ────────────────────────────────────────────────
    print("\n[6/6] Saving results...")

    # Compute overall summary dict
    summary = {
        "sample_sizes": {
            "user_turns": len(user_sample),
            "bot_turns": len(bot_sample),
            "pairs": len(pair_df),
        },
        "models": model_names,
        "binary_summary": {
            "n_fields": len(binary_kappas),
            "mean_pairwise_kappa": round(float(np.mean(binary_kappas)), 3) if binary_kappas else None,
            "min_pairwise_kappa": round(float(np.min(binary_kappas)), 3) if binary_kappas else None,
            "max_pairwise_kappa": round(float(np.max(binary_kappas)), 3) if binary_kappas else None,
            "mean_fleiss_kappa": round(float(np.mean(binary_fleiss)), 3) if binary_fleiss else None,
            "mean_agreement": round(float(np.mean(binary_agree)), 3) if binary_agree else None,
        },
        "ordinal_summary": {
            "n_fields": len(ordinal_kappas),
            "mean_qw_kappa": round(float(np.mean(ordinal_kappas)), 3) if ordinal_kappas else None,
            "min_qw_kappa": round(float(np.min(ordinal_kappas)), 3) if ordinal_kappas else None,
            "max_qw_kappa": round(float(np.max(ordinal_kappas)), 3) if ordinal_kappas else None,
            "mean_krippendorff_alpha": round(float(np.mean(ordinal_kripp)), 3) if ordinal_kripp else None,
        },
        "per_field_results": all_results,
    }

    # Convert numpy types to Python native for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return round(float(obj), 4)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    summary = make_serializable(summary)

    out_path = os.path.join(OUTPUT_DIR, "iaa_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")

    # Also save raw annotations for reproducibility
    raw_path = os.path.join(OUTPUT_DIR, "raw_annotations.json")
    raw = {
        "user_turns": {
            "item_ids": user_ids,
            "annotations": make_serializable({
                ORIGINAL_MODEL: orig_user,
                ALT_MODEL_OPENAI: nano_user,
                ALT_MODEL_GEMINI: gemini_user,
            }),
        },
        "bot_turns": {
            "item_ids": bot_ids,
            "annotations": make_serializable({
                ORIGINAL_MODEL: orig_bot,
                ALT_MODEL_OPENAI: nano_bot,
                ALT_MODEL_GEMINI: gemini_bot,
            }),
        },
        "pairs": {
            "item_ids": pair_ids,
            "annotations": make_serializable({
                ORIGINAL_MODEL: orig_pair,
                ALT_MODEL_OPENAI: nano_pair,
                ALT_MODEL_GEMINI: gemini_pair,
            }),
        },
    }
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {raw_path}")

    cache.close()
    print("\nDone!")

    return summary


if __name__ == "__main__":
    main()
