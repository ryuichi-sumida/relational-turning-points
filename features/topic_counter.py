#!/usr/bin/env python3
"""
FASTopic on intella-transcript-v2.0 TSV chat transcripts.

Input TSV format (tab-separated) with columns:
start_time  end_time  topic  speaker  text
(and optional conversation_id if you have multiple conversations in one file)

What this script does:
1) Loads TSV (ignores lines starting with '#')
2) Cleans obvious filler / empty rows
3) Builds "documents" by aggregating turn windows (recommended for chat)
4) Fits FASTopic and prints:
   - a topic timeline per window (start–end + top words)
   - overall top topics for the conversation (or per conversation_id)

Example:
  python fastopic_chat.py --input chat.tsv --num-topics 12 --window-size 6 --stride 3

If you have multiple conversations in one TSV and a column conversation_id:
  python fastopic_chat.py --input chats.tsv --conversation-col conversation_id
"""

import argparse
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from fastopic import FASTopic
from topmost.preprocess import Preprocess


DEFAULT_FILLERS = {
    "uh", "um", "yeah", "yea", "yep", "nope", "ok", "okay", "mm", "hmm", "erm",
    "ah", "oh", "right", "sure"
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to TSV transcript.")
    p.add_argument("--sep", default="\t", help="Field separator (default: tab).")
    p.add_argument("--conversation-col", default=None,
                   help="Column name for conversation id (optional). If omitted, entire file treated as one conversation.")
    p.add_argument("--min-window-tokens", type=int, default=15,
                   help="Drop windows with fewer than this many whitespace tokens.")
    p.add_argument("--drop-fillers", action="store_true",
                   help="Drop rows that are just filler tokens / very short.")
    p.add_argument("--include-speaker", action="store_true",
                   help="Prefix each turn with 'speaker: text' when building docs.")
    p.add_argument("--window-size", type=int, default=6, help="Turns per document window.")
    p.add_argument("--stride", type=int, default=3, help="Turn stride between windows.")
    p.add_argument("--use-only-speakers", default=None,
                   help="Comma-separated list of speakers to include (e.g. 'user,system'). Default: include all.")
    p.add_argument("--num-topics", type=int, default=10, help="Number of topics K.")
    p.add_argument("--vocab-size", type=int, default=5000, help="Max vocabulary size for preprocessing.")
    p.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    p.add_argument("--learning-rate", type=float, default=0.005, help="Learning rate.")
    p.add_argument("--device", default="cpu", help="Device for FASTopic: 'cpu' or e.g. 'cuda'.")
    p.add_argument("--normalize-embeddings", action="store_true",
                   help="Enable FASTopic normalize_embeddings=True (can help repetitive topics).")
    p.add_argument("--dt-alpha", type=float, default=None,
                   help="Set DT_alpha (e.g., 10.0) if topics are repetitive / loss stagnates.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    p.add_argument("--save-model", default=None, help="Optional path to save model (e.g., ./fastopic.zip).")
    return p.parse_args()


def is_filler_or_too_short(text: str, fillers: set) -> bool:
    s = (text or "").strip().lower()
    if not s:
        return True
    if len(s) < 3:
        return True
    # if it's basically a single filler token, drop
    if s in fillers:
        return True
    # common "just a fill."-type artifacts
    if s.replace(".", "").strip() in fillers:
        return True
    return False


def load_and_clean_tsv(
    path: str,
    sep: str,
    drop_fillers: bool,
    allowed_speakers: Optional[set],
) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=sep, comment="#", dtype=str)
    except Exception as e:
        raise RuntimeError(f"Failed to read TSV: {e}")

    needed = {"start_time", "end_time", "speaker", "text"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}. Found columns: {list(df.columns)}")

    df["text"] = df["text"].astype(str).fillna("")
    df["speaker"] = df["speaker"].astype(str).fillna("")

    # Filter speakers if requested
    if allowed_speakers is not None:
        df = df[df["speaker"].str.lower().isin({s.lower() for s in allowed_speakers})].copy()

    # Drop empty lines
    df = df[df["text"].str.strip().ne("")].copy()

    if drop_fillers:
        df = df[~df["text"].apply(lambda t: is_filler_or_too_short(t, DEFAULT_FILLERS))].copy()

    # Keep stable ordering if already in order; else you could sort by start_time if needed.
    df.reset_index(drop=True, inplace=True)
    return df


def build_turn_windows(
    turns: pd.DataFrame,
    window_size: int,
    stride: int,
    include_speaker: bool,
    min_window_tokens: int,
) -> Tuple[List[str], List[Dict]]:
    docs: List[str] = []
    meta: List[Dict] = []

    turns = turns.reset_index(drop=True)
    n = len(turns)

    for start_idx in range(0, n, stride):
        end_idx = start_idx + window_size
        chunk = turns.iloc[start_idx:end_idx]
        if chunk.empty:
            continue

        if include_speaker:
            doc = " ".join([f"{r.speaker}: {r.text}" for r in chunk.itertuples(index=False)])
        else:
            doc = " ".join(chunk["text"].tolist())

        if len(doc.split()) < min_window_tokens:
            continue

        docs.append(doc)
        meta.append({
            "start_time": chunk["start_time"].iloc[0],
            "end_time": chunk["end_time"].iloc[-1],
            "n_turns": int(len(chunk)),
            "row_start": int(start_idx),
            "row_end": int(min(end_idx, n)),
        })

    return docs, meta


def topic_words(model: FASTopic, topic_idx: int, n: int = 8) -> str:
    pairs = model.get_topic(int(topic_idx))[:n]
    return ", ".join([w for (w, _p) in pairs])


def run_fastopic_on_docs(
    docs: List[str],
    num_topics: int,
    vocab_size: int,
    epochs: int,
    learning_rate: float,
    device: str,
    normalize_embeddings: bool,
    dt_alpha: Optional[float],
    seed: int,
) -> Tuple[FASTopic, List, np.ndarray]:
    # Reproducibility
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
    except Exception:
        pass
    np.random.seed(seed)

    preprocess = Preprocess(vocab_size=vocab_size)

    kwargs = {
        "device": device,
        "normalize_embeddings": normalize_embeddings,
    }
    if dt_alpha is not None:
        kwargs["DT_alpha"] = float(dt_alpha)

    model = FASTopic(num_topics, preprocess, **kwargs)

    top_words, doc_topic_dist = model.fit_transform(
        docs,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    return model, top_words, doc_topic_dist


def print_conversation_report(
    conv_name: str,
    meta: List[Dict],
    model: FASTopic,
    doc_topic_dist: np.ndarray,
    overall_top_n: int = 5,
) -> None:
    if doc_topic_dist.size == 0:
        print(f"\n=== {conv_name} ===")
        print("No usable windows (docs) after filtering. Try lowering --min-window-tokens or increasing window size.")
        return

    top_topic_per_window = np.argmax(doc_topic_dist, axis=1)

    print(f"\n=== {conv_name} ===")
    print(f"Windows: {len(meta)} | Topics: {doc_topic_dist.shape[1]}")
    print("\nTopic timeline (per window):")
    for m, t in zip(meta, top_topic_per_window):
        words = topic_words(model, int(t), n=8)
        print(f"  {m['start_time']}–{m['end_time']}  topic {int(t)}: {words}")

    conv_topic_weights = doc_topic_dist.mean(axis=0)
    top_topics = conv_topic_weights.argsort()[::-1][:overall_top_n]

    print("\nOverall top topics:")
    for t in top_topics:
        words = topic_words(model, int(t), n=10)
        print(f"  topic {int(t)} weight={conv_topic_weights[t]:.3f}: {words}")


def main() -> int:
    args = parse_args()

    allowed_speakers = None
    if args.use_only_speakers:
        allowed_speakers = {s.strip() for s in args.use_only_speakers.split(",") if s.strip()}

    df = load_and_clean_tsv(
        path=args.input,
        sep=args.sep,
        drop_fillers=args.drop_fillers,
        allowed_speakers=allowed_speakers,
    )

    # Group by conversation id if provided; else treat as one conversation
    if args.conversation_col and args.conversation_col in df.columns:
        grouped = list(df.groupby(args.conversation_col, sort=False))
    else:
        grouped = [("conversation_1", df)]

    # Build docs for all conversations together (better training signal),
    # then we can report per conversation using the same model.
    all_docs: List[str] = []
    all_meta: List[Tuple[str, List[Dict]]] = []
    doc_ranges: List[Tuple[str, int, int]] = []  # (conv_name, start_doc_idx, end_doc_idx)

    cursor = 0
    for conv_id, g in grouped:
        docs, meta = build_turn_windows(
            turns=g,
            window_size=args.window_size,
            stride=args.stride,
            include_speaker=args.include_speaker,
            min_window_tokens=args.min_window_tokens,
        )
        start = cursor
        all_docs.extend(docs)
        cursor += len(docs)
        end = cursor
        doc_ranges.append((str(conv_id), start, end))
        all_meta.append((str(conv_id), meta))

    if not all_docs:
        print("No usable documents/windows produced. Try:")
        print("  - increasing --window-size")
        print("  - lowering --min-window-tokens")
        print("  - removing --drop-fillers (if used)")
        return 2

    model, _top_words, doc_topic_dist = run_fastopic_on_docs(
        docs=all_docs,
        num_topics=args.num_topics,
        vocab_size=args.vocab_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        normalize_embeddings=args.normalize_embeddings,
        dt_alpha=args.dt_alpha,
        seed=args.seed,
    )

    # Report per conversation using the slice of doc_topic_dist
    meta_by_conv = {name: meta for name, meta in all_meta}
    for conv_name, s, e in doc_ranges:
        print_conversation_report(
            conv_name=conv_name,
            meta=meta_by_conv[conv_name],
            model=model,
            doc_topic_dist=doc_topic_dist[s:e],
            overall_top_n=5,
        )

    if args.save_model:
        model.save(args.save_model)
        print(f"\nSaved model to: {args.save_model}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
