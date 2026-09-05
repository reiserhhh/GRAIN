"""Raw-comment history/current construction without personality-label selection."""

from __future__ import annotations

import hashlib
import re


def stable_key(value: str, seed: str) -> str:
    """Deterministic selection key, independent of outcomes and Python hash state."""
    return hashlib.sha256(f"{seed}::{value}".encode()).hexdigest()


def normalized_text_key(text: str) -> str:
    """Detect exact normalized-text duplicates without changing model inputs."""
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


def leakage_flags(text: str) -> dict[str, bool]:
    """Flag possible report clues; do not equate a trait word with actual leakage."""
    return {
        "mbti_type_term": bool(re.search(r"\b[EI][NS][FT][JP]\b", text, re.IGNORECASE)),
        "big5_term": bool(
            re.search(
                r"\b(big\s*(five|5)|ocean|openness|conscientiousness|extraversion|agreeableness|neuroticism)\b",
                text,
                re.IGNORECASE,
            )
        ),
        "report_clue": bool(
            re.search(
                r"\b(personality\s+test|percentile|my\s+(test\s+)?results|scored\s+\d+)\b",
                text,
                re.IGNORECASE,
            )
        ),
    }


def select_context_bundle(
    comments: list[dict], *, history_count=3, current_count=2
) -> dict | None:
    """Choose recent currents and earlier, thread-disjoint history from a reservoir."""
    if not comments:
        return None
    if any(not c.get("author_id") for c in comments):
        raise ValueError("Every comment requires an author ID")
    if len({c["author_id"] for c in comments}) != 1:
        raise ValueError("Cannot mix authors inside a context bundle")
    unique = {}
    for c in sorted(comments, key=lambda item: (item["timestamp"], item["comment_id"])):
        unique.setdefault(normalized_text_key(c["body"]), c)
    ordered = list(unique.values())
    current, seen_threads = [], set()
    for comment in reversed(ordered):
        if comment["thread_id"] and comment["thread_id"] not in seen_threads:
            current.append(comment)
            seen_threads.add(comment["thread_id"])
        if len(current) == current_count:
            break
    if len(current) < current_count:
        return None
    earliest = min(c["timestamp"] for c in current)
    eligible = [
        c
        for c in ordered
        if c["timestamp"] < earliest
        and c["thread_id"]
        and c["thread_id"] not in seen_threads
    ]
    # Keep one history per discussion, rather than three replicas of one thread.
    history, threads = [], set()
    for c in reversed(eligible):
        if c["thread_id"] not in threads:
            history.append(c)
            threads.add(c["thread_id"])
        if len(history) == history_count:
            break
    if len(history) != history_count:
        return None
    bundle = {
        "author_id": comments[0]["author_id"],
        "history": sorted(history, key=lambda c: c["timestamp"]),
        "current": sorted(current, key=lambda c: c["timestamp"]),
    }
    validate_context_bundle(bundle)
    return bundle


def validate_context_bundle(bundle: dict) -> None:
    """Reject same-fragment, same-thread, author, and temporal contamination."""
    history, current = bundle["history"], bundle["current"]
    if not history or not current:
        raise ValueError("Both history and current views are required")
    if any(c["author_id"] != bundle["author_id"] for c in history + current):
        raise ValueError("Author mismatch")
    for field in ["comment_id", "thread_id"]:
        if {c[field] for c in history} & {c[field] for c in current}:
            raise ValueError(f"History/current overlap: {field}")
    if {normalized_text_key(c["body"]) for c in history} & {
        normalized_text_key(c["body"]) for c in current
    }:
        raise ValueError("History repeats current text")
    if max(c["timestamp"] for c in history) >= min(c["timestamp"] for c in current):
        raise ValueError("History must strictly precede both current fragments")
