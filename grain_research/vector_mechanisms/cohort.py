"""Label-blind, multi-comment author views for natural mechanism discovery."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .data_views import normalized_text_key, stable_key

ID_SEED = "grain_vm02_raw_contexts_20260905"
COHORT_SEED = "grain_vm02_multicomment_20260906"


def select_multicomment_bundle(comments, *, units=12, forbidden_texts=()):
    """Select three chronological, thread-disjoint views without outcome access.

    A hash sample precedes chronological allocation. Multiple observations from
    the same discussion and normalized identical bodies are never counted as
    independent evidence. The three periods need not have equal duration.
    """
    if units < 1:
        raise ValueError("units must be positive")
    if not comments:
        return None
    if len({c["author_id"] for c in comments}) != 1:
        raise ValueError("Cannot mix authors")
    chosen, threads, texts = [], set(), set(forbidden_texts)
    for c in sorted(comments, key=lambda c: stable_key(c["comment_id"], COHORT_SEED)):
        key = normalized_text_key(c["body"])
        if not c["thread_id"] or c["thread_id"] in threads or key in texts:
            continue
        threads.add(c["thread_id"])
        texts.add(key)
        chosen.append(c)
        if len(chosen) == 3 * units:
            break
    if len(chosen) != 3 * units:
        return None
    chosen.sort(key=lambda c: (c["timestamp"], c["comment_id"]))
    result = {
        "author_id": chosen[0]["author_id"],
        "views": [chosen[i * units : (i + 1) * units] for i in range(3)],
    }
    validate_multicomment_bundle(result, units=units)
    return result


def validate_multicomment_bundle(bundle, *, units):
    """Fail on author, unit, text, discussion, or temporal contamination."""
    if len(bundle["views"]) != 3 or any(len(v) != units for v in bundle["views"]):
        raise ValueError("Expected three equal-size views")
    rows = [c for v in bundle["views"] for c in v]
    if any(c["author_id"] != bundle["author_id"] for c in rows):
        raise ValueError("Author mismatch")
    for field in ("comment_id", "thread_id"):
        if any(not c[field] for c in rows) or len({c[field] for c in rows}) != len(
            rows
        ):
            raise ValueError(f"Missing or duplicated {field}")
    if len({normalized_text_key(c["body"]) for c in rows}) != len(rows):
        raise ValueError("Repeated text")
    times = [c["timestamp"] for c in rows]
    if not np.isfinite(times).all() or times != sorted(times):
        raise ValueError("Nonfinite or unordered times")
    if any(
        bundle["views"][i][-1]["timestamp"] >= bundle["views"][i + 1][0]["timestamp"]
        for i in (0, 1)
    ):
        raise ValueError("Periods require strict temporal separation")


def summarize_bundles(bundles):
    """Aggregate coverage without publishing raw author IDs, quotations or labels."""
    comments = [c for b in bundles for v in b["views"] for c in v]

    def quantiles(values):
        return dict(
            zip(("min", "median", "max"), map(float, np.quantile(values, [0, 0.5, 1])))
        )

    spans = [
        (b["views"][-1][-1]["timestamp"] - b["views"][0][0]["timestamp"]) / 86400
        for b in bundles
    ]
    return {
        "authors": len(bundles),
        "comments": len(comments),
        "split_authors": dict(Counter(b["split"] for b in bundles)),
        "words_per_comment": quantiles([c["word_count"] for c in comments]),
        "days_covered_per_author": quantiles(spans),
        "unique_subreddits": len({c["subreddit"] for c in comments}),
        "subreddits_per_author": quantiles(
            [len({c["subreddit"] for v in b["views"] for c in v}) for b in bundles]
        ),
        "flagged_comments": {
            flag: sum(c["leakage_flags"][flag] for c in comments)
            for flag in ("mbti_type_term", "big5_term", "report_clue")
        },
    }
