#!/usr/bin/env python3
"""Lexical near-copy audit; flags overlap without pretending to detect all paraphrase."""

import re


def shingles(text):
    """Return lowercased word four-grams and length; this is only an audit view."""
    words = re.findall(r"\b\w+\b", text.lower())
    return {tuple(words[i : i + 4]) for i in range(max(0, len(words) - 3))}, len(words)


def copy_overlap(a, b):
    """Flag substantial near copies or contained passages using frozen thresholds."""
    x, n = a
    y, m = b
    shared = len(x & y)
    jaccard = shared / max(1, len(x | y))
    containment = shared / max(1, min(len(x), len(y)))
    flagged = min(n, m) >= 12 and shared >= 8 and (jaccard >= 0.8 or containment >= 0.9)
    return {
        "shared_4grams": shared,
        "jaccard": jaccard,
        "containment": containment,
        "flagged": flagged,
    }
