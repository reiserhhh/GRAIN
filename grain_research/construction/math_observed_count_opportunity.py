#!/usr/bin/env python3
"""Signed observation accounts for occurrence counts and word opportunities."""

from __future__ import annotations

import numpy as np


def count_opportunity_account(high_counts, low_counts, high_words, low_words):
    """Decompose a rate contrast without deleting count or denominator changes."""
    high, low = np.asarray(high_counts, float), np.asarray(low_counts, float)
    wh, wl = np.asarray(high_words, float), np.asarray(low_words, float)
    if high.shape != low.shape or wh.shape != wl.shape:
        raise ValueError("Different high/low observation shapes")
    if not all(np.isfinite(x).all() for x in (high, low, wh, wl)):
        raise ValueError("Nonfinite observation")
    if np.any(high < 0) or np.any(low < 0) or np.any(wh <= 0) or np.any(wl <= 0):
        raise ValueError("Nonnegative counts and positive opportunities required")
    difference = 100 * (high / wh - low / wl)
    occurrence = 50 * (high - low) * (1 / wh + 1 / wl)
    opportunity = 50 * (high + low) * (1 / wh - 1 / wl)
    return {
        "rate_difference": difference,
        "occurrence_term": occurrence,
        "opportunity_term": opportunity,
        "raw_count_difference": high - low,
        "reconstruction_error": difference - occurrence - opportunity,
    }
