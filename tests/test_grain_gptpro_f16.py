#!/usr/bin/env python3
"""F16 construction checks."""

from grain_research.gptpro_foundation_v3.f13 import gaussian_interval


def test_interval_is_ordered():
    lo, hi = gaussian_interval(.2, .05)
    assert lo < hi


def test_target_briefs_are_distinct():
    targets = {"T1": "condition", "T2": "state", "T3": "future"}
    assert len(set(targets.values())) == 3
