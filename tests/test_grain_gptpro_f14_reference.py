#!/usr/bin/env python3
"""Source-reference reconstruction must not depend on held numeric values."""

import copy

import numpy as np
import pytest

from grain_research.conditional_reference import fit_reference
from scripts.audit_grain_gptpro_f14_reference import (
    compare_reference,
    rebuild_reference,
)


def test_source_reference_author_balance_and_held_exclusion():
    authors = np.array([0, 0, 0, 1, 1, 2])
    c = np.array(["a", "a", "b", "a", "b", "held_only"])
    x = np.arange(12, dtype=float).reshape(6, 2)
    source = np.array([0, 1])
    original = fit_reference(c, x, authors, source)
    rebuilt = rebuild_reference(c, x, authors, source)
    assert max(compare_reference(rebuilt, original).values()) < 1e-12
    assert rebuilt["weights"][0] == pytest.approx((2 / 3 + 1 / 2) / 2)
    x[-1] = 1e100
    assert (
        max(
            compare_reference(
                rebuild_reference(c, x, authors, source), original
            ).values()
        )
        < 1e-12
    )
    corrupted = copy.deepcopy(original)
    corrupted["mean"][0, 0] += 0.1
    with pytest.raises(ValueError, match="reference mismatch"):
        compare_reference(rebuilt, corrupted)


def test_missing_source_is_not_silently_dropped():
    with pytest.raises(ValueError, match="missing source"):
        rebuild_reference(["a"], [[1.0]], [0], [0, 1])


def test_changed_condition_identity_is_rejected():
    ref = rebuild_reference(["a", "b"], [[1.0], [2.0]], [0, 1], [0, 1])
    other = copy.deepcopy(ref)
    other["keys"] = ["b", "a"]
    with pytest.raises(ValueError, match="condition identity"):
        compare_reference(ref, other)
