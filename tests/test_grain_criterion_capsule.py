#!/usr/bin/env python3
"""Synthetic capsule algebra tests without raw participant data or models."""

import numpy as np
import pytest

from grain_research.construction import math_criterion_capsule as C


def test_paired_capsule_preserves_baseline():
    rng = np.random.default_rng(223)
    b = rng.normal(size=(2, 3, 6, 2, 4, 5))
    i = rng.normal(size=b.shape)
    np.testing.assert_allclose(C.capsule(b + i, b), i[0] - i[1], atol=1e-15)
    with pytest.raises(ValueError):
        C.capsule(b[0], b[0])


def test_maps_are_not_event_replicates():
    a = np.random.default_rng(23).normal(size=(3, 6, 2, 4, 5))
    np.testing.assert_array_equal(
        C.condition_values(C.condition_values(a, "swapped"), "swapped"), a
    )
    common = C.condition_values(a, "common")
    np.testing.assert_array_equal(C.condition_values(common, "common"), common)
    np.testing.assert_allclose(a + C.condition_values(a, "swapped"), 2 * common)
    assert not np.shares_memory(a, C.condition_values(a, "aligned"))


def test_suffix_uses_relative_identical_tokens():
    rows = [[9] * q + [1, 2, 3, 8] for q in range(1, 7)]
    suffix = C.suffix_support(rows, 3)
    assert [[r[i] for i in a] for r, a in zip(rows, suffix, strict=True)] == [
        [1, 2, 3]
    ] * 6
    rows[0][-2] = 99
    with pytest.raises(ValueError):
        C.suffix_support(rows, 3)


def test_unobserved_and_wrong_contracts_rejected():
    with pytest.raises(ValueError):
        C.condition_values(np.zeros((3, 5, 2, 4, 5)), "aligned")
    with pytest.raises(ValueError):
        C.condition_values(np.full((3, 6, 2, 4, 5), np.nan), "aligned")
    with pytest.raises(ValueError):
        C.condition_values(np.zeros((3, 6, 2, 4, 5)), "optimized")
