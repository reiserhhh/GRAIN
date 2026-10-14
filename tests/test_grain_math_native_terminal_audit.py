#!/usr/bin/env python3
"""Nulls for terminal-only diagnostics; incomplete cases cannot become zeros."""

import numpy as np
import pytest

from grain_research.construction.math_joint_reading import BASIS
from grain_research.construction.math_native_terminal_audit import (
    character_table,
    complete_groups,
    cosine,
    effect_summary,
    matrix_sectors,
)


def test_incomplete_corner_target_or_arm_is_excluded():
    records = {
        (g, a, c, q): 1
        for g in range(2)
        for a in ("b", "d", "r")
        for c in range(4)
        for q in range(3)
    }
    del records[1, "r", 3, 2]
    assert complete_groups(records, range(2), 3, ("b", "d", "r")) == [0]
    assert complete_groups(records, range(2), 3, ("b",)) == [0, 1]


def test_all_characters_and_positive_author_law():
    x = np.random.default_rng(3).normal(size=(4, 6, 2))
    chars = character_table(x)
    plus, minus = (1 + 0.5 * BASIS[3]) / 4, (1 - 0.5 * BASIS[3]) / 4
    assert np.allclose(np.tensordot(plus - minus, x, axes=1), chars[3])
    with pytest.raises(ValueError):
        character_table(np.full((4, 6), np.nan))


def test_generic_changes_cannot_masquerade_as_coupled_target_changes():
    x = 4 + np.arange(6)[:, None] + np.arange(6)[None, :] * 2
    parts = matrix_sectors(x)
    assert np.max(abs(parts["coupled"])) < 1e-12
    assert np.allclose(sum(parts.values()), x)
    y = x + np.outer(np.arange(6) - 2.5, np.arange(6) - 2.5)
    coupled = matrix_sectors(y)["coupled"]
    assert np.allclose(coupled.sum(0), 0)
    assert np.allclose(coupled.sum(1), 0)
    assert np.linalg.norm(coupled) > 0


def test_no_native_bridge_and_directed_bridge_are_distinguishable():
    full = np.array([1.0, -1.0, 2.0, -2.0])
    no_bridge = effect_summary(full, np.zeros(4), np.zeros(4), full)
    assert no_bridge["native_local_cosine"] == pytest.approx(1)
    assert no_bridge["removed_effect"]["rms"] == 0
    actual = effect_summary(full, full / 2, np.array([2.0, 2.0, -1.0, -1.0]))
    assert actual["removed_alignment"] == pytest.approx(0.5)
    assert actual["random_alignment"] == pytest.approx(0)


def test_zero_native_and_missing_tables_do_not_gain_a_score():
    assert cosine(np.zeros(4), np.ones(4)) is None
    assert (
        effect_summary(np.zeros(4), np.ones(4), np.ones(4))["removed_alignment"] is None
    )
    with pytest.raises(ValueError):
        matrix_sectors(np.full((6, 6), np.nan))
