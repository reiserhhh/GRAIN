"""Synthetic-only invariants for response-context material and arithmetic."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.response_context import (
    context_derangements,
    cosine,
    packed_sequence,
    paired_interval,
)


def test_assignments_are_complete_fixed_point_free_and_reproducible():
    a, b = context_derangements([20, 21, 70, 71, 150, 151])
    x, y = context_derangements([20, 21, 70, 71, 150, 151])
    for z in (a, b):
        assert sorted(z) == list(range(6))
        assert not (z == np.arange(6)).any()
    assert np.array_equal(a, x) and np.array_equal(b, y)
    assert set(zip(range(6), a)) == {(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4)}


def test_current_tokens_positions_and_valid_sequence_preserved():
    prefix, body = [12, 13, 14], [99, 98, 97]
    padded, mask, indices = packed_sequence(prefix, body, 0)
    compact, cmask, ci = packed_sequence(prefix, body, 0, compact=True)
    assert len(padded) == len(compact) == 1024 and indices[0] == 768
    assert [padded[i] for i in indices] == [compact[i] for i in ci] == body
    assert np.array_equal(
        np.asarray(padded)[np.asarray(mask, bool)],
        np.asarray(compact)[np.asarray(cmask, bool)],
    )
    _, _, other = packed_sequence([44] * 90, body, 0)
    assert indices == other
    with pytest.raises(ValueError):
        packed_sequence([1] * 769, body, 0)


def test_fixed_material_interval_and_cosine():
    r = paired_interval([0.1] * 12)
    assert r["lower"] == pytest.approx(0.1) and r["upper"] == pytest.approx(0.1)
    assert r["positive_count"] == 12
    assert cosine([1, 0], [1, 0]) == 1
    assert cosine([1, 0], [0, 1]) == 0
    with pytest.raises(ValueError):
        paired_interval([np.nan])
