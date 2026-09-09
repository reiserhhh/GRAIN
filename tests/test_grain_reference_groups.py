#!/usr/bin/env python3
"""Author-reference and group-distance invariants on synthetic examples."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.functional_profiles import FixedAuthorCoordinates
from grain_research.vector_mechanisms.reference_groups import (
    distance_relation,
    group_permutation,
    separation_statistic,
)


def test_fixed_reference_does_not_move_when_batch_changes():
    ref = np.arange(36).reshape(12, 3)
    norm = FixedAuthorCoordinates().fit(ref)
    x = np.array([[1.0, 2.0, 3.0], [8.0, 7.0, 6.0]])
    expected = norm.transform(x)
    augmented = norm.transform(np.vstack([x, np.full((10, 3), 1000000)]))[:2]
    np.testing.assert_array_equal(expected, augmented)
    np.testing.assert_array_equal(expected[0], norm.transform(x[:1])[0])


def test_group_separation_and_author_permutation():
    response = np.array(
        [[0, 0], [0.1, 0.1], [0.2, 0.2], [3, 3], [3.1, 3.1], [3.2, 3.2]]
    )
    labels = [0, 0, 0, 1, 1, 1]
    result = group_permutation(response, labels, draws=99, seed=8)
    assert result["separation_gain"] > 8
    assert result["within_pairs"] == 6 and result["between_pairs"] == 9
    assert 0 < result["permutation_p_greater"] <= 1
    assert result == group_permutation(response, labels, draws=99, seed=8)
    with pytest.raises(ValueError):
        separation_statistic(response, [0] * 6)


def test_distance_relation_and_dimensions():
    x = np.arange(18).reshape(6, 3)
    result = distance_relation(x, x, draws=29)
    assert result["distance_spearman"] == pytest.approx(1)
    with pytest.raises(ValueError):
        distance_relation(x, x[:3])
