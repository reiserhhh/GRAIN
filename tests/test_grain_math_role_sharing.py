#!/usr/bin/env python3
"""Sharing is a source-parameter identity, not a free reader/domain conversion."""

import numpy as np
import pytest

from grain_research.construction import math_role_sharing as S


def fixture(seed, dim=5):
    rng = np.random.default_rng(seed)
    return {
        "w": rng.normal(size=(dim, 4)),
        "mu": rng.normal(size=dim),
        "bias": rng.normal(size=4),
    }


def test_compiled_roles_keep_source_intercept_and_exclude_relevance():
    model = fixture(1)
    x = np.random.default_rng(2).normal(size=(20, 5))
    observed = S.score_roles(S.affine_roles(model), x)
    np.testing.assert_allclose(
        observed, ((x - model["mu"]) @ model["w"] + model["bias"])[:, 1:]
    )


def test_common_and_difference_reconstruct_each_source_program():
    a, b = fixture(3), fixture(4)
    common, difference = S.split_common_roles(a, b)
    x = np.random.default_rng(5).normal(size=(20, 5))
    c, d = S.score_roles(common, x), S.score_roles(difference, x)
    np.testing.assert_allclose(c + d, S.score_roles(S.affine_roles(a), x))
    np.testing.assert_allclose(c - d, S.score_roles(S.affine_roles(b), x))


def test_shared_map_is_symmetric_but_target_remainder_changes_sign():
    a, b = fixture(6), fixture(7)
    c, d = S.split_common_roles(a, b)
    swapped, reverse = S.split_common_roles(b, a)
    np.testing.assert_array_equal(c["w"], swapped["w"])
    np.testing.assert_array_equal(d["w"], -reverse["w"])


def test_mismatched_reader_coordinates_are_not_silently_shared():
    with pytest.raises(ValueError):
        S.split_common_roles(fixture(8, 5), fixture(9, 6))


def test_common_role_map_does_not_commute_with_joint_qualification():
    a = {"w": np.zeros((1, 4)), "mu": np.zeros(1), "bias": np.array([1, 1, 1, 1])}
    b = {**a, "bias": np.array([1, 1, -1, -1])}
    common, _ = S.split_common_roles(a, b)
    p, o, actuality = S.score_roles(common, np.zeros((1, 1)))[0]
    shared_unit = p * (1 + o) * (1 + actuality) / 4
    assert shared_unit == 0.25
    assert shared_unit != (1.0 + 0.0) / 2
