#!/usr/bin/env python3
"""Synthetic checks for event qualification and a three-view moment compiler."""

import numpy as np
import pytest

from grain_research.construction import math_natural_proxy as M


def test_choice_and_total_vector_mass_are_retained():
    z = np.arange(72, dtype=float).reshape(2, 12, 3)
    c = np.tile([0, 1, 0], (2, 4))
    x = M.view_features(z, c, np.tile(np.arange(12), (2, 1)))
    assert x.shape == (2, 3, 8)
    for v in range(3):
        np.testing.assert_allclose(x[:, v, 2:5] + x[:, v, 5:], z[:, v::3].mean(1))
        np.testing.assert_allclose(x[:, v, 1], c[:, v::3].mean(1))


def test_event_copy_is_rejected():
    with pytest.raises(ValueError, match="distinct"):
        M.view_features(np.ones((2, 12, 2)), np.zeros((2, 12)), np.zeros((2, 12)))


def test_whole_author_fold_no_overlap():
    train, held = M.author_fold([0, 1, 0, 1], 0)
    assert not set(train) & set(held)
    assert sorted([*train, *held]) == list(range(4))
    with pytest.raises(ValueError):
        M.author_fold([0, 0, 0], 0)


def test_independent_proxy_world_recovers_coefficient_not_attenuation():
    # Complete weighted Cartesian design gives exactly independent proxy errors.
    u, e0, e1, e2 = np.meshgrid([-1, 1], [-1, 1], [-1, 1], [-1, 1], indexing="ij")
    vectors = np.stack([u + e0, u + e1, u + e2], axis=-1).reshape(-1, 3)
    x = np.stack([np.ones_like(vectors), vectors], axis=-1)
    y = (0.4 + 0.2 * u.reshape(-1))[:, None]
    program = M.construct(x, y, 0, 1)
    np.testing.assert_allclose(program["bridge"], [[0.4], [0.2]], atol=1e-14)
    np.testing.assert_allclose(program["ordinary"], [[0.4], [0.1]], atol=1e-14)
    error = M.moment_error(x, y, program["bridge"], 0, 2, program["source_rms"])
    assert max(error["normalized_rms"]) < 1e-14


def test_unused_view_can_detect_an_invalid_proxy_law():
    rng = np.random.default_rng(7)
    x = np.concatenate([np.ones((90, 3, 1)), rng.normal(size=(90, 3, 2))], axis=-1)
    y = (x[:, 2, 1] > 0).astype(float)[:, None]
    program = M.construct(x, y, 0, 1)
    error = M.moment_error(x, y, program["bridge"], 0, 2, program["source_rms"])
    assert error["normalized_rms"][0] > 0.1


def test_profile_permutation_preserves_joint_labels_per_fold():
    y = np.arange(24).reshape(8, 3)
    f = np.arange(8) % 2
    shuffled = M.permute_profiles(y, f, np.random.default_rng(253))
    for fold in (0, 1):
        assert sorted(map(tuple, y[f == fold])) == sorted(
            map(tuple, shuffled[f == fold])
        )


def test_stranger_matching_keeps_condition_and_length_cells():
    counts = np.array([0, 1, 4, 7, 12])
    lengths = np.array([10, 10, 30, 30, 30])
    donor = M.matched_donors(counts, lengths, 20)
    np.testing.assert_array_equal(donor, [1, 0, 3, 2, -1])


def test_auc_and_correlation_keep_direction():
    assert M.auc([0, 0, 1, 1], [0, 1, 2, 3]) == 1
    assert M.auc([0, 1], [2, 2]) == 0.5
    assert M.correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)
    assert M.correlation([1, 1], [2, 3]) is None
