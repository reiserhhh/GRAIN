#!/usr/bin/env python3
"""Finite algebra and failure-capable synthetic worlds, not personality validity."""

import numpy as np
import pytest

from grain_research.construction import math_joint_target_atlas as M


def test_complete_character_basis():
    h = M.characters()
    np.testing.assert_array_equal(h.T @ h, 32 * np.eye(32))


def world():
    """Make all profiles with four authors each and two separate text views."""
    codes = np.repeat(np.arange(32), 4)
    folds = M.author_folds(codes, np.arange(len(codes)), "synthetic")
    h = M.characters()
    coefficients = np.zeros((32, 3))
    coefficients[0] = (5, 4, 3)
    coefficients[1, 0] = 2
    coefficients[3, 1] = -4
    x = np.repeat((h @ coefficients)[codes, None, :], 2, axis=1)
    return codes, folds, x, coefficients


def test_affine_and_joint_world_reconstruction():
    c, folds, x, expected = world()
    f = M.construct(x, c, folds)["fields"]
    np.testing.assert_allclose(f, np.broadcast_to(expected, f.shape), atol=1e-13)
    expected[3] = 0
    affine = np.repeat((M.characters() @ expected)[c, None, :], 2, axis=1)
    f = M.construct(affine, c, folds)["fields"]
    np.testing.assert_allclose(f[:, :, M.orders() >= 2], 0, atol=1e-13)


def test_same_shared_mode_enters_two_conditional_target_families():
    e = M.edge_matrix()
    assert np.all(np.abs(e[0, :, 3]) == 2)
    assert np.all(np.abs(e[1, :, 3]) == 2)
    assert np.all(e[2:, :, 3] == 0)
    h = M.characters()
    for q in range(5):
        others = [k for k in range(5) if k != q]
        for b in range(16):
            lo = sum(((b >> j) & 1) << k for j, k in enumerate(others))
            np.testing.assert_array_equal(e[q, b], h[lo | (1 << q)] - h[lo])


def test_partition_disjoint_complete_and_deterministic():
    c = np.repeat(np.arange(32), 3)
    f = M.author_folds(c, np.arange(len(c)), "salt")
    np.testing.assert_array_equal(f, M.author_folds(c, np.arange(len(c)), "salt"))
    for fold in (0, 1):
        assert np.bincount(c[f == fold], minlength=32).min() > 0
    assert np.isin(f, (0, 1)).all()


def test_missing_profile_side_not_imputed():
    with pytest.raises(ValueError, match="Missing profile"):
        M.profile_weights(np.arange(31))
    with pytest.raises(ValueError, match="Both author folds"):
        M.author_folds(np.arange(32), np.arange(32), "salt")


def test_joint_permutation_keeps_fold_joint_counts():
    c, f, _, _ = world()
    permuted = M.permute_profiles(c, f, np.random.default_rng(185))
    for fold in (0, 1):
        np.testing.assert_array_equal(
            np.bincount(c[f == fold]), np.bincount(permuted[f == fold])
        )
    assert not np.array_equal(c, permuted)


def test_cross_author_negative_not_clipped():
    f = np.zeros((2, 2, 32, 3, 4))
    f[0, :, 3, :, 0], f[1, :, 3, :, 0] = 2, -2
    joint = [r for r in M.cross_author_statistics(f) if r["order"] == "joint"]
    assert all(r["signed_inner_product"] == -4 for r in joint)
    np.testing.assert_allclose(
        [r["cross_family_coherence"] for r in joint], -1, atol=1e-14, rtol=0
    )


def frame():
    """Use only numeric evidence and opaque addresses at application time."""
    return {
        "frame_key": "opaque",
        "events": [{"event_id": "one", "slot": 0, "vector": [1.0] * 1024}],
    }


def test_cold_complete_and_empty():
    f, s = np.ones((2, 32, 3, 1024)), np.ones((2, 3, 1024))
    result = M.cold_record(f, s, frame())
    assert np.asarray(result["modes"]).shape == (2, 32, 3)
    assert np.asarray(result["edges"]).shape == (2, 5, 16, 3)
    assert result["semantic"] == [[1024.0] * 3] * 2
    result = M.cold_record(f, s, {"frame_key": "empty", "events": []})
    assert result["modes"] is None and result["edges"] is None


def test_cold_labels_and_duplicates_rejected():
    f, s = np.zeros((2, 32, 3, 1024)), np.zeros((2, 3, 1024))
    with pytest.raises(ValueError, match="Only numeric"):
        M.cold_record(f, s, frame() | {"labels": [0] * 5})
    r = frame()
    r["events"] *= 2
    with pytest.raises(ValueError, match="Repeated"):
        M.cold_record(f, s, r)
