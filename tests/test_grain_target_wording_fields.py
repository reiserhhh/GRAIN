#!/usr/bin/env python3
"""Synthetic discrimination controls, not evidence of human trait validity."""

import numpy as np
import pytest

from grain_research.construction.math_target_wording_fields import (
    ORBIT,
    field_gram,
    gram_accounts,
)


def accounts(vectors):
    """Create an ordinary Euclidean Gram independently of vocabulary reduction."""
    x = np.asarray(vectors, float)
    return gram_accounts((x @ x.T)[None])


def test_target_only_has_cross_wording_support():
    a = accounts([[1, 0]] * 3 + [[-1, 0]] * 3)
    assert a["target_energy"][0] == pytest.approx(1)
    assert a["wording_energy"][0] == pytest.approx(0)
    assert a["interaction_energy"][0] == pytest.approx(0)
    assert a["intended_support"][0] == pytest.approx(2)
    assert a["strict_first"][0]
    np.testing.assert_allclose(a["cross_family_cosines"], 1)


def test_family_only_cannot_manufacture_target_support():
    x = [[1, 0], [0, 2], [-1, 0]]
    a = accounts(x + x)
    assert a["wording_energy"][0] > 0
    assert a["target_energy"][0] == pytest.approx(0)
    np.testing.assert_allclose(a["orbit_scores"], 0, atol=1e-14)
    assert a["tied_first"][0]
    assert np.isnan(a["mean_cross_family_cosine"][0])


def test_interaction_cancellation_keeps_negative_support():
    x = [[1], [-1], [0]]
    a = accounts(x + [[-v[0]] for v in x])
    assert a["target_energy"][0] == pytest.approx(0)
    assert a["interaction_energy"][0] == pytest.approx(2 / 3)
    assert a["intended_support"][0] == pytest.approx(-2 / 3)
    assert a["not_first"][0]
    assert np.isnan(a["mean_cross_family_cosine"][0])


def test_energy_and_orbit_identities_with_all_components():
    x = np.random.default_rng(5).normal(size=(6, 11))
    a = accounts(x)
    for key in ("energy_closure_error", "orbit_mean_error", "support_identity_error"):
        np.testing.assert_allclose(a[key], 0, atol=1e-14)
    for key in (
        "common_energy",
        "target_energy",
        "wording_energy",
        "interaction_energy",
    ):
        assert a[key][0] >= 0
    d = (x[:3] - x[3:]) / 2
    brute = [
        (np.sum(d * e[:, None], axis=0) ** 2).sum() / 3 - (d**2).sum() / 3
        for e in ORBIT
    ]
    np.testing.assert_allclose(a["orbit_scores"][0], brute, atol=1e-14)


def test_global_label_flip_is_same_partition():
    x = np.random.default_rng(7).normal(size=(6, 5))
    first, flipped = accounts(x), accounts(np.r_[x[3:], x[:3]])
    np.testing.assert_allclose(first["orbit_scores"], flipped["orbit_scores"])


def test_common_reference_gauge_preserves_full_offsets():
    rng = np.random.default_rng(9)
    u, z, a = (
        rng.normal(size=(6, 2, 7)),
        rng.normal(size=(2, 7)),
        rng.normal(size=(6, 2)),
    )
    shifts = rng.normal(size=(6, 2))
    first = field_gram(u, z, a, [1, 3])
    second = field_gram(u + shifts[:, :, None], z + 11, a + shifts, [1, 3])
    for key in ("gram", "selected_shape", "calibration", "reconstructed_k"):
        np.testing.assert_allclose(first[key], second[key], atol=1e-13)
    np.testing.assert_allclose(
        first["selected_shape"] + first["calibration"], first["reconstructed_k"]
    )


def test_common_q_gram_matches_direct_weighted_inner_products():
    rng = np.random.default_rng(10)
    u = rng.normal(size=(6, 2, 8))
    z = np.log(np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3]] * 2))
    result = field_gram(u, z, np.zeros((6, 2)), [0, 1])
    q = np.exp(z)
    for t in range(2):
        centered = u[:, t] - u[:, t] @ q[t, :, None]
        np.testing.assert_allclose(result["gram"][t], (centered * q[t]) @ centered.T)


def test_orthogonal_coordinates_preserve_accounts():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(6, 4))
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    a, b = accounts(x), accounts(x @ rotation)
    for key in ("target_energy", "interaction_energy", "orbit_scores"):
        np.testing.assert_allclose(a[key], b[key], atol=1e-13)


def test_invalid_or_indefinite_observations_are_not_repaired():
    with pytest.raises(ValueError):
        gram_accounts(np.full((1, 6, 6), np.nan))
    with pytest.raises(ValueError):
        gram_accounts((-np.eye(6))[None])
    with pytest.raises(ValueError):
        field_gram(np.zeros((6, 2, 5)), np.zeros((2, 5)), np.zeros((6, 2)), [5, 0])
    with pytest.raises(ArithmeticError):
        field_gram(np.zeros((6, 1, 2)), [[0, -1000]], np.zeros((6, 1)), [0])
