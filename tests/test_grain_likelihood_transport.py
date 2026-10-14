#!/usr/bin/env python3
"""Finite positive-world tests for retained source-profile update geometry."""

import numpy as np
import pytest

from grain_research.construction import math_likelihood_transport as M


def test_identity():
    """Repeated observation without an endpoint change has no new update."""
    lp = np.linspace(-1, 1, 32)[None]
    r = M.transport(lp, lp)
    for key in ("delta", "joint_energy", "total_energy", "character_delta"):
        np.testing.assert_allclose(r[key], 0, atol=1e-15)


def test_additive_target_tilt():
    """An additive log tilt is completely visible in the target tangent space."""
    a = np.linspace(-0.8, 0.4, 32)[None]
    b = a + M.BITS @ np.array([0.4, -0.2, 0.1, 0.3, -0.15])
    r = M.transport(a, b)
    np.testing.assert_allclose(r["path_joint"], 0, atol=2e-15)
    np.testing.assert_allclose(r["main_end"], r["probability_end"], atol=2e-15)
    np.testing.assert_allclose(r["integrated_slope"], r["delta"], atol=2e-15)


def test_pure_joint_configuration():
    """Joint profile changes can exist while every single target mean stays fixed."""
    a = np.zeros((1, 32))
    b = a + 0.4 * M.CHARACTERS[:, 3]
    r = M.transport(a, b)
    np.testing.assert_allclose(r["delta"], 0, atol=3e-16)
    assert r["joint_energy"][0] > 0
    assert r["character_delta"][0, 0] > 0
    np.testing.assert_allclose(r["main_energy"], 0, atol=1e-29)


def test_weighted_target_recoding():
    """Invertible affine recoding preserves the statistic span and score split."""
    p = np.exp(M.normalize(np.linspace(-1, 0.5, 32)))
    ell = M.CHARACTERS[:, 3] + M.BITS @ np.arange(5) / 10
    a = M.split_score(p, ell)
    recode = M.BITS @ (np.eye(5) + 0.1 * np.ones((5, 5))) + np.arange(5)
    b = M.split_score(p, ell, recode)
    for key in ("main", "joint", "energy", "main_energy", "joint_energy"):
        np.testing.assert_allclose(a[key], b[key], atol=2e-14)


def test_reference_and_finite_response():
    """Locally invisible scores need not have zero finite marginal response."""
    a = (0.6 * M.BITS[:, 0] + 0.4 * M.CHARACTERS[:, 3])[None]
    b = a + 0.7 * M.CHARACTERS[:, 3] + 0.2 * M.BITS[:, 1]
    r = M.transport(a, b)
    assert np.max(np.abs(r["joint_delta"])) > 1e-5
    np.testing.assert_allclose(
        r["delta"],
        r["main_delta"] + r["joint_delta"] + r["interaction_delta"],
        atol=2e-16,
    )
    np.testing.assert_allclose(r["total_energy"], r["jeffreys"], atol=2e-15)


def test_direction_and_reconstruction():
    """Reverse endpoints reverses target change, not its nonnegative energy."""
    a = np.zeros((1, 32))
    b = (0.5 * M.BITS[:, 0] + 0.3 * M.CHARACTERS[:, 7])[None]
    r, s = M.transport(a, b), M.transport(b, a)
    np.testing.assert_allclose(r["delta"], -s["delta"], atol=2e-16)
    np.testing.assert_allclose(r["total_energy"], s["total_energy"], atol=2e-15)
    np.testing.assert_allclose(
        r["character_delta"],
        r["character_main_delta"] + r["character_joint_delta"],
        atol=2e-15,
    )


@pytest.mark.parametrize(
    "bad", [np.zeros(31), np.full(32, np.nan), np.full(32, -np.inf)]
)
def test_invalid_support(bad):
    """Unavailable profile states are not clipped into a usable measure."""
    with pytest.raises(ValueError):
        M.normalize(bad)


def test_cold_contract():
    """Endpoint transport accepts provenance, but no labels or model answers."""
    record = {
        "evidence_key": "u/window",
        "source_contract": "source-only-law",
        "log_start": [0] * 32,
        "log_end": M.BITS[:, 0].tolist(),
    }
    result = M.cold(record)
    assert result["target_delta"][0] > 0
    assert len(result["joint_character_masks"]) == 26
    np.testing.assert_allclose(
        np.array(result["target_means_end"]) - result["target_means_start"],
        result["target_delta"],
        atol=2e-16,
    )
    for key in ("profile_probabilities_start", "profile_probabilities_end"):
        np.testing.assert_allclose(sum(result[key]), 1, atol=2e-16)
    with pytest.raises(ValueError):
        M.cold({**record, "answer": "high"})
