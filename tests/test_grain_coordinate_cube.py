#!/usr/bin/env python3
"""Finite mathematical witnesses for the M24 intervention construction."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.coordinate_cube import (
    BITS,
    anchored_effects,
    query_cube,
)


def test_exact_endpoints_and_zero_mean_configuration():
    """The cube adds existing displacement components, not imagined opposite traits."""
    rng = np.random.default_rng(24)
    p, pe = rng.normal(size=(2, 5, 4))
    q, qe = rng.normal(size=(2, 4))
    cube = query_cube(p, q, pe, qe)
    assert np.allclose(cube["000"][0], p)
    assert np.allclose(cube["111"][0], pe)
    assert np.allclose(cube["111"][1], qe)
    assert np.allclose((cube["010"][0] - p).mean(0), 0)
    assert np.allclose(cube["100"][0] - p + cube["010"][0] - p, pe - p)


def test_cube_recovers_polynomial_terms():
    """All interaction orders are recovered exactly, without an orthogonality claim."""
    values = {
        "".join(map(str, (a, b, c))): 5
        + a
        + 2 * b
        + 3 * c
        + 4 * a * b
        + 5 * a * c
        + 6 * b * c
        + 7 * a * b * c
        for a, b, c in BITS
    }
    effects = anchored_effects(values)
    assert effects == {
        "mean": 1,
        "configuration": 2,
        "final": 3,
        "mean_configuration": 4,
        "mean_final": 5,
        "configuration_final": 6,
        "triple": 7,
    }


def test_affine_reader_interactions_zero():
    """Any fixed linear feature map has zero interaction on this additive state cube."""
    rng = np.random.default_rng(8)
    p, pe = rng.normal(size=(2, 5, 4))
    q, qe = rng.normal(size=(2, 4))
    wp, wq = rng.normal(size=(5, 4)), rng.normal(size=4)
    values = {
        k: float(np.sum(a * wp) + b @ wq + 3)
        for k, (a, b) in query_cube(p, q, pe, qe).items()
    }
    effects = anchored_effects(values)
    assert all(abs(effects[k]) < 1e-12 for k in effects if "_" in k or k == "triple")
    with pytest.raises(ValueError):
        anchored_effects({"000": 0})


def test_bad_state_rejected():
    """Missing dimensions or nonfinite displacements cannot silently enter a cube."""
    with pytest.raises(ValueError):
        query_cube(np.ones((3, 4)), np.ones(4), np.ones((2, 4)), np.ones(4))
