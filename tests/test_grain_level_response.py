#!/usr/bin/env python3
"""Synthetic checks for reference anchoring, local charts and exact accounting."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.level_response import (
    affine_parts,
    atlas_readout,
    paired_quality,
    response_ledger,
)


def test_affine_value_response_and_anchor_sign():
    rng = np.random.default_rng(4)
    z, ze, zr, m = rng.normal(size=(4, 3, 5))
    h, he, hr, g = rng.normal(size=(4, 5))
    f = lambda a, b: 0.2 + np.sum(m * a) + g @ b
    full = affine_parts(z, h, zr, hr, m, g)
    edit = affine_parts(ze, he, zr, hr, m, g)
    assert f(zr, hr) + full["PQ"] == pytest.approx(f(z, h))
    assert edit["PQ"] - full["PQ"] == pytest.approx(f(ze, he) - f(z, h))
    assert -full["PQ"] == pytest.approx(f(zr, hr) - f(z, h))


def test_ledger_does_not_count_interaction_twice():
    r = response_ledger(0.3, 0.1, 0.7, 0.4, 0.5, 0.9)
    explicit = (
        r["rebase"]
        + r["E_main"]
        + r["P_main"]
        + r["EP_interaction"]
        + r["Q_conditional"]
    )
    allocated = r["rebase"] + r["E_allocated"] + r["P_allocated"] + r["Q_conditional"]
    assert explicit == pytest.approx(r["natural"])
    assert allocated == pytest.approx(r["natural"])


def test_atlas_chart_selection_has_no_outcome_dependency():
    zs = np.array([np.zeros((2, 3)), np.ones((2, 3))])
    hs, gs = np.zeros((2, 3)), np.ones((2, 3))
    first = atlas_readout(zs[1], hs[1], zs, hs, zs, gs, [0, 1])
    second = atlas_readout(zs[1], hs[1], zs, hs, zs, gs, [400, -30])
    assert first["chart"] == second["chart"] == 1
    assert first["score"] == 1


def test_quality_covariance_identity_and_perfect_gain():
    y = np.array([-2, 0, 1, 3.0])
    q = paired_quality(y, y, 0, draws=50)
    assert q["rmse"] == 0
    assert q["mse_ratio"] == 0
    assert q["mse_gain_ci"][0] > 0
    q = paired_quality(y, y * 0.4 + 1, 0, draws=50)
    assert q["var_predicted"] + q["var_remainder"] + 2 * q[
        "cov_predicted_remainder"
    ] == pytest.approx(q["var_actual"])
    assert q["cov_predicted_actual"] + q["cov_remainder_actual"] == pytest.approx(
        q["var_actual"]
    )


def test_invalid_shapes_and_zero_target_energy():
    with pytest.raises(ValueError):
        affine_parts(
            np.zeros((2, 3)),
            np.zeros(4),
            np.zeros((2, 3)),
            np.zeros(3),
            np.zeros((2, 3)),
            np.zeros(3),
        )
    q = paired_quality(np.zeros(4), np.ones(4), 0, draws=20)
    assert q["mse_ratio"] is None and q["r"] is None
