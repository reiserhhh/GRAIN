"""Calibration direction, missing anchors and nonlinear support safeguards."""

import numpy as np

from grain_research.gptpro_foundation_v3.f12_transport import (
    fit_transport,
    transport_score,
)


def pilot():
    r = np.tile(np.linspace(-2, 2, 30), 2)
    z = np.repeat([-1.0, 1.0], 30)
    return r, z


def test_negative_affine_scale_preserved():
    r, z = pilot()
    s = 1 - 1.5 * r
    model = fit_transport(r, s, z, "global")
    p, support = transport_score(model, s, z)
    assert support.all()
    np.testing.assert_allclose(p, r, atol=1e-12)


def test_condition_dependent_inverse_exact():
    r, z = pilot()
    s = z + (1 + 0.25 * z) * r
    model = fit_transport(r, s, z, "conditional")
    p, support = transport_score(model, s, z)
    assert support.all()
    np.testing.assert_allclose(p, r, atol=1e-12)


def test_missing_condition_rank_and_no_silent_score():
    r = np.linspace(-2, 2, 60)
    z = np.ones(60)
    model = fit_transport(r, r, z, "conditional")
    assert model["status"] == "MISSING_ANCHOR"
    p, support = transport_score(model, r, z)
    assert not support.any() and np.isnan(p).all()


def test_cubic_nonmonotonicity_rejected():
    s, z = pilot()
    old = s - 0.5 * s**3
    model = fit_transport(old, s, z, "cubic")
    assert model["status"] == "NONMONOTONE_INVERSE"


def test_cubic_range_and_condition_support():
    s, z = pilot()
    model = fit_transport(s + 0.1 * s**3, s, z, "cubic")
    p, support = transport_score(model, [0, 3, 0], [1, 1, 0])
    np.testing.assert_array_equal(support, [True, False, False])
    assert abs(p[0]) < 1e-12 and np.isnan(p[1:]).all()
