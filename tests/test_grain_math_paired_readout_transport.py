"""Synthetic mathematical checks of HM21 without any original text or model."""

import numpy as np
import pytest

from grain_research.construction import math_paired_readout_transport as P


def test_hessian_gradient_finite_difference():
    """Check the analytic Hessian and exact common-shift null direction."""
    z = np.array([0.2, -0.6, 1.4, 0.9])
    a = np.linspace(0, 1, len(z))
    _, g, h = P.gradient_hessian(z, a)
    eps = 1e-5
    numeric = np.column_stack(
        [
            (
                P.gradient_hessian(z + eps * e, a)[1]
                - P.gradient_hessian(z - eps * e, a)[1]
            )
            / (2 * eps)
            for e in np.eye(len(z))
        ]
    )
    np.testing.assert_allclose(h, numeric, atol=1e-10)
    np.testing.assert_allclose(h, h.T, atol=1e-15)
    np.testing.assert_allclose(h @ np.ones(len(z)), 0, atol=1e-15)
    assert abs(g.sum()) < 1e-15


def test_path_completeness_and_curvature():
    """Compare both integral identities and a deliberately nonlinear segment."""
    a, r, t = np.array([0.0, 1.0]), np.array([0.0, -2.0]), np.array([0.0, 3.0])
    v = P.paired_action(r, t, a)
    assert abs(v["curvature"]) > 0.3
    assert v["native"] == pytest.approx(v["gradient_integral"], abs=1e-12)
    assert v["curvature"] == pytest.approx(v["hessian_integral"], abs=1e-12)
    assert v["digit_path_action"].sum() == pytest.approx(v["native"], abs=1e-12)
    reverse = P.paired_action(t, r, a)
    assert reverse["native"] == pytest.approx(-v["native"], abs=1e-15)
    assert reverse["tangent"] != pytest.approx(-v["tangent"])


def test_gauge_alphabet_units_and_noop():
    """Units transform with the fixed alphabet; no-op is not a missing value."""
    r, t, a = (
        np.array([0.1, 0.8, -1.0]),
        np.array([0.6, -0.4, 0.2]),
        np.array([0, 0.5, 1]),
    )
    v = P.paired_action(r, t, a)
    gauge, scaled = P.paired_action(r + 11, t - 7, a), P.paired_action(r, t, 3 * a + 5)
    for k in (
        "native",
        "tangent",
        "curvature",
        "gradient_integral",
        "hessian_integral",
    ):
        assert gauge[k] == pytest.approx(v[k], abs=1e-14)
        assert scaled[k] == pytest.approx(3 * v[k], abs=1e-14)
        assert P.paired_action(r, r, a)[k] == 0
    assert P.paired_action(r, t, np.ones(3))["native"] == pytest.approx(0, abs=1e-15)


def test_saturated_distribution():
    """Stable softmax handles a nearly deterministic readout without clipping."""
    v = P.paired_action([1000, 0, -1000], [1001, 0, -1000], [0, 0.5, 1])
    assert v["native"] == v["tangent"] == v["curvature"] == 0


def test_signed_joint_recombination():
    """Cross terms can cancel and must not be converted to unsigned percentages."""
    la, lb = np.array([1, 2, 3.0]), np.array([-2, 3, 1.0])
    ya, yb = la - 2 * lb, lb + 3 * la
    v = P.covariance_parts(ya, yb, la, lb, np.full(3, 1 / 3))
    assert v["recombination_error"] < 1e-13
    assert v["tangent_a_curvature_b"] > 0
    assert v["curvature_a_tangent_b"] < 0
    z = P.covariance_parts(np.zeros(3), np.zeros(3), la, lb, np.full(3, 1 / 3))
    assert z["link_native_ratio"] is None


@pytest.mark.parametrize(
    "r,t,a,order",
    [
        ([0, np.nan], [0, 1], [0, 1], 32),
        ([0, 1], [0, 1, 2], [0, 1], 32),
        ([0, 1], [0, 1], [0, np.inf], 32),
        ([0, 1], [0, 1], [0, 1], 1),
    ],
)
def test_invalid_inputs(r, t, a, order):
    """Reject incompatible shapes/nonfinite evidence rather than manufacture scores."""
    with pytest.raises(ValueError):
        P.paired_action(r, t, a, order)
