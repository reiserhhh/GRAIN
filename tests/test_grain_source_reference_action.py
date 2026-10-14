#!/usr/bin/env python3
"""Designed null/alternative worlds for reference response, not human evidence."""

import numpy as np
import pytest

from grain_research.construction import math_source_reference_action as M
from grain_research.construction.math_relational_diagnostic_update import BITS


def potentials():
    """Positive fully supported two-observation profile law."""
    return np.random.default_rng(215).normal(size=(3, 2, 32)) / 4


def test_pure_calibration_and_common_tilt():
    """Profile offsets and common observation tilts change scores but not defects."""
    s = potentials()
    d = s + np.linspace(-0.2, 0.3, 32) + np.array([0.7, -0.1])[None, :, None]
    result = M.certificate(s, d)
    np.testing.assert_allclose(result["response_defect"], 0, atol=1e-14)
    np.testing.assert_allclose(result["conditional_cross_ratio"], 0, atol=1e-14)
    assert result["control_profile_error"] < 1e-14


def test_target_response_changes():
    """A target-specific tilt must remain after matching the full control profile."""
    s, d = potentials(), potentials()
    d[:, 0] += 0.4 * (2 * BITS[:, 0] - 1)
    r = M.certificate(s, d)
    np.testing.assert_allclose(r["conditional_cross_ratio"][:, 0], 0.8, atol=1e-14)
    np.testing.assert_allclose(r["response_defect"][:, 0], 0.8, atol=1e-14)


def test_other_profile_routing_is_not_zero_marginal():
    """No local q tilt does not imply an invariant marginal when targets associate."""
    s = np.zeros((1, 2, 32))
    s[:, 0] = 2 * (2 * BITS[:, 0] - 1) * (2 * BITS[:, 1] - 1)
    d = s.copy()
    d[:, 0] += 0.5 * (2 * BITS[:, 1] - 1)
    r = M.certificate(s, d)
    np.testing.assert_allclose(r["conditional_cross_ratio"][:, 0], 0, atol=0)
    assert abs(r["response_defect"][0, 0]) > 0.1


def test_literal_and_signed_identity():
    """Probability ratios agree with log arithmetic, including all signed accounts."""
    s = potentials()
    d = s + np.random.default_rng(217).normal(size=s.shape) / 2
    r, a = M.certificate(s, d), M.literal_certificate(np.exp(s), np.exp(d))
    for k, v in a.items():
        np.testing.assert_allclose(r[k], v, atol=1e-14)
    np.testing.assert_allclose(r["defect_identity"], r["response_defect"], atol=1e-14)
    np.testing.assert_allclose(
        r["destination_action"] - r["source_action"],
        r["calibration_action"] + r["response_defect"],
        atol=1e-14,
    )


def test_gauge_and_reverse_reference():
    """Point normalizers cannot manufacture conditional response drift."""
    s, d = potentials(), -potentials()
    a = M.certificate(s, d)
    b = M.certificate(s + np.array([0.5, 0.1])[None, :, None], d - 0.6)
    for k in ("conditional_cross_ratio", "response_defect", "profile_relative_eta"):
        np.testing.assert_allclose(a[k], b[k], atol=1e-14)
    np.testing.assert_allclose(
        a["conditional_cross_ratio"],
        -M.certificate(d, s)["conditional_cross_ratio"],
        atol=1e-14,
    )


@pytest.mark.parametrize("bad", [np.zeros((2, 31)), np.full((2, 32), np.nan)])
def test_missing_support_rejected(bad):
    """Invalid support cannot be represented as a successful zero action."""
    with pytest.raises(ValueError):
        M.certificate(bad, bad)
