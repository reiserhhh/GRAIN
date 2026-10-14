#!/usr/bin/env python3
"""Operation-level checks for the finite HM130 measurement-object diagnostic."""

import numpy as np
import pytest

from grain_research.construction.math_observer_order import (
    JOINT_MINUS,
    JOINT_PLUS,
    affine_certificate,
    contrast_class,
    evidence_mean,
    fixed_margin_contrast,
    pair_orders,
)


def test_positive_affinity_preserves_event_and_mixture_order():
    f = np.array([-2.0, 0, 1, 4])
    n = 2.5 * f + 7
    assert affine_certificate(f, n)["positive_affine_on_support"]
    assert all(r["classification"] == "concordant" for r in pair_orders(f, n))
    p, q = np.array([0.1, 0.2, 0.3, 0.4]), np.ones(4) / 4
    assert np.isclose(
        evidence_mean(n, p) - evidence_mean(n, q),
        2.5 * (evidence_mean(f, p) - evidence_mean(f, q)),
    )


def test_increasing_event_map_does_not_preserve_author_ties():
    f, n = np.array([0.0, 1, 2]), np.array([0.0, 1, 4])
    assert all(r["classification"] == "concordant" for r in pair_orders(f, n))
    assert not affine_certificate(f, n)["positive_affine_on_support"]
    p, q = [0.4, 0.2, 0.4], [0.1, 0.8, 0.1]
    assert np.isclose(evidence_mean(f, p), evidence_mean(f, q))
    assert np.isclose(evidence_mean(n, p) - evidence_mean(n, q), 0.6)


def test_negative_affinity_is_not_same_directed_measurement():
    f = np.arange(4.0)
    c = affine_certificate(f, -3 * f + 2)
    assert c["relative_residual"] < 1e-14
    assert not c["positive_affine_on_support"]
    assert all(r["classification"] == "reversed" for r in pair_orders(f, -f))


def test_fixed_margins_and_joint_character_formula():
    for p in (JOINT_PLUS, JOINT_MINUS):
        assert np.allclose(p.reshape(2, 2).sum(0), [0.5, 0.5])
        assert np.allclose(p.reshape(2, 2).sum(1), [0.5, 0.5])
    f = np.array([3.0, 2, -1, 4])
    assert np.isclose(fixed_margin_contrast(f), (f[0] - f[1] - f[2] + f[3]) / 4)


def test_constants_keep_undefined_units_and_all_ties():
    c = affine_certificate(np.ones(4), np.full(4, 9.0))
    assert c["positive_affine_on_support"] and c["slope"] is None
    assert c["relative_residual"] is None
    assert not affine_certificate(np.ones(4), np.arange(4.0))[
        "positive_affine_on_support"
    ]
    assert contrast_class(0, 1) == "one_sided_tie"
    assert contrast_class(0, 0) == "both_tied"


def test_validation_rejects_missing_mismatched_and_invalid_laws():
    for f, n in (([1], [2]), ([1, 2], [1]), ([1, np.nan], [1, 2])):
        with pytest.raises(ValueError):
            pair_orders(f, n)
    for p in ([0.2, 0.2], [-0.2, 1.2], [np.nan, 0.5]):
        with pytest.raises(ValueError):
            evidence_mean([1, 2], p)
    with pytest.raises(ValueError):
        fixed_margin_contrast([1, 2, 3])


def test_tolerance_is_numerical_not_a_selected_psychological_threshold():
    assert contrast_class(1e-11, 1, 1e-10) == "one_sided_tie"
    assert contrast_class(1e-11, -1, 0) == "reversed"
    for tol in (-1, np.nan, np.inf):
        with pytest.raises(ValueError):
            pair_orders([1, 2], [2, 3], tol)
