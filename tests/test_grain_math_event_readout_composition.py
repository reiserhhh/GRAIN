#!/usr/bin/env python3
"""Synthetic worlds that support and violate shared score-level event mixtures."""

import numpy as np
import pytest

from grain_research.construction.math_event_readout_composition import (
    segment_certificate,
    update_decomposition,
    vector,
)


@pytest.mark.parametrize("t", [0, 0.2, 0.5, 0.9, 1])
def test_exact_shared_mixture(t):
    """A genuine common-weight world must not fail this certificate."""
    x, y = np.array([0.8, 0.1, 0.6]), np.array([0.2, 0.9, 0.3])
    result = segment_certificate(t * x + (1 - t) * y, x, y)
    assert result["segment_distance"] < 1e-14
    assert result["dual_identity_error"] < 1e-14
    assert result["weight_shared_positive"] == pytest.approx(t)


def test_query_specific_weights_inside_box_but_not_common_segment():
    """Different queries can individually be plausible without any shared weight."""
    result = segment_certificate([0.2, 0.8], [1, 1], [0, 0])
    assert result["box_distance"] == 0
    assert result["segment_distance"] > 0.4
    assert result["dual_gap"] == pytest.approx(result["segment_distance"] ** 2)


def test_contextual_output_outside_coordinate_range():
    """A failing nonlinear world must remain visible, not be fitted away."""
    result = segment_certificate([0.95, 0.8], [0.8, 0.8], [0.2, 0.2])
    assert result["box_distance"] == pytest.approx(0.15)
    assert result["coordinate_violations"] == 1
    assert result["dual_gap"] > 0


def test_negative_common_weight_is_affine_not_convex():
    """Extrapolation is not normalized positive evidence aggregation."""
    result = segment_certificate([-0.5, -0.5], [1, 1], [0, 0])
    assert result["affine_distance"] == 0
    assert result["segment_distance"] > 0
    assert result["weight_unconstrained"] == -0.5


def test_degenerate_singleton_and_separation():
    """Equal event outputs do not create an arbitrary nonzero direction."""
    r = segment_certificate([0.4, 0.6], [0.5, 0.5], [0.5, 0.5])
    assert r["dual_identity_error"] < 1e-14
    assert r["affine_distance"] == pytest.approx(r["segment_distance"])


def test_projection_witness_bounds_every_convex_combination():
    """The dual certificate covers all weights, not a chosen grid solution."""
    x, y, z = np.array([0.7, 0.2]), np.array([0.1, 0.9]), np.array([0.8, 0.8])
    r = segment_certificate(z, x, y)
    w = r["witness"]
    for t in np.linspace(0, 1, 101):
        assert w @ z - w @ (t * x + (1 - t) * y) >= r["dual_gap"] - 1e-14


def test_projection_orthogonal_coordinate_invariance():
    """Euclidean distances are invariant to a common orthogonal frame change."""
    q = np.array([[0, -1], [1, 0]])
    x, y, z = np.array([0.7, 0.2]), np.array([0.1, 0.9]), np.array([0.8, 0.8])
    a, b = segment_certificate(z, x, y), segment_certificate(q @ z, q @ x, q @ y)
    assert a["segment_distance"] == pytest.approx(b["segment_distance"])
    assert a["dual_gap"] == pytest.approx(b["dual_gap"])


def test_weight_only_update_has_no_remainder():
    """Updating one shared event weight changes only the event-segment direction."""
    x, y = np.array([0.8, 0.1]), np.array([0.2, 0.9])
    r = update_decomposition(0.25 * (x - y), x, y)
    assert r["shared_weight_change"] == pytest.approx(0.25)
    assert r["remainder_norm"] < 1e-14


def test_nonparallel_update_is_preserved():
    """A context-specific change must not disappear into a fitted shared weight."""
    r = update_decomposition([0.1, -0.1], [1, 1], [0, 0])
    assert r["shared_weight_change"] == 0
    assert r["remainder_energy_share"] == pytest.approx(1)
    assert r["pythagorean_error"] < 1e-14


@pytest.mark.parametrize("bad", [[], [[1]], [float("nan")], [float("inf")]])
def test_reject_invalid_vector(bad):
    """Invalid/missing evidence must not silently turn into zero."""
    with pytest.raises(ValueError):
        vector(bad)


def test_reject_coordinate_mismatch():
    """Different query rosters cannot be pooled into one certificate."""
    with pytest.raises(ValueError):
        segment_certificate([0, 1], [0], [1])
