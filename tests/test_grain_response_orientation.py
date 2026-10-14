#!/usr/bin/env python3
"""Orientation and finite orbit tests, not an assumed psychology simulator."""

from itertools import permutations

import numpy as np
import pytest

from grain_research.construction import math_criterion_components as C
from grain_research.construction import math_response_orientation as M


def test_odd_orientation_and_window_exchange():
    a, b, k = np.array([-1, 0, 1.0]), np.array([-2, 0, 2.0]), np.array([1, 2, 4.0])
    t = M.oriented_accounts(a, b, k)
    assert t > 0
    assert M.oriented_accounts(-a, -b, k) == pytest.approx(-t)
    assert M.oriented_accounts(b, a, k) == pytest.approx(t)
    assert M.oriented_accounts(a, b, np.ones(3)) == pytest.approx(0)
    assert M.oriented_accounts(a, b, -k) < 0


def test_pullback_and_exact_conditional_orbit_variance():
    a = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [2, 1, 0, 0.0]])
    b = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 2, 0.0]])
    y, groups, k = np.array([0, 0, 1, 1.0]), np.zeros(4, int), np.array([2, -1, 3.0])
    ell = M.source_pullback(a, b, k)
    samples = []
    for ix in permutations(range(4)):
        e = y[list(ix)] - 0.5
        value = M.oriented_accounts(a @ e, b @ e, k)
        assert ell @ e == pytest.approx(value)
        samples.append(value)
    assert np.mean(samples) == pytest.approx(0, abs=1e-14)
    assert M.orbit_variance(ell, C.orbit_covariance(y, groups)) == pytest.approx(
        np.var(samples), abs=1e-14
    )


@pytest.mark.parametrize("bad", [np.array([1, np.nan]), np.array([1.0])])
def test_invalid_views_rejected(bad):
    with pytest.raises(ValueError):
        M.oriented_accounts(bad, bad, np.ones(len(bad)))


def numeric_record():
    """Build a qualified finite fixture with a channel closure, not a trait claim."""
    score = np.zeros((2, 3, 5))
    score[:, :, 0] = [-1, 0, 1]
    grams = np.zeros((4, 5, 3, 3))
    grams[1, 0] = np.diag([1, 2, 4])
    grams[0] = grams[1:].sum(0)
    return {
        "version": "HM284-v1",
        "author_keys": ["a", "b", "c"],
        "left_event_ids": [[f"left-{u}-0", f"left-{u}-1"] for u in range(3)],
        "right_event_ids": [[f"right-{u}-0", f"right-{u}-1"] for u in range(3)],
        "source_readings": score.tolist(),
        "cross_output_kernels": grams.tolist(),
    }


def test_cold_signed_numeric_interface_and_all_channels():
    record = numeric_record()
    result = np.asarray(M.cold(record)["oriented_source_query_channels"])
    assert result.shape == (5, 5, 4)
    assert result[0, 0, 0] == pytest.approx(1)
    assert result[:, :, 0] == pytest.approx(result[:, :, 1:].sum(-1))
    record["source_readings"] = (-np.asarray(record["source_readings"])).tolist()
    assert np.asarray(
        M.cold(record)["oriented_source_query_channels"]
    ) == pytest.approx(-result)


@pytest.mark.parametrize("failure", ["reused_event", "application_label"])
def test_cold_preserves_event_identity_and_no_outcome_fields(failure):
    record = numeric_record()
    if failure == "reused_event":
        record["right_event_ids"][0][0] = record["left_event_ids"][0][0]
    else:
        record["application_labels"] = [0, 1, 0]
    with pytest.raises(ValueError):
        M.cold(record)
