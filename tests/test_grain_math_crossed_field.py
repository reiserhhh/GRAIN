#!/usr/bin/env python3
"""Synthetic tests for complete crossed fields, signed effects and metric units."""

import numpy as np
import pytest

from grain_research.construction import math_crossed_field as C
from grain_research.construction.math_source_field import moment_map


def test_material_coverage():
    assert len({C.material(c, p) for c in range(8) for p in range(4)}) == 32
    with pytest.raises(ValueError):
        C.material(8, 0)


def test_declared_directions():
    b = np.array([1.0, 2.0])
    d = C.directions(b)
    assert tuple(d) == C.ARMS
    np.testing.assert_allclose(
        moment_map(4) @ d["unseen_source"],
        np.outer([0, -1, 0, 1], b / np.linalg.norm(b)),
        atol=1e-15,
    )


def test_additive_grid():
    x = 8 + np.arange(3)[:, None] + np.arange(4)[None, :]
    p, s = C.decompose(x)
    np.testing.assert_allclose(p["interaction"], 0)
    assert s["energy_identity_error"] < 1e-14
    assert s["interaction_share"] == 0


def test_interaction_only():
    x = np.outer([-1.0, 1], [-1.0, 1])
    p, s = C.decompose(x)
    assert s["interaction_share"] == 1
    assert s["cancellation_ratio"] == 0
    assert not p["mean"]


def test_zero_not_nan():
    _, s = C.decompose(np.zeros((2, 4)))
    assert s["context_share"] is None and s["cancellation_ratio"] is None


def test_vector_reconstruction_zero_sums():
    x = np.random.default_rng(3).normal(size=(3, 4, 4, 5))
    p, s = C.decompose(x)
    np.testing.assert_allclose(
        x, p["mean"] + p["context"][:, None] + p["wording"][None] + p["interaction"]
    )
    np.testing.assert_allclose(p["context"].sum(0), 0, atol=1e-14)
    np.testing.assert_allclose(p["interaction"].sum(0), 0, atol=1e-14)
    np.testing.assert_allclose(p["interaction"].sum(1), 0, atol=1e-14)
    assert s["energy_identity_error"] < 1e-13


def test_metric_frame_covariance():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(2, 4, 4, 6))
    frame = np.eye(4) + rng.normal(size=(4, 4)) * 0.2
    y = np.einsum("ij,cpjd->cpid", np.linalg.inv(frame).T, x)
    _, a = C.decompose(x, np.eye(4))
    _, b = C.decompose(y, frame @ frame.T)
    for key in a:
        if a[key] is not None:
            assert a[key] == pytest.approx(b[key], abs=1e-12)


def test_signed_covariance():
    x = np.outer([-1.0, 1], [1.0, -1])
    s = C.covariance_parts(x, -x)
    assert s["native"] == -1 and s["interaction"] == -1
    assert s["identity_error"] == 0


def test_permutation_equivariance():
    x = np.random.default_rng(2).normal(size=(3, 4, 4, 2))
    _, a = C.decompose(x)
    _, b = C.decompose(x[[2, 0, 1]][:, [3, 1, 2, 0]])
    for key in a:
        assert a[key] == pytest.approx(b[key], abs=1e-12)


@pytest.mark.parametrize("x", [np.zeros((1, 4)), np.zeros(3), [[0, 1], [2, np.nan]]])
def test_missing_or_nonfinite_grid(x):
    with pytest.raises(ValueError):
        C.decompose(x)


def test_bad_metric():
    with pytest.raises(ValueError):
        C.decompose(np.zeros((2, 3, 4, 5)), np.diag([1, 1, 1, -1]))


def test_normalized_physical_error():
    rows = [
        {
            "context": i,
            "wording": 0,
            "odd_action": 2 * d,
            "physical_delta_norm": d,
            "even_change": 0,
            **{m: d for m in C.METHODS},
        }
        for i, d in enumerate([1.0, 4])
    ]
    s = C.summarize(rows)
    assert s["native_unit_action_rms"] == 2
    assert s["global_source_unit_rmse"] == 1
