#!/usr/bin/env python3
"""Checks of condition reference, unknown material and joint-vector geometry."""

import numpy as np
import pytest

from grain_research.conditional_reference import (
    cell_means,
    completion_envelope,
    fit_reference,
    score_reference,
)
from scripts.score_grain_conditional_reference import restore_reference, score_packet


def test_author_balance_and_source_isolation():
    c = np.array(["a", "a", "b", "a", "b", "held"])
    x = np.array([[0], [0], [1], [2], [3], [999]], float)
    u = np.array([0, 0, 0, 1, 1, 2])
    r = fit_reference(c, x, u, [0, 1])
    assert r["keys"] == ["a", "b"]
    np.testing.assert_allclose(r["weights"], [7 / 12, 5 / 12])
    np.testing.assert_allclose(r["mean"].ravel(), [1, 2])
    x[-1] = -999
    other = fit_reference(c, x, u, [0, 1])
    np.testing.assert_array_equal(other["mean"], r["mean"])


def test_single_condition_and_unknown_are_not_zero():
    r = fit_reference(["a", "a"], [[0.2], [0.4]], [0, 1], [0, 1])
    s = score_reference(["a", "a"], [[0.5], [0.7]], r)
    assert s["complete"]
    np.testing.assert_allclose(s["point"], [0.6])
    np.testing.assert_allclose(s["actual"], s["context"] + s["departure"] + s["unseen"])
    unknown = score_reference(["z"], [[0.9]], r)
    assert unknown["point"] is None and unknown["supported_target"] is None
    assert unknown["coverage"] == 0 and unknown["missing"] == 1
    np.testing.assert_allclose(unknown["unseen"], [0.9])


def test_completion_geometry_and_sharp_extreme():
    d = np.array([[1.0, 1.0], [0.0, 0.0]])
    t = np.array([[1, 0, 1], [0, 1, -1]], float)
    low, high, radius, box = completion_envelope(np.zeros(2), 0.5, d, t)
    np.testing.assert_allclose(radius, [0.5, 0.5, 0])
    np.testing.assert_allclose(box, [0.5, 0.5, 1])
    np.testing.assert_allclose(np.array([0.5, 0.5]) @ t, high)
    assert np.all(high >= low)


def test_reveal_nesting_and_exact_width_change():
    rng = np.random.default_rng(9)
    h = rng.normal(size=(4, 7))
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    d = rng.normal(size=(7, 3))
    scores = h @ d
    r = fit_reference(["a", "b", "a", "b"], scores, [0, 0, 1, 1], [0, 1])
    full = score_reference(["a", "b"], scores[:2], r)
    hidden = score_reference(["b"], scores[1:2], r)
    t = np.column_stack([np.eye(3), np.ones(3) / 3])
    lo, hi, radius, _ = completion_envelope(hidden["center"], hidden["missing"], d, t)
    known = full["point"] @ t
    assert np.all(known >= lo - 1e-12) and np.all(known <= hi + 1e-12)
    np.testing.assert_allclose(2 * radius, np.linalg.norm(d @ t, axis=0))


def test_invalid_evidence_is_not_imputed():
    with pytest.raises(ValueError):
        cell_means(["a"], [[np.nan]])
    with pytest.raises(ValueError):
        cell_means([""], [[0]])
    with pytest.raises(ValueError):
        completion_envelope([0], -0.1, [[1]], [[1]])


def test_float32_input_uses_float64_frozen_scoring():
    rng = np.random.default_rng(16)
    h = rng.normal(size=(8, 11)).astype("float32")
    d = rng.normal(size=(11, 3))
    h64 = h.astype(float)
    scores = h64 / np.linalg.norm(h64, axis=1, keepdims=True) @ d
    c = np.array(["a", "b"] * 4)
    ref = fit_reference(c[:4], scores[:4], [0, 0, 1, 1], [0, 1])
    exported = {
        k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in ref.items()
    }
    restored = restore_reference(exported)
    cold = score_packet(c[4:], h[4:], restored, d, np.eye(3))
    expected = score_reference(c[4:], scores[4:], ref)
    np.testing.assert_allclose(cold["point"], expected["point"], rtol=0, atol=1e-13)
    one = score_packet(c[4:6], h[4:6], restored, d, np.eye(3))
    np.testing.assert_allclose(one["actual"], scores[4:6].mean(0), rtol=0, atol=1e-13)


def test_cold_scorer_rejects_bad_reference_and_zero_vector():
    ref = fit_reference(["a", "b"], [[0.1], [0.2]], [0, 1], [0, 1])
    damaged = ref.copy()
    damaged["weights"] = [0.1, 0.1]
    with pytest.raises(ValueError):
        restore_reference(damaged)
    with pytest.raises(ValueError):
        score_packet(["a"], [[0]], ref, np.ones((1, 1)), np.ones((1, 1)))
