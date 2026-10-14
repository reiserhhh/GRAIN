#!/usr/bin/env python3
"""Synthetic mechanism checks; no private data, model calls or fitted teachers."""

import numpy as np
import pytest

from grain_research.construction import math_prospective_evidence as P
from grain_research.construction import math_source_target_law as M


def bank():
    """Create all profiles with deterministic finite joint anchors."""
    rng = np.random.default_rng(229)
    return {
        "source_means": rng.normal(size=(32, 2, 4)),
        "codes": np.arange(32),
        "variance": 2.0,
        "anchors": rng.normal(size=(32, 5)),
    }


def entry():
    """Use twelve distinct opaque event addresses."""
    return {
        "author": 0,
        "event_ids": [f"e{i}" for i in range(12)],
        "vectors": np.random.default_rng(19).normal(size=(12, 4)).tolist(),
        "support": [0, 1, 2, 3],
        "counter": [4, 5, 6, 7],
    }


def test_old_binary_score_projection_and_batch_invariance():
    b, x = bank(), np.random.default_rng(2).normal(size=(4, 4))
    w = M.weights(b, x)
    np.testing.assert_allclose(w.sum(1), 1, atol=1e-14)
    np.testing.assert_allclose(
        M.summaries(b, w)["high_log_odds"], P.accounts(b, x)[:, 0], atol=1e-14
    )
    np.testing.assert_allclose(M.weights(b, x[:1])[0], w[0], atol=1e-15)


def test_joint_moments_and_cdf_ties_literal():
    b = bank()
    b["anchors"][:3, 0] = 0
    w = M.weights(b, np.zeros((1, 4)))
    a = M.summaries(b, w)
    mean = sum(w[0, i] * b["anchors"][i] for i in range(32))
    cov = sum(
        w[0, i] * np.outer(b["anchors"][i] - mean, b["anchors"][i] - mean)
        for i in range(32)
    )
    np.testing.assert_allclose(a["mean"][0], mean, atol=1e-14)
    np.testing.assert_allclose(a["covariance"][0], cov, atol=1e-14)
    np.testing.assert_allclose(
        M.cdf(w, b["anchors"], 0, [0])[0, 0],
        sum(w[0, i] for i in range(32) if b["anchors"][i, 0] <= 0),
    )
    assert np.linalg.eigvalsh(cov).min() > -1e-13


def test_binary_superiority_identity_and_common_null():
    b = bank()
    bits = ((b["codes"][:, None] >> np.arange(5)) & 1).astype(float)
    w = M.weights(b, np.random.default_rng(4).normal(size=(4, 4)))
    p = w @ bits
    pi = M.superiority(w, bits)
    np.testing.assert_allclose(pi, 0.5 + 0.5 * (p[:, None] - p[None]), atol=1e-14)
    np.testing.assert_allclose(pi + pi.swapaxes(0, 1), 1, atol=1e-14)
    common = np.broadcast_to(w[:1], w.shape)
    np.testing.assert_allclose(M.superiority(common, b["anchors"]), 0.5, atol=1e-14)


def test_cited_equal_mean_nontransitive_dice():
    faces = np.array([1, 5, 9, 3, 4, 8, 2, 6, 7], float)
    y = np.tile(faces[:, None], (1, 5))
    w = np.zeros((3, 9))
    for i in range(3):
        w[i, 3 * i : 3 * i + 3] = 1 / 3
    np.testing.assert_allclose(w @ y, 5)
    pi = M.superiority(w, y)
    np.testing.assert_allclose(pi[[0, 1, 2], [1, 2, 0], 0], 5 / 9)
    edges = np.array([[0, 1], [0, 2], [1, 2]])
    d = 2 * pi[edges[:, 0], edges[:, 1], 0] - 1
    assert M.directed_cycle(edges, d, 3) == [0, 1, 2, 0]


def test_source_row_permutation_equivariance_and_fixed_reference():
    b = bank()
    w = M.weights(b, np.zeros((1, 4)))
    ix = np.random.default_rng(9).permutation(32)
    other = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in b.items()}
    for key in ("source_means", "codes", "anchors"):
        other[key] = other[key][ix]
    w2 = M.weights(other, np.zeros((1, 4)))
    np.testing.assert_allclose(w2, w[:, ix], atol=1e-14)
    for key, values in M.summaries(b, w).items():
        np.testing.assert_allclose(M.summaries(other, w2)[key], values, atol=1e-13)


@pytest.mark.parametrize(
    "bad", ["label", "answer", "duplicate", "overlap", "range", "float"]
)
def test_cold_inputs_fail_closed(bad):
    e = entry()
    if bad in ("label", "answer"):
        e[bad] = 0.9
    elif bad == "duplicate":
        e["event_ids"][-1] = e["event_ids"][0]
    elif bad == "overlap":
        e["counter"][-1] = 0
    elif bad == "range":
        e["counter"][-1] = 12
    else:
        e["support"] = [0.0, 1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        M.score(bank(), e)


def test_scored_record_preserves_three_arms():
    b, e = bank(), entry()
    out = M.score(b, e)
    assert out["weights"].shape == (3, 32)
    assert out["covariance"].shape == (3, 5, 5)
    assert out["mean"].shape == (3, 5)


@pytest.mark.parametrize("bad", ["mass", "negative", "nonfinite", "keys"])
def test_invalid_laws_not_silently_repaired(bad):
    b = bank()
    w = M.weights(b, np.zeros((1, 4)))
    if bad == "keys":
        b["application_labels"] = np.ones(5)
        with pytest.raises(ValueError):
            M.weights(b, np.zeros((1, 4)))
        return
    if bad == "mass":
        w *= 0.9
    elif bad == "negative":
        w[0, 0] = -0.1
    else:
        w[0, 0] = np.nan
    with pytest.raises(ValueError):
        M.superiority(w, b["anchors"])
