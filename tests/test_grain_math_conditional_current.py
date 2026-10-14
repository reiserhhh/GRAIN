#!/usr/bin/env python3
"""Synthetic tests for the conditional-current measurement interface."""

import numpy as np
import pytest

from grain_research.construction import math_conditional_current as M


def fixture_counts():
    """Generate legal nonempty three-view author paths for numerical tests."""
    rng = np.random.default_rng(95)
    u = np.repeat(np.arange(8), 12)
    v = np.tile(np.repeat(np.arange(3), 4), 8)
    z = np.tile([0, 0, 1, 1], 24)
    counts = M.counts_for_comments(rng.integers(0, 4, (96, 4)), u, v, z, 8, 2)
    refs = np.full((4, 2, 3, 4, 4), 1 / 16)
    return counts, refs, u, v, z


def test_lift_norm_and_covariance():
    counts, refs, *_ = fixture_counts()
    f, q = M.lift(counts, refs)
    assert np.allclose(q.sum((2, 3)), 1)
    assert np.allclose(
        (f["resolved"] ** 2).sum(-1),
        (f["visible"] ** 2).sum(-1) + (f["hidden"] ** 2).sum(-1),
    )
    cov = {k: M.author_covariance(x, np.array([4] * 4 + [5] * 4)) for k, x in f.items()}
    assert np.allclose(cov["resolved"], cov["visible"] + cov["hidden"])


def test_empty_views_are_not_zero_measurements():
    counts, refs, *_ = fixture_counts()
    counts[0, 0] = 0
    with pytest.raises(ValueError, match="Nonempty"):
        M.lift(counts, refs)


def test_source_null_legal_comments_preserve_opportunities():
    counts, refs, u, v, z = fixture_counts()
    patterns, probabilities, _, _ = M.source_markov(counts.sum(1), 0.5)
    assert patterns.shape == (256, 4)
    assert np.allclose(probabilities.sum(1), 1)
    x = M.sample_comments(patterns, probabilities, z, np.random.default_rng(3))
    ghost = M.counts_for_comments(x, u, v, z, 8, 2)
    assert np.array_equal(counts.sum((-2, -1)), ghost.sum((-2, -1)))
    real, _ = M.lift(counts, refs)
    null, _ = M.lift(ghost, refs)
    for key in real:
        assert np.allclose(real[key][:, :, :, 1], null[key][:, :, :, 1])


def test_fisher_loss_and_legal_tangents():
    base, tangents, full, pooled = M.fisher_witness()
    assert np.allclose(full, 0.24 * np.eye(3))
    assert np.array_equal(pooled, np.zeros((3, 3)))
    for tangent in tangents:
        assert (base + tangent).min() > 0
        assert np.isclose((base + tangent).sum(), 1)
        assert np.array_equal(tangent.sum(0), np.zeros((3, 3)))


def test_cancellation_can_act_only_on_conditional_reader():
    _, tangents, _, _ = M.fisher_witness()
    kernel = tangents[0].copy()
    assert np.sum(tangents[0] * kernel) > 0
    assert np.sum(tangents[0].sum(0) * kernel.mean(0)) == 0
    constant = np.broadcast_to(kernel[0], kernel.shape)
    assert np.isclose(np.sum(tangents[0] * constant), 0)
    reset = np.broadcast_to(np.arange(3)[None, None, :], (4, 3, 3))
    assert np.isclose(np.sum(tangents[0] * reset), 0)


def test_state_relabeling_covariance_invariance():
    counts, refs, *_ = fixture_counts()
    order = [3, 1, 0, 2]
    a, _ = M.lift(counts, refs)
    b, _ = M.lift(counts[..., order, :][..., order], refs[..., order, :][..., order])
    for key in a:
        assert np.allclose(
            M.author_covariance(a[key], np.ones(8)),
            M.author_covariance(b[key], np.ones(8)),
        )
