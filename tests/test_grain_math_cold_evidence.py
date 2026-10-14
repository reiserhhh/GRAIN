#!/usr/bin/env python3
"""Synthetic tests for body-only normalization, rotary queries and finite actions."""

import numpy as np
import pytest

from grain_research.construction import math_cold_evidence as C
from grain_research.construction import math_query_evidence as M


def sample():
    """Return a deterministic grouped GQA carrier, unrelated to experiment truths."""
    rng = np.random.default_rng(87)
    return (
        rng.normal(size=(12, 2, 4)),
        rng.normal(size=(12, 2, 4)),
        rng.normal(size=(4, 4)),
        np.repeat(np.arange(3), 4),
    )


def test_body_normalization_and_pairing():
    """Removing non-body keys matches full-reading conditional normalization."""
    k, v, q, ids = sample()
    groups = [np.flatnonzero(ids == i) for i in range(3)]
    full = M.reading(
        np.concatenate((k, k[:2] * 3)), np.concatenate((v, v[:2])), q, groups
    )
    cold = C.reading(k, v, q, ids)
    np.testing.assert_allclose(cold[0], full["conditional"], atol=1e-12)
    np.testing.assert_allclose(cold[1], full["conditional_uniform"], atol=1e-12)
    np.testing.assert_allclose(cold[2], full["conditional_pairing"], atol=1e-12)
    np.testing.assert_allclose(cold[0], cold[1] + cold[2], atol=1e-12)


def test_rotary_zero_and_relative_frame():
    """Zero position is identity; common exact rotations preserve dot products."""
    _, _, q, _ = sample()
    freq = np.asarray([1, 0.01], np.float32)
    np.testing.assert_allclose(C.rotate(q, 0, freq, 1), q)
    q1, q2 = C.rotate(q, 10, freq, 1), C.rotate(q, 13, freq, 1)
    np.testing.assert_allclose(
        np.sum(q1 * q2, axis=1), np.sum(q * C.rotate(q, 3, freq, 1), axis=1), atol=1e-6
    )


def test_all_queries_no_truth():
    """Duplicate numerical query seeds produce identical readings without labels."""
    k, v, q, ids = sample()
    bundle = {
        "raw_queries": [q, q],
        "gaps": [4, 4],
        "frequencies": [1, 0.01],
        "scaling": 0.5,
    }
    result = C.all_queries(k, v, ids, 20, bundle)
    np.testing.assert_array_equal(result[0], result[1])


def test_polarization():
    """Finite nonlinear body/query interaction is retained rather than discarded."""
    high, low = lambda q: 3 * q + q * q, lambda q: q - q * q
    b, q, total = C.polarized_action(high, low, np.asarray([2.0]), np.asarray([-1.0]))
    np.testing.assert_allclose(b + q, total)


def test_event_labels_fail_closed():
    """A missing event is not silently returned as a fake zero observation."""
    k, v, q, ids = sample()
    with pytest.raises(ValueError):
        C.reading(k, v, q, np.zeros_like(ids))
