#!/usr/bin/env python3
"""Finite transport tests independent of private authors or model weights."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.functional_transfer import shared_gradient_scores


def test_shared_linear_suffix_exact_transport_and_equal_view_pooling():
    """A common linear suffix transports without a target fitted at inference."""
    rng = np.random.default_rng(19)
    g = rng.normal(size=7)
    d = rng.normal(size=(13, 3, 7))
    reference, views, scores = shared_gradient_scores(np.tile(g, (5, 1)), d)
    np.testing.assert_allclose(reference, g)
    np.testing.assert_allclose(views, np.einsum("uvd,d->uv", d, g))
    np.testing.assert_allclose(scores, d.mean(axis=1) @ g)


def test_transfer_reference_does_not_depend_on_new_authors():
    """Changing a held author cannot modify the reference or other predictions."""
    rng = np.random.default_rng(20)
    g, d = rng.normal(size=(8, 6)), rng.normal(size=(9, 3, 6))
    original = shared_gradient_scores(g, d)
    d[0] *= 1000
    changed = shared_gradient_scores(g, d)
    np.testing.assert_array_equal(original[0], changed[0])
    np.testing.assert_array_equal(original[1][1:], changed[1][1:])


@pytest.mark.parametrize("bad", [np.zeros((2, 3)), np.full((2, 3, 4), np.nan)])
def test_invalid_transfer_vectors_fail(bad):
    """Reject wrong rank and nonfinite arrays instead of silently pooling them."""
    with pytest.raises(ValueError):
        shared_gradient_scores(np.ones((3, 4)), bad)
