#!/usr/bin/env python3
"""Witnesses for conditional mixed-action readers and reference path matrices."""

import numpy as np

from grain_research.vector_mechanisms.joint_readout import (
    centered_logits,
    fit_joint,
    pair_kernel,
    predict_joint,
)
from grain_research.vector_mechanisms.query_functional import (
    frozen_reference_matrix,
    integrated_matrix,
    matrix_response,
)


def test_pair_kernel_is_explicit_psd_tensor():
    """The kernel is an ordinary positive-semidefinite feature Gram matrix."""
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(2, 7, 4))
    phi = np.array(
        [(np.outer(x, y) + np.outer(y, x)).ravel() / 2 for x, y in zip(a, b)]
    )
    k = pair_kernel(a, b, a, b)
    np.testing.assert_allclose(k, phi @ phi.T, atol=1e-12)
    assert np.linalg.eigvalsh(k).min() > -1e-10


def test_joint_is_swap_invariant_and_zero_displacement():
    """Argument order is not an additional unmeasured causal direction."""
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=(2, 9, 3))
    y = centered_logits(rng.normal(size=(9, 5)))
    model = fit_joint(a[:6], b[:6], y[:6])
    np.testing.assert_allclose(
        predict_joint(a[6:], b[6:], model), predict_joint(b[6:], a[6:], model)
    )
    np.testing.assert_array_equal(
        predict_joint(a[6:] * 0, b[6:], model), np.zeros((3, 5))
    )
    np.testing.assert_allclose(
        predict_joint(a[6:], b[6:], model).mean(1), 0, atol=1e-12
    )


def test_held_targets_do_not_enter_fit():
    """Changing evaluation outcomes cannot alter fitted normalization or weights."""
    rng = np.random.default_rng(8)
    a, b = rng.normal(size=(2, 10, 4))
    y = rng.normal(size=(10, 2))
    first = fit_joint(a[:8], b[:8], y[:8])
    y[8:] += 1e6
    second = fit_joint(a[:8], b[:8], y[:8])
    np.testing.assert_array_equal(first["weights"], second["weights"])
    assert first["scale"] == second["scale"]


def test_reference_matrix_and_two_body_conditions():
    """Bilinear M/R response gives the symmetric allocation under exact Simpson integration."""
    m0, m1 = 2.0, 4.0
    d = np.array([[1.0, 2.0], [3.0, 1.0]])
    base = np.array([[0.2, 0.1], [-0.1, 0.3]])
    gradients = np.array([[base * m0] * 3, [base * m1] * 3])
    matrix = integrated_matrix(gradients)
    assert abs(matrix_response(matrix, d) - ((m0 + m1) / 2 * np.sum(base * d))) < 1e-12
    entries = np.array([matrix, matrix + 2, matrix + 999])
    got = frozen_reference_matrix(entries, ["reference", "reference", "evaluation"])
    np.testing.assert_allclose(got, matrix + 1)
