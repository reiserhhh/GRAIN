#!/usr/bin/env python3
"""Synthetic tests for finite response bookkeeping and model-family distinctions."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.conditional_inputs import (
    evidence_mask,
    evidence_summaries,
)
from grain_research.vector_mechanisms.conditional_navigation import (
    error_accounting,
    projected_response_error,
    response_metrics,
    scalar_gain_oracle,
)


def test_correlation_is_not_amplitude_recovery():
    """Perfect ordering can coexist with a tiny response amplitude."""
    a = np.array([-2, -1, 1, 2.0])
    m = response_metrics(a, 0.07 * a)
    assert m["r"] == pytest.approx(1)
    assert m["amplitude_ratio"] == pytest.approx(0.07)
    assert m["mse_over_zero"] == pytest.approx(0.93**2)


def test_error_cross_term_is_required():
    """Two biased approximations can cancel rather than contribute additive MSE."""
    m = error_accounting([1, 2, 3], [2, 3, 4], [1, 2, 3])
    assert m["native_minus_program_mse"] == 0
    assert m["twice_error_cross"] == -2
    assert m["accounting_error"] < 1e-12


def test_projected_orthogonal_terms_can_have_nonzero_cross():
    """Reader projection destroys the orthogonality of state-space errors."""
    m = projected_response_error([0, 0], [2, 3], [1, 0], [[1, 0]], [1, 1])
    assert m["state_cross"] == 0
    assert m["reader_cross"] == 6
    assert m["reader_total_sq"] == 16
    assert m["reader_accounting_error"] < 1e-12


def test_gain_oracle_is_not_an_independent_program():
    """A retrospectively fitted gain is marked as an oracle even if perfect."""
    m = scalar_gain_oracle([1, 2, 3], [0.1, 0.2, 0.3])
    assert m["oracle_gain"] == pytest.approx(10)
    assert m["oracle_sse"] < 1e-25
    assert m["inference_usable"] is False


def test_degenerate_responses_report_undefined_ratios():
    """A constant/zero teacher is not reported as a perfect recovered response."""
    m = response_metrics([0, 0, 0], [1, 2, 3])
    assert m["r"] is None
    assert m["mse_over_zero"] is None
    assert m["amplitude_ratio"] is None
    with pytest.raises(ValueError):
        response_metrics([np.nan], [1])


def test_reader_quotient_commutes_with_unrestricted_ridge():
    """Changing only a fixed output projection cannot create a new estimator."""
    rng = np.random.default_rng(12)
    x, y, l = (
        rng.normal(size=(20, 5)),
        rng.normal(size=(20, 8)),
        rng.normal(size=(8, 3)),
    )
    a = x.T @ x + np.eye(5)
    state_then_reader = np.linalg.solve(a, x.T @ y) @ l
    direct_reader = np.linalg.solve(a, x.T @ (y @ l))
    np.testing.assert_allclose(state_then_reader, direct_reader, atol=1e-13)


def test_existing_secants_do_not_expand_endpoint_span():
    """Finite differences reweight existing targets; new endpoints are different."""
    rng = np.random.default_rng(23)
    y = rng.normal(size=(4, 12))
    d = np.diff(np.eye(4), axis=0)
    assert np.linalg.matrix_rank(np.vstack([y, d @ y])) == np.linalg.matrix_rank(y)


def test_evidence_summaries_do_not_read_scaffold_positions():
    """Changing instructions outside selected body/query does not change summaries."""
    offsets = [(0, 3), (4, 6), (6, 8), (9, 11), (12, 13)]
    body = evidence_mask(offsets, [(4, 8)], [False, False, False, True, False])
    h = np.arange(15, dtype=float).reshape(5, 3)
    q = [False, False, False, True, False]
    a = evidence_summaries(h, body, q)
    h[0] = -1000
    h[-1] = 1000
    np.testing.assert_allclose(a, evidence_summaries(h, body, q))
    np.testing.assert_allclose(a[0], h[1:3].mean(0))
