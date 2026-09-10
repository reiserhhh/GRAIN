#!/usr/bin/env python3
"""Component identities, mixed derivatives and signed error allocations."""

import numpy as np
import pytest

from grain_research.vector_mechanisms import evidence_components as f
from grain_research.vector_mechanisms import normalized_evidence as reader
from grain_research.vector_mechanisms.token_potential import hidden_gradient
from tests.test_grain_nonlinear_pooling import toy


@pytest.mark.parametrize("arm", reader.ARMS)
def test_components_reconstruct_gradient_actions_and_finite_difference(arm):
    rng, records, local = toy()
    model = reader.reference(records, local, arm)
    model["coef"] = np.r_[local["coef"], rng.normal(size=48)]
    b, last = records[0]["body"], records[0]["last"]
    parts = f.densities(b, last, model)
    direct = reader.predict(b, last, model)[1]
    np.testing.assert_allclose(parts.sum(0), direct, atol=1e-12)
    ds = rng.normal(size=(4, *b.shape))
    full, allocated = f.actions(parts, ds, model)
    g = hidden_gradient(direct, model)
    np.testing.assert_allclose(full.sum(0), np.einsum("th,fth->f", g, ds), atol=1e-12)
    np.testing.assert_allclose(
        allocated.sum(0),
        np.einsum("th,fth->f", g, ds - ds.mean(1, keepdims=True)),
        atol=1e-12,
    )
    fd = (
        reader.scalar(b + 1e-5 * ds[0], last, model)
        - reader.scalar(b - 1e-5 * ds[0], last, model)
    ) / 2e-5
    assert fd == pytest.approx(full[:, 0].sum(), abs=1e-8)
    shifted = b.copy()
    shifted[:20] += rng.normal(size=b[:20].shape) * 0.01
    change = f.densities(shifted, last, model)[:, 20:] - parts[:, 20:]
    np.testing.assert_array_equal(change[0], 0)
    if arm == reader.ARMS[0]:
        np.testing.assert_array_equal(change[1:3], 0)
    else:
        assert np.linalg.norm(change[1:3]) > 1e-8
    assert np.linalg.norm(change[3]) > 1e-8


def test_signed_moments_reconstruct_error_and_allow_cancellation():
    y = np.array([1.0, -2.0, 3.0, 0.5])
    parts = np.stack((2 * y, -y, np.zeros_like(y)))
    a = f.projection_account(*f.moments(y, parts))
    assert a["sse_over_zero"] == pytest.approx(0, abs=1e-14)
    np.testing.assert_allclose(a["projection"], [2, -1, 0])
    assert a["gram"][0, 1] < 0
    assert a["gram"].sum() == pytest.approx(1)
    b = f.improvement_account(y, np.zeros_like(y), parts)
    assert b["relative_improvement"] == pytest.approx(1)
    np.testing.assert_allclose(b["credits"], [2, -1, 0])
    assert b["closure_error"] < 1e-14


def test_random_error_credit_and_pair_gram_identity():
    rng = np.random.default_rng(710)
    y, baseline = rng.normal(size=(8, 4)), rng.normal(size=(8, 4))
    changes = rng.normal(size=(4, 8, 4))
    a = f.improvement_account(y, baseline, changes)
    actual = np.mean((baseline + changes.sum(0) - y) ** 2)
    assert a["new_error"] == pytest.approx(actual)
    assert a["closure_error"] < 1e-12
    assert a["credits"].sum() == pytest.approx(
        1 - actual / np.mean((baseline - y) ** 2)
    )
    np.testing.assert_allclose(a["gram"], a["gram"].T)


def test_empty_nonfinite_and_misaligned_inputs_stop():
    with pytest.raises(ValueError, match="Aligned"):
        f.moments(np.zeros(4), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="Finite"):
        f.moments(np.ones(4), np.full((2, 4), np.nan))
    with pytest.raises(ValueError, match="Positive"):
        f.improvement_account(np.ones(4), np.ones(4), np.zeros((2, 4)))
    with pytest.raises(ValueError, match="Positive"):
        f.projection_account(0, np.ones(2), np.eye(2))
