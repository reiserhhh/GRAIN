#!/usr/bin/env python3
"""Conditional reference invariance is not invariance under an unseen text edit."""

import numpy as np
import pytest

from grain_research.construction import math_context_anchored_operation as M


def toy():
    """Nonorthogonal source even/odd states with an exactly known context map."""
    e = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    d = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [-1.0, -1.0, 0.0]])
    qe = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    qd = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [-1.0, -1.0, 0.0]])
    return e + d, e - d, qe + qd, qe - qd


def test_source_joint_exchange_and_involution():
    bp, bm, qp, qm = toy()
    models, info = M.fit(bp, bm, qp, qm)
    assert info["source_even_fit_max_error"] < 1e-12
    for model in models.values():
        for b, q, expected_b, expected_q in zip(bp, qp, bm, qm, strict=True):
            new_b, new_q = M.operate(model, b, q, conditioned=True)
            np.testing.assert_allclose(new_b, expected_b, atol=1e-12)
            np.testing.assert_allclose(new_q, expected_q, atol=1e-12)
            restored = M.operate(model, new_b, new_q, conditioned=True)
            np.testing.assert_allclose(restored, (b, q), atol=1e-12)
        rng = np.random.default_rng(71)
        b, q = rng.normal(size=(2, 3))
        restored = M.operate(
            model, *M.operate(model, b, q, conditioned=True), conditioned=True
        )
        np.testing.assert_allclose(restored, (b, q), atol=1e-12)


def test_oblique_anchor_is_inert_even_on_cold_input():
    model = M.fit(*toy())[0]["oblique"]
    for b, q in np.random.default_rng(18).normal(size=(20, 2, 3)):
        plain = M.operate(model, b, q, conditioned=False)
        anchored = M.operate(model, b, q, conditioned=True)
        np.testing.assert_allclose(plain, anchored, atol=1e-12)


@pytest.mark.parametrize("method", ("orthogonal", "oblique"))
@pytest.mark.parametrize("conditioned", (False, True))
def test_three_cause_account(method, conditioned):
    model = M.fit(*toy())[0][method]
    bp, bm, qp, qm = np.random.default_rng(20).normal(size=(4, 9, 3))
    rows = M.accounts(model, bp, bm, qp, qm, conditioned=conditioned)
    total = []
    for b, bb, q, qq in zip(bp, bm, qp, qm, strict=True):
        a = M.operate(model, b, q, conditioned=conditioned)[1] - qq
        c = M.operate(model, bb, qq, conditioned=conditioned)[1] - q
        total.append((a @ a + c @ c) / 2)
    np.testing.assert_allclose(total, rows["operator_error_sq"], atol=1e-12)
    pieces = 4 * (
        rows["context_sq"]
        + rows["missing_sq"]
        + rows["contamination_sq"]
        + rows["odd_cross"]
    )
    np.testing.assert_allclose(pieces, total, atol=1e-12)
    if method == "orthogonal":
        np.testing.assert_allclose(rows["odd_cross"], 0, atol=1e-12)


def test_cold_true_edit_contaminates_a_predicted_invariant_anchor():
    model = M.fit(*toy())[0]["orthogonal"]
    # Source A kills D_b, not every new true-edit direction.
    bp, bm = np.array([[1.0, 0.0, 0.0]]), np.array([[-1.0, 0.0, 0.0]])
    qp, qm = np.array([[1.0, 1.0, 0.0]]), np.array([[-1.0, -1.0, 0.0]])
    rows = M.accounts(model, bp, bm, qp, qm, conditioned=True)
    assert rows["contamination_sq"][0] > 1
    assert rows["missing_sq"][0] < 1e-20


def test_source_and_single_input_guards():
    values = toy()
    with pytest.raises(ValueError, match="Aligned source"):
        M.fit(values[0][:-1], *values[1:])
    model = M.fit(*values)[0]["orthogonal"]
    with pytest.raises(ValueError, match="One aligned"):
        M.operate(model, values[0], values[2], conditioned=True)
    with pytest.raises(ValueError, match="Finite input"):
        M.operate(model, np.array([np.nan, 0.0, 0.0]), values[2][0], conditioned=True)


def test_application_root_is_not_a_source():
    from scripts.run_grain_context_anchored_operation import OUT, load_joint

    with pytest.raises(ValueError, match="Source fit"):
        load_joint(OUT, source_only=True)
