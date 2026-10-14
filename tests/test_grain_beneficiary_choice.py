#!/usr/bin/env python3
"""Synthetic controls for beneficiary-qualified real-evidence observations."""

import numpy as np
import pytest

from grain_research.construction import math_beneficiary_choice as M


def test_order_orientation():
    z = np.zeros((1, 2, 2, 2, 2, 2))
    z[..., 0, 0] = 2
    z[..., 1, 1] = 2
    a = M.accounts(z)
    assert np.all(a["oriented"] == 2)
    assert np.all(a["order_defect"] == 0)


def test_shared_utility_gauge():
    rng = np.random.default_rng(2)
    z = rng.normal(size=(3, 2, 2, 2, 2, 2))
    shift = rng.normal(size=z.shape[:-1] + (1,))
    np.testing.assert_allclose(M.accounts(z)["delta"], M.accounts(z + shift)["delta"])


def test_constant_evidence_cannot_make_action():
    z = np.zeros((2, 2, 2, 2, 2, 2))
    z[:, :, 0, 0, 0, 0] = 8
    a = M.accounts(z)
    assert not np.any(a["selective"])
    assert not np.any(a["delta"])


def test_primary_and_cost_contrast():
    z = np.zeros((1, 2, 2, 2, 2, 2))
    for b, row in enumerate(((2, 4), (1, 2))):
        for c, d in enumerate(row):
            z[0, 0, b, c, 0, 0] = d
            z[0, 0, b, c, 1, 1] = d
    a = M.accounts(z)
    np.testing.assert_array_equal(a["delta"][0], [[2, 4], [1, 2]])
    assert a["selective"][0] == 1.5
    assert a["cost_interaction"][0] == 1


def test_full_character_reconstruction_and_parseval():
    a = np.random.default_rng(4).normal(size=(5, 2, 2, 2, 2))
    masks, co, rebuilt = M.characters(a)
    assert len(masks) == 16
    np.testing.assert_allclose(a, rebuilt, atol=1e-14)
    np.testing.assert_allclose((a * a).mean((1, 2, 3, 4)), (co * co).sum(1))


def test_selective_and_cost_accounts_are_fixed_characters():
    z = np.random.default_rng(17).normal(size=(5, 2, 2, 2, 2, 2))
    a = M.accounts(z)
    masks, co, _ = M.characters(a["oriented"])
    np.testing.assert_allclose(a["selective"], 4 * co[:, masks.index((1, 1, 0, 0))])
    np.testing.assert_allclose(
        a["cost_interaction"], 8 * co[:, masks.index((1, 1, 1, 0))]
    )


def test_body_and_option_relabel_preserved():
    texts = [f"original data {i}" for i in range(4)]
    p0, p1 = (M.render(texts, "other", 3, o) for o in (0, 1))
    assert p0.split("Question:")[0] == p1.split("Question:")[0]
    for t in texts:
        assert t in p0 and t in p1
    assert "L: The writer keeps 7 points." in p0
    assert "R: The writer keeps 7 points." in p1
    assert "another participant receives 10 additional points" in p0
    assert "the writer receives 10 additional points" in M.question("self", 7, 0)


def test_order_defect_is_retained_not_assumed_zero():
    z = np.zeros((1, 2, 2, 2, 2, 2))
    z[..., 0, 0] = 2
    assert np.all(M.accounts(z)["order_defect"] == 1)


def test_invalid_scope_rejected():
    with pytest.raises(ValueError):
        M.question("other", 4, 0)
    with pytest.raises(ValueError):
        M.accounts(np.zeros((24, 2, 2, 2, 2)))
    with pytest.raises(ValueError):
        M.render(["a"], "other", 3, 0)
