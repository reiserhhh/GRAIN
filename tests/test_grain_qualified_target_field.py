#!/usr/bin/env python3
"""Synthetic tests for complete qualified action and composition accounts."""

import numpy as np
import pytest

from grain_research.construction import math_qualified_target_field as M


def test_complete_material_support_and_word_bags():
    rows = M.materials()
    assert len(rows) == 32
    assert len(M.questions()) == 12
    assert {r["role"] for r in rows} == {"source", "held_expression"}
    for r in rows:
        for a, b in r["intervals"].values():
            assert 0 <= a < b <= len(r["body"])


def test_full_character_inverse_and_parseval():
    x = np.random.default_rng(9).normal(size=(2, 2, 2, 6, 3))
    c = M.characters(x)
    np.testing.assert_allclose(M.reconstruct(c), x, atol=1e-14)
    np.testing.assert_allclose(np.mean(x * x, axis=(0, 1, 2)), np.sum(c * c, axis=0))


def test_qualified_masks_not_plain_action_or_query_bias():
    e, n, p = M.SIGNS.T
    x = (3 + 2 * e + 4 * n + 5 * p + 7 * e * p + 11 * n * p + 13 * e * n * p).reshape(
        2, 2, 2
    )
    np.testing.assert_allclose(M.characters(x), [3, 2, 4, 0, 5, 7, 11, 13])


def test_constant_prompt_field_has_zero_qualification():
    c = M.characters(np.full((2, 2, 2, 7), 2.0))
    assert np.max(abs(c[1:])) == 0


def test_composition_retains_nonlinear_and_nonlocal_parts():
    b, e, n, both, whole = map(np.array, ([1, 2], [3, 4], [4, 6], [9, 15], [12, 14]))
    parts = M.composition(b, e, n, both, whole)
    np.testing.assert_array_equal(
        sum(parts[k] for k in ("e", "n", "interaction", "nonlocal_remainder")),
        whole - b,
    )
    assert np.any(parts["interaction"] != 0)
    assert np.any(parts["nonlocal_remainder"] != 0)


def test_signed_transport_not_accuracy_and_undefined_reference():
    assert M.action_account([-1, -2], [1, 2])["signed_projection"] == -1
    assert M.action_account([1, 2], [0, 0])["signed_projection"] is None


def test_cross_wording_support_keeps_negative_agreement():
    u = np.array([[1, 0], [-1, 0], [0, 1]])
    assert M.cross_wording_support(u @ u.T) == pytest.approx(-2 / 3)


def test_incomplete_nonfinite_or_nonpsd_rejected():
    with pytest.raises(ValueError):
        M.characters(np.zeros((2, 2, 1)))
    with pytest.raises(ValueError):
        M.characters(np.full((2, 2, 2), np.nan))
    with pytest.raises(ValueError):
        M.cross_wording_support(-np.eye(3))


def test_two_local_donors_are_not_a_fabricated_joint_corner():
    from scripts.run_grain_qualified_target_field_hm155 import donors_for

    rows = M.materials()
    indexed = {(r["bank"], r["condition"], r["order"], r["e"], r["n"]): r for r in rows}
    row = {**rows[0], "scopes": {M.AXES[0]: [0, 1], M.AXES[1]: [2, 3]}}
    natural = {
        r["id"]: {
            "history_key": np.full((1, 1, 5, 1), r["id"], float),
            "history_value": np.full((1, 1, 5, 1), 10 + r["id"], float),
        }
        for r in rows
    }
    k, v = donors_for(row, "both_local", natural, indexed)
    np.testing.assert_array_equal(k.ravel(), [2, 2, 1, 1, 0])
    np.testing.assert_array_equal(v.ravel(), [12, 12, 11, 11, 10])
    whole_k, _ = donors_for(row, "both_whole", natural, indexed)
    assert np.all(whole_k == 3)
    assert np.all(natural[0]["history_key"] == 0)
