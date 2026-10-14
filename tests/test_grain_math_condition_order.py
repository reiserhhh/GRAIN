#!/usr/bin/env python3
"""Finite synthetic tests of order, compilation and cold input boundaries."""

import copy

import numpy as np
import pytest

from grain_research.construction import math_condition_order as M


def world():
    """Create independent, balanced c/x with an intentionally mixed observation."""
    c = np.array([-1, -1, 1, 1.0])
    x = np.array([-1, 1, -1, 1.0])
    return c, x, np.full(4, 0.25)


def test_choice_mixes_into_within_after_compression():
    c, x, p = world()
    account = M.order_account((c + x)[:, None], p, c)
    assert account["c_visibility"] == pytest.approx(0.5)
    assert account["choice_to_within_norm2"] == pytest.approx(0.25)
    assert account["E_commutator_Frobenius"] == pytest.approx(np.sqrt(0.5))
    assert account["EPc_identity_error"] < 1e-12


def test_condition_anchor_not_multiplication_closure():
    c, x, p = world()
    anchor = M.order_account(np.column_stack((c, x)), p, c)
    closed = M.order_account(np.column_stack((c, x, c * x)), p, c)
    assert anchor["E_commutator_Frobenius"] < 1e-12
    assert anchor["M_span_escape_Frobenius"] > 0.5
    assert closed["M_commutator_Frobenius"] < 1e-12


def test_compiled_function_handles_unseen_values_without_lookup():
    _, _, p = world()
    b = np.array([[2, 4], [3, 6], [4, 8], [5, 10.0]])
    t = (3 * b[:, 0] - 7)[:, None]
    compiled = M.compile_function(b, t, p)
    assert compiled["rank"] == 1
    actual = np.array([[6, 12.0]]) @ compiled["beta"] + compiled["offset"]
    assert actual[0, 0] == pytest.approx(11)


@pytest.mark.parametrize("c", [None, -1, 1])
def test_cold_scores_no_application_truth_or_batch_refitting(c):
    calibration = {
        "observer_id": "frozen",
        "grain_center": [0] * 4,
        "grain_scale": [1] * 4,
        "query_center": [0.5] * 6,
        "query_scale": [1] * 6,
        "feature_center": [0] * 49,
        "feature_scale": [1] * 49,
    }
    dimensions = (10, 49, 50, 99)
    bundle = {
        "observation": calibration,
        "functions": {
            name: {"beta": np.ones((d, 4)), "offset": np.zeros(4)}
            for name, d in zip(M.ARMS, dimensions, strict=True)
        },
    }
    record = {
        "event_id": "unseen",
        "observer_id": "frozen",
        "grain": [0.1] * 4,
        "query": [0.6] * 6,
        "condition": c,
    }
    result = M.cold_score(bundle, record)
    assert result == M.cold_score(copy.deepcopy(bundle), copy.deepcopy(record))
    assert result["arms"]["joint49"]["status"] == "OBSERVED_FUNCTION"
    if c is None:
        assert result["arms"]["anchor_joint50"]["scores"] is None
    else:
        assert result["arms"]["anchor_joint50"]["scores"] is not None
    with pytest.raises(ValueError, match="cold inputs"):
        M.cold_score(bundle, record | {"truth": 1})


def test_unbalanced_reference_cannot_use_balanced_identity():
    c, x, _ = world()
    with pytest.raises(ValueError, match="balanced"):
        M.order_account(x[:, None], [0.1, 0.1, 0.4, 0.4], c)
