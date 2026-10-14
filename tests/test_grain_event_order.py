#!/usr/bin/env python3
"""Synthetic observation contracts for the single registered order operation."""

import copy

import numpy as np
import pytest

from grain_research.construction import math_event_order as M

ORDER = [0, 1, 2, 3, 10, 9, 8, 7, 6, 5, 4, 11]


def sample():
    """Create finite synthetic events, not simulated psychological subjects."""
    rng = np.random.default_rng(245)
    return rng.normal(size=(8, 3)), rng.normal(size=(8, 3)), rng.normal(size=(12, 3))


def test_permutation_preserves_mean_and_boundary():
    _, _, b = sample()
    p = M.validate_order(ORDER)
    np.testing.assert_array_equal(b[:4], b[p][:4])
    np.testing.assert_array_equal(b[-1], b[p][-1])
    np.testing.assert_allclose(b.mean(0), b[p].mean(0), atol=1e-15)


@pytest.mark.parametrize(
    "order", [ORDER[:-1], [0] * 12, np.roll(ORDER, 1), np.array(ORDER, float)]
)
def test_bad_order_rejected(order):
    with pytest.raises(ValueError):
        M.validate_order(order)


def test_context_third_tensor_and_lower_levels():
    a, c, b = sample()
    out = M.ordered_interaction(a, c, b, ORDER)
    np.testing.assert_allclose(out["mixed"][0], 0, atol=1e-13)
    np.testing.assert_allclose(out["mixed"][1], 0, atol=1e-13)
    np.testing.assert_allclose(out["mixed"][2], out["factorized_third"], atol=1e-12)
    assert np.linalg.norm(out["factorized_third"]) > 0


def test_second_direct_matches_chen():
    _, _, b = sample()
    np.testing.assert_allclose(
        M.signature_levels(b)[1], M.second_level_direct(b), atol=1e-13
    )


def test_identity_and_same_focal_are_zero():
    a, c, b = sample()
    for arms, order in (((a, c), list(range(12))), ((a, a), ORDER)):
        out = M.ordered_interaction(*arms, b, order)
        np.testing.assert_array_equal(out["mixed"][2], 0)


def test_arbitrary_additive_event_windows_cancel():
    a, c, b = sample()
    for width in range(1, 6):

        def potential(events, width=width):
            return sum(
                np.sin(events[i : i + width].ravel() @ np.arange(width * 3))
                for i in range(len(events) - width + 1)
            )

        actions = [
            potential(np.vstack((a, bg))) - potential(np.vstack((c, bg)))
            for bg in (b, b[ORDER])
        ]
        np.testing.assert_allclose(actions[0], actions[1], atol=1e-13)


def payload():
    """Make a qualified numeric cold request with no scores or questionnaire."""
    a, _, b = sample()
    return {
        "author": "synthetic",
        "order": ORDER,
        "focal": [
            {"event_id": f"a{i}", "owner": "synthetic", "vector": v.tolist()}
            for i, v in enumerate(a)
        ],
        "background": [
            {"event_id": f"b{i}", "owner": "synthetic", "vector": v.tolist()}
            for i, v in enumerate(b)
        ],
    }


def test_cold_interface_retains_order_without_target():
    result = M.cold_account(payload())
    assert result["target_score"] is None
    np.testing.assert_allclose(
        result["levels_original"][0], result["levels_reordered"][0], atol=1e-13
    )
    assert not np.allclose(result["levels_original"][1], result["levels_reordered"][1])


@pytest.mark.parametrize("fault", ["answer", "owner", "duplicate", "event_score"])
def test_cold_interface_rejects_injected_truth_or_identity(fault):
    p = copy.deepcopy(payload())
    if fault == "answer":
        p["answer"] = 0.9
    elif fault == "owner":
        p["background"][0]["owner"] = "other"
    elif fault == "duplicate":
        p["background"][0]["event_id"] = p["focal"][0]["event_id"]
    else:
        p["focal"][0]["score"] = 0.9
    with pytest.raises(ValueError):
        M.cold_account(p)
