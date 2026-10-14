#!/usr/bin/env python3
"""Real ledger operations, support and covariance checks replace placeholder flags."""

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f13_history_contract import score_history

MODEL = {"mean": [0.2, 0.3], "components": [0.4, 0.2, 0.1, 0.08, 0.03]}


def test_deterministic_nonzero_mean_is_supported():
    """Support applies to y-Xmu, not uncentered y; inconsistent data are rejected."""
    model = {"mean": [2.0, 1.0], "components": [0.0, 0.0, 0.0, 0.0, 0.0]}
    out = score_history([record("a", "e1", 1, 3.0)], model, [1, 1])
    assert out["score"] == 3.0 and out["variance"] == 0.0
    with pytest.raises(ValueError, match="outside specified Gaussian support"):
        score_history([record("a", "e1", 1, 3.1)], model, [1, 1])


def record(identity, event, z, value, kind="new", **kwargs):
    """One explicitly linked numeric observation for a single author."""
    return {
        "record_id": identity,
        "event_id": event,
        "condition": z,
        "value": value,
        "block_id": "block",
        "author_id": "author",
        "kind": kind,
        **kwargs,
    }


def test_dense_conditioning_and_components():
    """Compare ledger output against independent Gaussian precision algebra."""
    history = [record("a", "e1", -1, 0.4), record("b", "e2", 1, 0.8)]
    out = score_history(history, MODEL, [0, 2])
    x = np.array([[1, -1], [1, 1.0]])
    v = np.diag([0.4, 0.2])
    r = 0.1 * np.ones((2, 2)) + 0.11 * np.eye(2)
    covariance = np.linalg.inv(np.linalg.inv(v) + x.T @ np.linalg.solve(r, x))
    mean = covariance @ (
        np.linalg.solve(v, MODEL["mean"]) + x.T @ np.linalg.solve(r, [0.4, 0.8])
    )
    np.testing.assert_allclose(out["score"], 2 * mean[1])
    np.testing.assert_allclose(out["variance"], 4 * covariance[1, 1])
    np.testing.assert_allclose(
        sum(out["variance_components"].values()), out["variance"]
    )


def test_alias_does_not_shrink_and_duplicate_rejected():
    """A duplicate cannot create independent precision."""
    first = record("a", "e1", -1, 0.4)
    alias = {
        "record_id": "alias",
        "author_id": "author",
        "kind": "alias",
        "parent_id": "a",
    }
    assert score_history([first], MODEL, [1, 0]) == score_history(
        [first, alias], MODEL, [1, 0]
    )
    with pytest.raises(ValueError):
        score_history([first, first], MODEL, [1, 0])
    with pytest.raises(ValueError):
        score_history([first, record("b", "e1", -1, 0.4)], MODEL, [1, 0])


def test_revision_withdrawal_and_stale_summary():
    """A correction replaces its parent and withdrawal removes active support."""
    first = record("a", "e1", -1, 0.4)
    summary = {
        "record_id": "summary",
        "author_id": "author",
        "kind": "summary",
        "parents": ["a"],
    }
    revised = record("b", "e1", -1, 0.9, "revise", parent_id="a")
    out = score_history([first, summary, revised], MODEL, [1, 0])
    equivalent = score_history([record("b", "e1", -1, 0.9)], MODEL, [1, 0])
    assert (
        out["score"] == equivalent["score"]
        and out["variance"] == equivalent["variance"]
    )
    assert out["stale_derived"] == ["summary"]
    retract = {
        "record_id": "withdraw",
        "author_id": "author",
        "kind": "retract",
        "parent_id": "b",
    }
    empty = score_history([first, revised, retract], MODEL, [1, 0])
    assert empty["status"] == "model_borrowed_no_active_evidence"
    assert empty["score"] == 0.2 and empty["variance"] == 0.4


def test_null_support_and_invalid_inputs():
    """No model is null; model support is not manufactured empirical coverage."""
    assert score_history([], None, [1, 0])["score"] is None
    one = [record("a", "e1", 1, 0.4)]
    assert (
        score_history(one, MODEL, [0, 2])["status"] == "model_borrowed_partial_support"
    )
    assert score_history(one, MODEL, [1, 1])["empirical_target_identified"]
    with pytest.raises(ValueError):
        score_history(
            one + [record("b", "e2", -1, 0.4) | {"author_id": "other"}], MODEL, [1, 0]
        )
    with pytest.raises(ValueError):
        score_history(one, MODEL | {"components": [-0.4, 0.2, 0.1, 0.08, 0.03]}, [1, 0])
