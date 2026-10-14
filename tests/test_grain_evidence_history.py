#!/usr/bin/env python3
"""History identity, correlated innovations and revision-aware measurement tests."""

import numpy as np
import pytest

from grain_research.evidence_history import (
    active_records,
    append_innovation,
    fit_source,
    kernels,
    restore_summary,
    score_records,
)
from grain_research.evidence_history_innovation import square_root_append


def row(i, *, event=None, block=None, z=None, kind="new", **extra):
    """An explicitly identified observation; no text-similarity identity guess."""
    return dict(
        record_id=f"r{i}",
        kind=kind,
        event_id=f"e{i}" if event is None else event,
        block_id=f"b{i // 3}" if block is None else block,
        condition=(-1 if i < 3 else 1) if z is None else z,
        **extra,
    )


def model():
    """A declared positive working covariance, not fitted personality truth."""
    return {"mean": np.array([0.1, -0.03]), "components": [0.12, 0.08, 0.3, 0.5, 0.02]}


def test_revision_replaces_claim_but_preserves_event_and_history():
    history = [row(0), row(1)]
    history += [{"record_id": "alias", "kind": "alias", "parent_id": "r0"}]
    history += [row(2, event="e0", block="b0", z=-1, kind="revise", parent_id="r0")]
    state = active_records(history)
    assert {r["record_id"] for r in state["records"]} == {"r1", "r2"}
    assert state["active_events"] == state["observed_events"] == 2
    assert state["stale_derived"] == ["alias"]
    assert len(state["history"]) == 4


def test_new_contradiction_does_not_revise_another_event():
    state = active_records([row(0, value=1), row(1, value=-1)])
    assert len(state["records"]) == 2


def test_cross_event_revision_and_duplicate_identity_rejected():
    with pytest.raises(ValueError):
        active_records([row(0), row(1, kind="revise", parent_id="r0")])
    with pytest.raises(ValueError):
        active_records([row(0), row(0)])


def test_context_drift_with_same_event_rejected():
    with pytest.raises(ValueError):
        active_records([row(0), row(1, event="e0", z=1, kind="repeat")])


def test_numeric_summary_preserves_revision_state():
    initial = [row(0), row(1)]
    packed = {"format": "versioned_numeric_evidence_v1", "history": initial}
    restored = restore_summary(packed)
    change = row(2, event="e0", block="b0", z=-1, kind="revise", parent_id="r0")
    assert active_records(initial + [change]) == active_records(
        restored["history"] + [change]
    )
    with pytest.raises(ValueError):
        restore_summary({"score": 0.0, "variance": 0.3})


def test_exact_duplicate_innovation_is_zero_and_conflict_rejected():
    records = [row(i) for i in range(6)]
    y = np.array([[0.1, 0.2, 0.3, -0.1, 0.4, 0.6]])
    duplicate = dict(records[0], record_id="copy", origin_id="r0")
    a = score_records(model(), y, records)
    b = append_innovation(model(), y, records, y[:, 0], duplicate)
    np.testing.assert_allclose(a["mean"], b["mean"], atol=1e-12)
    np.testing.assert_allclose(a["covariance"], b["covariance"], atol=1e-12)
    assert b["innovation_variance"] < 1e-10
    with pytest.raises(ValueError):
        append_innovation(model(), y, records, [2], duplicate)


def test_correlated_innovation_equals_batch():
    records = [row(i) for i in range(6)]
    rng = np.random.default_rng(84)
    y = rng.normal(size=(9, 7))
    new = row(6, event="e5", block="b1", z=1, kind="repeat")
    incremental = append_innovation(model(), y[:, :-1], records, y[:, -1], new)
    batch = score_records(model(), y, records + [new])
    np.testing.assert_allclose(incremental["mean"], batch["mean"], atol=1e-12)
    np.testing.assert_allclose(
        incremental["covariance"], batch["covariance"], atol=1e-12
    )


def test_retraction_can_increase_uncertainty():
    records = [row(i) for i in range(6)]
    prior = score_records(model(), np.zeros((1, 5)), records[:5])
    full = score_records(model(), np.zeros((1, 6)), records)
    assert np.linalg.eigvalsh(prior["covariance"] - full["covariance"]).min() > -1e-12
    assert np.trace(prior["covariance"]) > np.trace(full["covariance"])


def test_event_average_weights_apply_to_original_records():
    records = [row(i) for i in range(6)]
    records += [row(6, event="e0", block="b0", z=-1, kind="repeat")]
    y = np.arange(14).reshape(2, 7) / 10
    scored = score_records(model(), y, records, "event_only")
    np.testing.assert_allclose(
        scored["mean"], y @ scored["weights"].T + scored["offset"]
    )


def test_source_variances_from_observations_only():
    records = [row(i, z=-1 if (i // 3) % 2 == 0 else 1) for i in range(12)]
    records += [dict(r, record_id=r["record_id"] + "b", kind="repeat") for r in records]
    x, bases = kernels(records)
    rng = np.random.default_rng(83)
    covariance = np.einsum("k,kij->ij", model()["components"], bases)
    y = rng.multivariate_normal(x @ model()["mean"], covariance, size=1000)
    fitted = fit_source(y, records)
    assert np.asarray(fitted["components"]).min() > 0
    assert fitted["source_authors"] == 1000
    np.testing.assert_allclose(fitted["components"], model()["components"], atol=0.12)


def test_square_root_update_handles_nearly_duplicate_reports():
    """Avoid cancellation before squaring a tiny innovation coordinate."""
    m = model()
    m["components"][-1] = 1e-6
    records = [row(i) for i in range(6)]
    rng = np.random.default_rng(931)
    y = rng.normal(size=(7, 6))
    new = row(6, event="e0", block="b0", z=-1, kind="repeat")
    next_y = y[:, 0] + rng.normal(scale=1e-3, size=7)
    root = square_root_append(m, y, records, next_y, new)
    batch = score_records(m, np.c_[y, next_y], records + [new])
    np.testing.assert_allclose(root["mean"], batch["mean"], atol=1e-9)
    np.testing.assert_allclose(root["covariance"], batch["covariance"], atol=1e-10)


def test_square_root_alias_has_zero_new_information():
    records = [row(i) for i in range(6)]
    y = np.arange(6)[None, :] / 10
    alias = dict(records[0], record_id="copy", origin_id="r0")
    actual = square_root_append(model(), y, records, y[:, 0], alias)
    expected = score_records(model(), y, records)
    np.testing.assert_allclose(actual["mean"], expected["mean"], atol=1e-12)
    np.testing.assert_allclose(actual["gain"], 0)


def test_square_root_first_observation():
    """An empty old ledger starts from the frozen source prior."""
    y = np.array([[0.2], [-0.3]])
    actual = square_root_append(model(), np.empty((2, 0)), [], y[:, 0], row(0))
    expected = score_records(model(), y, [row(0)])
    np.testing.assert_allclose(actual["mean"], expected["mean"], atol=1e-12)
    np.testing.assert_allclose(actual["covariance"], expected["covariance"], atol=1e-12)
