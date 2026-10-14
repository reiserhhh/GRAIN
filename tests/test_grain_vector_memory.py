#!/usr/bin/env python3
"""Fail-able synthetic worlds for actual-prefix memory and cold continuation."""

import copy
import json

import numpy as np
import pytest

from grain_research.construction import math_vector_memory as M
from scripts.run_grain_vector_memory_hm211 import block_means, event_record


def bank():
    """Known response memory whose current profiles alone discard the next update."""
    codes = np.repeat(np.arange(32), 2)
    signs = np.tile([-1.0, 1.0], 32)
    z = np.empty((64, 3, 1))
    z[:, 0, 0] = signs
    z[:, 1, 0] = signs
    z[:, 2, 0] = signs * (2 * M.BITS[codes, 0] - 1)
    return M.compile_bank(z, codes, 1.0, np.array([f"source{i}" for i in range(64)]))


def input_record(t, frame="person"):
    """Eight distinct addressed events per declared acquisition block."""
    return event_record(np.ones((2, 12, 1)), np.zeros((2, 12), int), frame, t)


def prefix():
    """A reachable synthetic prefix, not a manually selected component posterior."""
    b = bank()
    state = M.initialize(b, "person")
    for t in (0, 1):
        state = M.ingest(b, state, input_record(t))
    return b, state


def test_bank_full_support_and_source_binding():
    b = bank()
    with pytest.raises(ValueError, match="Unsupported"):
        M.compile_bank(b["prototypes"][2:], b["codes"][2:], 1.0, b["source_ids"][2:])
    ids = b["source_ids"].copy()
    ids[1] = ids[0]
    with pytest.raises(ValueError):
        M.compile_bank(b["prototypes"], b["codes"], 1.0, ids)


def test_reachable_prefix_current_profile_equal_next_reading_different():
    b = bank()
    r = M.experiment(b, np.ones((1, 3, 1)))
    np.testing.assert_allclose(
        r["before_profile"], r["before_reset_profile"], atol=1e-14
    )
    assert r["prefix_memory_TV"][0] > 0.4
    assert r["delta_targets"][0, 0] > 0.3
    np.testing.assert_allclose(r["delta_targets"], r["residual_identity"], atol=1e-14)


def test_reset_preserves_all32_profiles_not_only_five_marginals():
    b = bank()
    w = M.normalize(np.arange(64, dtype=float)[None] / 9)
    r = M.reset_weights(b, w)
    np.testing.assert_allclose(M.readout(b, w)["profile"], M.readout(b, r)["profile"])
    for g in range(32):
        assert r[0, 2 * g] == r[0, 2 * g + 1]


def test_identical_within_profile_future_kernel_is_exact_null():
    b = bank()
    z = b["prototypes"].copy()
    z[:, 2, 0] = b["codes"] / 32
    b = M.compile_bank(z, b["codes"], 1.0, b["source_ids"])
    r = M.experiment(b, np.ones((1, 3, 1)))
    assert r["prefix_memory_TV"][0] > 0.4
    np.testing.assert_allclose(r["delta_targets"], 0, atol=1e-14)
    np.testing.assert_allclose(r["observation_log_evidence_ratio"], 0, atol=1e-14)


def test_source_independent_next_kernel_is_exact_null():
    b = bank()
    w = M.normalize(np.arange(64, dtype=float)[None] / 9)
    r = M.compare_next(b, w, np.zeros_like(w))
    np.testing.assert_allclose(r["delta_targets"], 0, atol=1e-14)


def test_sequential_json_checkpoint_matches_batch_without_prefix_vectors():
    b, state = prefix()
    reset = M.reset_memory(b, state)
    checkpoint = json.loads(json.dumps({"fine": state, "reset": reset}))
    # The checkpoint retains hashes/addresses and weights, not previous vectors.
    assert all("vector" not in e for e in state["event_ledger"])
    readings = [
        M.describe(b, M.ingest(b, checkpoint[k], input_record(2)))["target_readings"]
        for k in ("fine", "reset")
    ]
    delta = np.array(list(readings[0].values())) - np.array(list(readings[1].values()))
    np.testing.assert_allclose(
        delta, M.experiment(b, np.ones((1, 3, 1)))["delta_targets"][0]
    )
    k = sum(M.likelihood(b, np.ones((1, 1)), t)[0] for t in (0, 1))
    np.testing.assert_allclose(state["log_weights"], M.normalize(M.prior(b) + k))


def test_duplicate_event_ids_are_not_repeated_evidence():
    b, state = prefix()
    r = input_record(2)
    r["events"][0]["event_id"] = state["event_ledger"][0]["event_id"]
    with pytest.raises(ValueError, match="Repeated"):
        M.ingest(b, state, r)


def test_labels_wrong_frame_and_wrong_block_are_rejected():
    b, state = prefix()
    r = input_record(2)
    r["label"] = 1
    with pytest.raises(ValueError):
        M.ingest(b, state, r)
    with pytest.raises(ValueError):
        M.ingest(b, state, input_record(2, "other"))
    with pytest.raises(ValueError):
        M.ingest(b, state, input_record(1))
    r = input_record(2)
    r["events"][0]["answer"] = "yes"
    with pytest.raises(ValueError):
        M.ingest(b, state, r)


def test_changed_bank_cannot_resume_same_memory():
    b, state = prefix()
    other = M.compile_bank(b["prototypes"], b["codes"], 2.0, b["source_ids"])
    with pytest.raises(ValueError, match="bank"):
        M.ingest(other, state, input_record(2))


def test_completed_history_cannot_append_a_fourth_block():
    b, state = prefix()
    state = M.ingest(b, state, input_record(2))
    with pytest.raises(ValueError):
        M.ingest(b, state, input_record(2))


def test_checkpoint_ledger_tampering_and_weights_are_rejected():
    b, state = prefix()
    modified = copy.deepcopy(state)
    modified["event_ledger"][1]["slot"] = modified["event_ledger"][0]["slot"]
    with pytest.raises(ValueError):
        M.validate_state(b, modified)
    modified = copy.deepcopy(state)
    modified["log_weights"][0] += 1
    with pytest.raises(ValueError):
        M.validate_state(b, modified)


def test_partition_uses_every_original_event_once_with_balanced_views():
    vectors = np.arange(24, dtype=float).reshape(1, 2, 12, 1)
    blocks = block_means(vectors)
    np.testing.assert_allclose(blocks.ravel(), [7.5, 11.5, 15.5])
    np.testing.assert_allclose(blocks.mean(1), vectors.mean((1, 2)))
    ids = []
    for t in range(3):
        r = event_record(vectors[0], np.zeros((2, 12), int), "p", t)
        assert sum(e["view"] == 0 for e in r["events"]) == 4
        ids.extend(e["event_id"] for e in r["events"])
    assert len(ids) == len(set(ids)) == 24
