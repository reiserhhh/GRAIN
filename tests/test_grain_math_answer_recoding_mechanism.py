#!/usr/bin/env python3
"""Synthetic recoding identities and failure controls without neural model calls."""

import numpy as np
import pytest

from grain_research.construction import math_answer_recoding_mechanism as M
from grain_research.construction.math_relational_cut import call_allowed


@pytest.mark.parametrize("x", [0.0, 0.2, 0.5, 1.0])
def test_coordinate_identity_and_involution(x):
    assert M.decode(x, 1) == x
    assert M.decode(M.decode(x, -1), -1) == pytest.approx(x)
    with pytest.raises(ValueError):
        M.decode(x, 0)


def test_decoded_log_odds_and_candidate_mass():
    n = {
        "conditional_score": 0.8,
        "candidate_logits": [0.0] * 5 + [2.0] * 5,
        "candidate_mass": 0.99,
    }
    a, b = M.readout(n, 1), M.readout(n, -1)
    assert a["raw_log_odds"] == b["raw_log_odds"] == 2
    assert b["semantic_log_odds"] == -2
    assert b["semantic_score"] == pytest.approx(0.2)
    assert a["candidate_mass"] == b["candidate_mass"]


@pytest.mark.parametrize(
    "a,b,even,odd", [(2, 2, 2, 0), (2, -2, 0, 2), (3, 1, 2, 1), (0, 0, 0, 0)]
)
def test_raw_semantic_mixed_and_null_operation(a, b, even, odd):
    assert M.orbit(a, b) == {"raw_even": even, "raw_odd": odd}
    # The finite decomposition is reconstructible even when both components coexist.
    assert even + odd == a and even - odd == b


def row(codec=1, person="A", direction=0):
    """Provide a tiny declared suffix with two code slots and one actor slot."""
    return {
        "role": "application",
        "content": "e",
        "query": "e",
        "class": "own",
        "body": "body",
        "rendered": "body " + (M.FORWARD_CUE if codec == 1 else M.REVERSE_CUE),
        "person": person,
        "direction": direction,
        "codec": codec,
        "partition": {"suffix": [2, 3, 4, 5]},
        "input_ids": [
            6,
            6,
            1 if person == "A" else 2,
            0 if codec == 1 else 9,
            8,
            9 if codec == 1 else 0,
        ],
        "cue_slots": [1, 3],
        "actor_slot": 0,
        "standard_reference_uid": 2,
    }


def test_exact_two_cue_recoding_proof():
    a, b = row(), row(-1)
    assert M.code_pair(a, b)["changed_slots"] == [1, 3]
    with pytest.raises(ValueError):
        M.code_pair(a, b | {"input_ids": [6, 5, 1, 9, 8, 0]})
    with pytest.raises(ValueError):
        M.code_pair(a, b | {"body": "different"})


def test_mapping_allows_only_named_code_and_actor_slots():
    a, b = row(), row(-1, "B")
    with pytest.raises(ValueError):
        M.prove_mapping(a, b)
    assert M.prove_mapping(a, b, allow_actor=True)["changed_slots"] == [0, 1, 3]
    with pytest.raises(ValueError):
        M.prove_mapping(a, row() | {"input_ids": [6, 6, 1, 0, 7, 9]})
    with pytest.raises(ValueError):
        M.prove_mapping(a, row() | {"query": "n"})
    assert M.prove_mapping(a, row(direction=1))["changed_slots"] == []


@pytest.mark.parametrize("arm", M.ARMS)
def test_all_fixed_patches_and_random_norms(arm):
    r = row()
    current = np.zeros((6, 3), np.float32)
    donor = np.ones((6, 3), np.float32)
    src = {
        M.source_key("e", p, c): np.ones((4, 3)) * (c if p == "A" else 2 * c)
        for p in ("A", "B")
        for c in (1, -1)
    }
    e = {"arm": arm, "donor_positions": [2, 3, 4, 5]}
    positions, values = M.patch_values(e, r, current, donor, src, 5301)
    after = current.copy()
    after[positions] = values
    np.testing.assert_array_equal(after[:2], current[:2])
    if arm in ("exact_suffix", "source_forward"):
        np.testing.assert_array_equal(values, np.ones((4, 3)))
    if arm == "source_reverse":
        np.testing.assert_array_equal(values, -np.ones((4, 3)))
    if arm == "wrong_person":
        np.testing.assert_array_equal(values, 2 * np.ones((4, 3)))
    if arm == "random":
        np.testing.assert_allclose(
            np.linalg.norm(values, axis=1), np.sqrt(3), rtol=1e-6
        )
    if arm == "noop":
        np.testing.assert_array_equal(values, 0)


def test_receiver_code_selects_wrong_person_and_random_reference():
    r, current = row(-1), np.zeros((6, 2), np.float32)
    src = {
        M.source_key("e", p, c): np.ones((4, 2)) * (c if p == "A" else 2 * c)
        for p in ("A", "B")
        for c in (1, -1)
    }
    _, value = M.patch_values({"arm": "wrong_person"}, r, current, current, src, 5301)
    np.testing.assert_array_equal(value, -2)
    with pytest.raises(ValueError):
        M.patch_values({"arm": "unknown"}, r, current, current, src, 5301)


def test_source_seal_refuses_application_counts_or_inflight(tmp_path):
    from scripts.run_grain_math_answer_recoding_mechanism import seal_source

    with pytest.raises(ValueError, match="Exactly24"):
        seal_source(tmp_path, {}, {"forward_calls": 25, "in_flight": None})
    with pytest.raises(ValueError, match="Exactly24"):
        seal_source(tmp_path, {}, {"forward_calls": 24, "in_flight": 23})


def test_unknown_or_terminal_attempts_never_repeat():
    spec = {"max_forwards": 648, "max_model_seconds": 600}
    status = {"status": "COLLECTING", "in_flight": None, "attempted_forward_calls": 647}
    assert call_allowed(status, spec, 599)
    for change in (
        {"status": "STOPPED"},
        {"status": "COLLECTED"},
        {"in_flight": 2},
        {"attempted_forward_calls": 648},
    ):
        assert not call_allowed(status | change, spec, 599)
    assert not call_allowed(status, spec, 600)
