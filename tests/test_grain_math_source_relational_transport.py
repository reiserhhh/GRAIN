#!/usr/bin/env python3
"""Synthetic source-transport tests without neural or private data dependencies."""

from __future__ import annotations

import numpy as np
import pytest

from grain_research.construction import math_source_relational_transport as M
from grain_research.construction.math_relational_cut import call_allowed


def fixture():
    """Return a tiny complete cut and a finite source change pair."""
    current = np.arange(40, dtype=np.float32).reshape(8, 5)
    donor = current + 2
    sectors = M.source_sectors(np.ones((3, 5)), 3 * np.ones((3, 5)))
    entry = {
        "partition": {"suffix": [5, 6, 7]},
        "person": "A",
        "direction": 0,
        "pair": 24,
        "arm": "source_full",
    }
    return current, donor, sectors, entry


def test_shared_bound_retained_and_exact_reconstruction():
    _, _, s, _ = fixture()
    np.testing.assert_array_equal(s["shared"] + s["bound"], s["A"])
    np.testing.assert_array_equal(s["shared"] - s["bound"], s["B"])
    assert np.linalg.norm(s["shared"]) > 0


@pytest.mark.parametrize("arm", M.ARMS)
def test_all_arms_change_only_suffix(arm):
    c, d, s, e = fixture()
    e["arm"] = arm
    p, v = M.patch_values(e, c, d, s, 5101)
    after = c.copy()
    after[p] = v
    np.testing.assert_array_equal(after[:5], c[:5])
    if arm == "exact_suffix":
        np.testing.assert_array_equal(v, d[p])
    if arm == "noop":
        np.testing.assert_array_equal(after, c)


def test_wrong_person_is_not_correct_person_delta():
    _, _, s, _ = fixture()
    correct = M.source_delta(s, "A", 0, "source_full", 24, 5101)
    wrong = M.source_delta(s, "A", 0, "wrong_person", 24, 5101)
    np.testing.assert_array_equal(wrong, s["B"])
    assert not np.array_equal(correct, wrong)


def test_random_source_only_norms_inverse_and_zero():
    _, _, s, _ = fixture()
    d = M.source_delta(s, "A", 0, "random", 24, 5101)
    inverse = M.source_delta(s, "A", 1, "random", 25, 5101)
    np.testing.assert_array_equal(inverse, -d)
    np.testing.assert_allclose(
        np.linalg.norm(d, axis=1), np.linalg.norm(s["A"], axis=1)
    )
    z = M.source_sectors(np.zeros((3, 5)), np.zeros((3, 5)))
    np.testing.assert_array_equal(
        M.source_delta(z, "A", 0, "random", 24, 5101), np.zeros((3, 5))
    )


def test_actor_sign_does_not_delete_shared_sector():
    _, _, s, _ = fixture()
    for person, sign in (("A", 1), ("B", -1)):
        np.testing.assert_array_equal(
            M.source_delta(s, person, 0, "source_bound", 24, 5101), sign * s["bound"]
        )
        np.testing.assert_array_equal(
            M.source_delta(s, person, 0, "source_shared", 24, 5101), s["shared"]
        )


def test_exact_query_alignment_keeps_absolute_position_shift():
    row = {"input_ids": [7, 8, 9, 10], "person": "A", "query": "j", "content": "j"}
    app = {**row, "input_ids": [1, 2, 3, 4, 9, 10]}
    proof = M.prove_transport(
        row, {"partition": {"suffix": [2, 3]}}, app, {"partition": {"suffix": [4, 5]}}
    )
    assert (
        proof["relative_slots"] == [0, 1]
        and proof["source_positions"] != proof["application_positions"]
    )
    app["person"] = "B"
    with pytest.raises(ValueError, match="person"):
        M.prove_transport(
            row,
            {"partition": {"suffix": [2, 3]}},
            app,
            {"partition": {"suffix": [4, 5]}},
        )


def test_actor_slot_and_extra_query_difference():
    assert M.actor_slot_proof([9, 1, 7], [9, 2, 7])["intentional_actor_slot"] == 1
    for b in ([9, 1, 7], [8, 2, 7], [9, 2]):
        with pytest.raises(ValueError):
            M.actor_slot_proof([9, 1, 7], b)


def test_token_mismatch_and_noncontiguous_suffix_fail():
    r = {"input_ids": [1, 2, 3, 4], "person": "A", "query": "j", "content": "j"}
    c = {"partition": {"suffix": [2, 3]}}
    with pytest.raises(ValueError, match="token"):
        M.prove_transport(r, c, {**r, "input_ids": [1, 2, 3, 9]}, c)
    with pytest.raises(ValueError, match="contiguous"):
        M.suffix_ids(r, {"partition": {"suffix": [1, 3]}})


def test_null_and_nonlinear_interaction_worlds():
    assert M.four_cells(3, 3, 3, 3)["interaction"] == 0
    assert M.four_cells(1, 3, 4, 6)["interaction"] == 0
    # State components add; a square readout still has a mixed term.
    x = M.four_cells(0, 1, 4, 9)
    assert x["interaction"] == 4 and x["reconstruction_error"] == 0


def test_invalid_sectors_actors_and_observations_fail():
    with pytest.raises(ValueError):
        M.source_sectors(np.ones((2, 3)), np.ones((3, 3)))
    _, _, s, _ = fixture()
    with pytest.raises(ValueError):
        M.source_delta(s, "C", 0, "source_full", 24, 5101)
    with pytest.raises(ValueError):
        M.four_cells(1, 2, 3, float("nan"))


def test_uncertain_terminal_and_resource_limits():
    s = {"in_flight": None, "status": "PREPARED", "attempted_forward_calls": 0}
    spec = {"max_forwards": 168, "max_model_seconds": 600}
    assert call_allowed(s, spec, 599)
    assert not call_allowed(s, spec, 600)
    assert not call_allowed(s | {"in_flight": 0}, spec, 0)
    assert not call_allowed(s | {"status": "STOPPED"}, spec, 0)
    assert not call_allowed(s | {"attempted_forward_calls": 168}, spec, 0)
