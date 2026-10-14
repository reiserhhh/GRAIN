#!/usr/bin/env python3
"""Finite exact laws, graph closure, author references and cold-input guards."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from grain_research.construction import math_path_completion as M


@pytest.mark.parametrize("states,edges", [(2, 8), (3, 6)])
def test_matrix_tree_best_agrees_with_every_small_word(states, edges):
    for size in range(1, edges + 2):
        groups = M.enumeration_groups(product(range(states), repeat=size), states)
        for (flat, start, end), count in groups.items():
            n = np.array(flat, int).reshape(states, states)
            assert M.word_count(n, start, end) == count


def test_doob_law_and_completion_recurrence_exact():
    path = [0, 1, 0, 2, 3]
    n = M.count_matrix(path, 4)
    local, complete = M.next_laws(n, 0, 3)
    assert local[1] == local[2] == Fraction(1, 2)
    assert complete[1] == 1 and complete[2] == 0
    h = M.success_probability(n, 0, 3)
    assert h == Fraction(1, 2)
    for j in (1, 2):
        remainder = n.copy()
        remainder[0, j] -= 1
        assert complete[j] == local[j] * M.success_probability(remainder, j, 3) / h
    values, count = M.likelihoods(path, 0, 3, n)
    assert count == 4
    assert values[-1].tolist() == [0.5, 1.0, 0.5]


def test_joint_structure_changes_anchor_without_local_first_step_change():
    assert M.toy_results()["first_edge_posterior_local_PC_GC"] == [0.5, 1.0, 1.0]


def test_global_completion_changes_prior_not_per_catalog_prior():
    p = np.array([[[0, 1, 0, 2, 3]], [[0, 1, 2, 3, 3]]])
    y = np.array([[[0] * 5, [1] * 5]] * 2)
    b = M.compile_bundle(p, y, np.array([0, 1]))
    r = {
        "author_key": "unknown",
        "events": [{"event_id": "first", "timestamp": 1, "cell": 0}],
    }
    scores = M.score_record(b, r)["scores"][0, 0, 0, :, 0]
    np.testing.assert_allclose(scores, [0.5, 0.5, 2 / 3])


def test_loops_parallel_edges_empty_graph_and_disconnected_support():
    n = M.count_matrix([0, 0, 1, 0, 1, 0], 3)
    assert M.word_count(n, 0, 0) == len(M.enumerable_words(n, 0, 0))
    assert M.word_count(np.zeros((3, 3), int), 1, 1) == 1
    assert M.word_count(np.zeros((3, 3), int), 1, 2) == 0
    disconnected = np.diag([1, 1, 0])
    assert M.word_count(disconnected, 0, 0) == 0
    assert M.word_count(n, 1, 0) == 0


def test_transition_equivalent_words_have_identical_likelihood():
    a, b = [0, 1, 0, 2, 0], [0, 2, 0, 1, 0]
    n = M.count_matrix(a, 3)
    np.testing.assert_array_equal(n, M.count_matrix(b, 3))
    np.testing.assert_array_equal(
        M.likelihoods(a, 0, 0, n)[0][-1], M.likelihoods(b, 0, 0, n)[0][-1]
    )


def bundle_record():
    """One finite two-bank law and an attested-path-shaped input."""
    paths = np.array([[[0, 1, 0]], [[0, 2, 0]]])
    y = np.array([[[0] * 5, [1] * 5]] * 2)
    bundle = M.compile_bundle(paths, y, np.array([0, 1]))
    record = {
        "author_key": "unknown",
        "events": [
            {"event_id": str(i), "timestamp": i, "cell": s}
            for i, s in enumerate((0, 1, 0))
        ],
    }
    return bundle, record


def test_unsupported_is_null_not_half_and_full_profile_is_preserved():
    bundle, r = bundle_record()
    result = M.score_record(bundle, r)
    assert np.isnan(result["scores"][2, :, -1]).all()
    assert result["support_catalogues"][2, -1].tolist() == [0, 0, 0]
    assert result["profiles"][0, 0, -1, 0, 0] == 1
    assert M.serializable(result["scores"])[2][0][-1] == [[None] * 5] * 3


def test_rename_invariance_and_single_shared_event_not_independent_repetition():
    b, r = bundle_record()
    renamed = r | {
        "author_key": "different",
        "events": [e | {"event_id": "new/" + e["event_id"]} for e in r["events"]],
    }
    np.testing.assert_array_equal(
        M.score_record(b, r)["scores"], M.score_record(b, renamed)["scores"]
    )
    with pytest.raises(ValueError):
        M.score_record(b, r | {"events": r["events"] + [r["events"][0]]})


@pytest.mark.parametrize("change", ["label", "reverse", "bad_cell", "bad_time"])
def test_cold_schema_chronology_and_no_outcomes(change):
    b, r = bundle_record()
    if change == "label":
        r["label"] = 1
    elif change == "reverse":
        r["events"] = r["events"][::-1]
    elif change == "bad_cell":
        r["events"][0]["cell"] = 1.5
    else:
        r["events"][0]["timestamp"] = float("nan")
    with pytest.raises(ValueError):
        M.score_record(b, r)


def test_bareiss_pivot_singular_and_input_rejection():
    assert M.determinant_bareiss([[0, 1], [2, 3]]) == -2
    assert M.determinant_bareiss([[1, 1], [1, 1]]) == 0
    with pytest.raises(ValueError):
        M.word_count(np.array([[0.5]]), 0, 0)
    with pytest.raises(ValueError):
        M.count_matrix([0.5])
