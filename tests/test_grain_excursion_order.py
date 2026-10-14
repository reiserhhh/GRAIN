#!/usr/bin/env python3
"""Synthetic and exact finite tests for the conditional order instrument."""

from itertools import product

import numpy as np
import pytest

from grain_research.construction import math_excursion_order as M
from grain_research.construction.math_path_completion import count_matrix

A = (0, 4, 0, 1, 0, 2, 0, 5, 0, 3)
B = (0, 4, 0, 2, 0, 1, 0, 5, 0, 3)


def toy_bundle():
    """Construct a labeled synthetic SOURCE, not a psychological calibration."""
    paths = np.array([[A, A], [B, B]])
    y = np.zeros((2, 2, 5), int)
    y[0, 0, 0] = 1
    y[1, 1, 0] = 1
    return M.compile_bundle(paths, y, [0, 1])


def test_same_counts_different_order_contrast():
    a, b = M.orbit_counts(A), M.orbit_counts(B)
    np.testing.assert_array_equal(count_matrix(A), count_matrix(B))
    np.testing.assert_array_equal(a["numerator"], b["numerator"])
    assert a["orbit_size"] == b["orbit_size"] == 2
    index = (1, 0, 2)
    ea = (a["observed_counts"][index] - a["numerator"][index] / a["denominator"]) / 8
    eb = (b["observed_counts"][index] - b["numerator"][index] / b["denominator"]) / 8
    assert ea == 1 / 16
    assert eb == -1 / 16


def test_all_short_binary_paths_against_enumeration():
    for length in range(3, 8):
        for path in product(range(2), repeat=length):
            account = M.orbit_counts(path)
            words = M.orbit_words(path)
            assert len(words) == account["orbit_size"]
            total = sum(
                (M.triple_counts(w) for w in words), start=np.zeros((8, 8, 8), int)
            )
            np.testing.assert_array_equal(
                total * account["denominator"], account["numerator"] * len(words)
            )
            assert M.adjacent_margin_check(path, account)


def test_loops_and_fixed_boundaries():
    path = (3, 0, 1, 0, 0, 2, 0, 0, 4, 0, 5)
    account = M.orbit_counts(path)
    for word in M.orbit_words(path):
        np.testing.assert_array_equal(count_matrix(word), count_matrix(path))
        np.testing.assert_array_equal(
            np.bincount(word, minlength=8), np.bincount(path, minlength=8)
        )
        assert word[:2] == path[:2] and word[-2:] == path[-2:]
    assert M.adjacent_margin_check(path, account)


def test_pair_additive_field_has_no_order_contrast():
    field = np.arange(64).reshape(8, 8)
    a = M.orbit_counts(A)
    delta = a["observed_counts"] - a["numerator"] / a["denominator"]
    assert np.sum(delta * (field[:, :, None] + field[None])) == 0


def test_unknown_markov_kernels_and_mixture_cancel():
    matrices = [np.ones((8, 8)) + 2 * np.eye(8), np.arange(1, 65).reshape(8, 8)]
    probabilities = []
    for matrix in matrices:
        p = matrix / matrix.sum(1)[:, None]
        values = [np.exp(np.sum(np.log(p) * count_matrix(w))) for w in M.orbit_words(A)]
        assert values[0] == values[1]
        probabilities.append(values)
    mixture = 0.3 * np.array(probabilities[0]) + 0.7 * np.array(probabilities[1])
    assert mixture[0] == mixture[1]


def test_full_profiles_raw_reference_contrast():
    bundle = toy_bundle()
    a, b = M.evaluate(bundle, A), M.evaluate(bundle, B)
    np.testing.assert_allclose(
        a["profile_readings"][0, :, 2], -b["profile_readings"][0, :, 2]
    )
    np.testing.assert_allclose(a["profile_readings"][0, :, :2].sum(-1), 1)
    np.testing.assert_allclose(a["profile_readings"][0, :, 2].sum(-1), 0)
    np.testing.assert_allclose(
        a["scores"][0, :, 0] - a["scores"][0, :, 1], a["scores"][0, :, 2]
    )
    assert a["scores"][0, 0, 2, 0] > 0
    assert b["scores"][0, 0, 2, 0] < 0


def test_unseen_orbit_cells_are_not_filled():
    account = M.evaluate(toy_bundle(), A)
    assert np.isnan(account["scores"][1:]).all()
    assert M.json_safe(account["scores"])[1][0][0][0] is None


def test_singleton_is_structural_zero_not_absent_trait():
    path = (0, 1, 2, 3)
    account = M.orbit_counts(path)
    assert account["orbit_size"] == 1
    np.testing.assert_array_equal(account["numerator"], account["observed_counts"])


@pytest.mark.parametrize("path", [[], [0, 1], [0.0, 1.0, 2.0], [0, 8, 1], [0, -1, 2]])
def test_invalid_paths(path):
    with pytest.raises(ValueError):
        M.orbit_counts(path)


def test_enumeration_bound_does_not_truncate():
    with pytest.raises(ValueError):
        M.orbit_words(A, maximum=1)


def test_record_guard_does_not_score_author_id():
    def record(author):
        return {
            "author_key": author,
            "events": [
                {"event_id": str(i), "timestamp": i, "cell": x} for i, x in enumerate(A)
            ],
        }

    a = M.score_record(toy_bundle(), record("first"))
    b = M.score_record(toy_bundle(), record("second"))
    np.testing.assert_allclose(a["scores"], b["scores"], equal_nan=True)


def test_duplicate_event_guard():
    record = {
        "author_key": "test",
        "events": [
            {"event_id": "same", "timestamp": i, "cell": x} for i, x in enumerate(A)
        ],
    }
    with pytest.raises(ValueError):
        M.score_record(toy_bundle(), record)
