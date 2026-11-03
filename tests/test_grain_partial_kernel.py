#!/usr/bin/env python3
"""Collapsed masked-state sampling,exact controls and no hidden-bit/ID leakage."""

from itertools import product

import numpy as np
import pytest
from scipy.special import logsumexp
from test_grain_conditional_memory import bundle, record

from grain_research.construction import math_author_belief as B
from grain_research.construction import math_conditional_memory as C
from grain_research.construction import math_partial_kernel as M


def test_full_masks_match_all_exact_prefixes_and_count_statistics():
    b, path = bundle(), np.array([0, 5, 2, 7, 2, 2])
    result = M.filter_masks(b, C.observation_masks(path, "full"), 16, 1)
    exact = M.exact_state(b, path)
    np.testing.assert_allclose(
        result["log_evidence"], exact["log_evidence"], atol=1e-13
    )
    np.testing.assert_array_equal(result["count0"], np.repeat(exact["count0"], 16, 1))
    np.testing.assert_array_equal(result["count"], np.repeat(exact["count"], 16, 1))


def test_shared_kernel_integral_is_not_product_of_row_means():
    b = {"alpha0": np.ones((1, 2)), "alpha": np.ones((1, 2, 2))}
    path = np.array([0, 0, 0])
    masks = np.array([[True, False]] * 3)
    result = M.filter_masks(b, masks, 8, 7)
    assert np.exp(result["log_evidence"][-1, 0]) == pytest.approx(1 / 6)
    assert np.exp(B.component_log_likelihood(b, [path], False)[0]) == pytest.approx(
        1 / 8
    )


def test_completely_hidden_observation_retains_normalized_evidence():
    result = M.filter_masks(bundle(), np.ones((8, 8), bool), 32, 5)
    np.testing.assert_allclose(result["log_evidence"], 0, atol=1e-13)
    np.testing.assert_array_equal(result["count0"].sum(-1), 1)
    np.testing.assert_array_equal(result["count"].sum((-2, -1)), 7)


@pytest.mark.parametrize("view", ["condition", "expression"])
def test_hidden_actual_bits_never_change_particle_state(view):
    b, path = bundle(), np.array([0, 5, 2, 7, 1])
    hidden_change = path ^ (3 if view == "condition" else 4)
    first = M.filter_masks(b, C.observation_masks(path, view), 16, 8)
    second = M.filter_masks(b, C.observation_masks(hidden_change, view), 16, 8)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_prefix_values_do_not_use_future_observations_or_seeds():
    b, path = bundle(), np.array([0, 5, 2, 7, 1])
    masks = C.observation_masks(path, "condition")
    all_rows = M.filter_masks(b, masks, 16, 3)
    for length in (1, 2, 3):
        prefix = M.filter_masks(b, masks[:length], 16, 3)
        np.testing.assert_array_equal(
            prefix["log_evidence"], all_rows["log_evidence"][: length + 1]
        )


def test_exact_late_replays_shared_full_and_fresh_observers():
    b, early, next_path = bundle(), np.array([0, 5, 2]), np.array([7, 1, 0])
    first = B.observe(b, early, True, True)
    weights = first["carry"]["logw"]
    row = M.late(b, next_path, weights, M.exact_state(b, early))
    old = B.observe(b, next_path, True, True, first["carry"])
    for key in row:
        np.testing.assert_allclose(row[key], old[key], atol=1e-13)
    exact = M.exact_state(b, np.array([0]))
    empty = {
        "count0": np.zeros_like(exact["count0"]),
        "count": np.zeros_like(exact["count"]),
    }
    row = M.late(b, next_path, np.full(3, -np.log(3)), empty)
    old = B.observe(b, next_path, True, True)
    for key in row:
        np.testing.assert_allclose(row[key], old[key], atol=1e-13)


def test_particle_likelihood_tiny_world_fixed_replication_diagnostic():
    b = {"alpha0": np.array([[0.7, 1.3]]), "alpha": np.array([[[0.2, 0.8], [3, 2]]])}
    masks = np.array([[1, 1], [1, 0], [1, 1], [0, 1]], bool)
    total = 0.0
    for path in product(range(2), repeat=4):
        if all(masks[t, s] for t, s in enumerate(path)):
            total += np.exp(B.component_log_likelihood(b, [np.array(path)], True)[0])
    values = [
        np.exp(M.filter_masks(b, masks, 32, seed)["log_evidence"][-1, 0])
        for seed in range(200)
    ]
    assert abs(np.mean(values) - total) < 0.015


def test_source_labels_do_not_enter_filtering_or_next_state_predictions():
    b, early, next_path = bundle(), np.array([0, 5, 2]), np.array([7, 1, 0])
    other = {**b, "codes": b["codes"][:, ::-1]}
    a, z = M.early_contracts(b, early, 16, 1), M.early_contracts(other, early, 16, 1)
    np.testing.assert_array_equal(a["fields"], z["fields"])
    for w, state, v, other_state in zip(
        a["logweights"], a["states"], z["logweights"], z["states"], strict=True
    ):
        first, second = (
            M.late(b, next_path, w, state),
            M.late(other, next_path, v, other_state),
        )
        np.testing.assert_array_equal(first["predictions"], second["predictions"])


def test_cold_author_key_changes_metadata_only_and_rejects_labels():
    b, r = bundle(), record()
    first = M.cold(b, r, 16, 3)
    second = M.cold(b, dict(r, author_key="different"), 16, 3)
    assert first["contracts"] == second["contracts"]
    with pytest.raises(ValueError):
        M.cold(b, dict(r, big5=[1] * 5), 16, 3)


@pytest.mark.parametrize(
    "bad", [np.zeros((2, 8), bool), np.ones((2, 8)), np.zeros((0, 8), bool)]
)
def test_invalid_observation_masks_are_not_filled(bad):
    with pytest.raises(ValueError):
        M.filter_masks(bundle(), bad, 16, 1)


def test_source_prior_zero_is_not_smoothed():
    b = bundle()
    b["alpha"][0, 0, 0] = 0
    with pytest.raises(ValueError):
        M.filter_masks(b, np.ones((2, 8), bool), 16, 1)


def test_conditional_log_ratio_preserves_natural_chain_rule():
    x = M.early_contracts(bundle(), np.array([0, 5, 2]), 16, 1)
    fields = x["fields"]
    values = C.contract_log_likelihoods(fields)
    np.testing.assert_allclose(values[2] + values[4], values[1], atol=1e-13)
    np.testing.assert_allclose(values[3] + values[5], values[1], atol=1e-13)
    assert abs(logsumexp(x["logweights"][4])) < 1e-13
