#!/usr/bin/env python3
"""Synthetic construction and independent actor/focus count-account tests."""

import numpy as np
import pytest
import torch

from grain_research.construction import math_expression_response as M
from grain_research.construction import math_recursive_collector as C
from tests.test_grain_math_recursive_collector import oracle, toy


def history_case():
    """Address the old independent toy oracle through the new context interface."""
    library, weights, config = toy()
    case = {"levels": [0, 1, 2, 1, 1, 0], "owner": 1}
    keys, values, masks = [], [], []
    for layer in range(2):
        k, v, mask = C.assemble_context(library, [case], layer, 9)
        keys.append(k[0])
        values.append(v[0])
        masks.append(mask[0])
    histories = {"k": np.stack(keys)[None], "v": np.stack(values)[None]}
    seeds = np.stack((library["seed"], library["seed"] + 0.02))
    return library, weights, config, case, histories, seeds, np.array(masks[:1])


def test_constructed_focuses_against_independent_oracle():
    """Both new focus seeds equal independent NumPy full scalar-loop oracles."""
    library, weights, config, case, histories, seeds, valid = history_case()
    k, v = M.compile_histories(
        histories, seeds, library["cos"], library["sin"], valid, weights, config, "cpu"
    )
    for focus in range(2):
        kk, vv = oracle({**library, "seed": seeds[focus]}, case, weights, config)
        np.testing.assert_allclose(k[0, focus], kk, atol=4e-7, rtol=3e-6)
        np.testing.assert_allclose(v[0, focus], vv, atol=4e-7, rtol=3e-6)


def test_unused_padding_values_do_not_change_current():
    """Large excluded history components never acquire attention opportunity."""
    library, weights, config, _, histories, seeds, valid = history_case()
    baseline = M.compile_histories(
        histories, seeds, library["cos"], library["sin"], valid, weights, config, "cpu"
    )
    changed = {name: value.copy() for name, value in histories.items()}
    for value in changed.values():
        value[:, :, 8] = 999
    for actual, expected in zip(
        M.compile_histories(
            changed,
            seeds,
            library["cos"],
            library["sin"],
            valid,
            weights,
            config,
            "cpu",
        ),
        baseline,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_invalid_history_support_rejected():
    """Missing support is not filled by an artificial zero reading."""
    library, weights, config, _, histories, seeds, valid = history_case()
    with pytest.raises(ValueError, match="support"):
        M.compile_histories(
            histories,
            seeds,
            library["cos"],
            library["sin"],
            valid * False,
            weights,
            config,
            "cpu",
        )


def test_complete_sectors_and_count_identity():
    """Random fields reconstruct with all sectors and direct positive-law counts."""
    n = np.random.default_rng(119).normal(size=(3, 2, 6, 2, 2, 2, 2, 2, 6))
    fields = M.response_fields(n)
    assert max(M.verify_response_identities(n, fields).values()) < 1e-14


def test_other_actor_changes_are_not_self_evidence():
    """Other-actor response is retained but excluded from own author's account."""
    n = np.zeros((3, 2, 6, 2, 2, 2, 2, 2, 6))
    n[:, :, :, :, 1, 0, 0] = 2
    fields = M.response_fields(n)
    assert np.any(fields["other_actor_pairing_diagnostic"])
    assert not fields["qualified_family_pairing"].any()


def test_identical_focus_has_zero_pairing_not_zero_behavior():
    """A missing focus interaction does not delete a real content response."""
    n = np.zeros((3, 2, 6, 2, 2, 2, 2, 2, 6))
    n[:, :, :, :, 1] = 2
    fields = M.response_fields(n)
    assert not fields["gamma"].any()
    assert np.all(fields["actor_even_focus_even"] == 1)


def test_focus_swap_reverses_pairing_only():
    """Viewpoint is typed as an instrument coordinate, not a different author."""
    n = np.random.default_rng(120).normal(size=(3, 2, 6, 2, 2, 2, 2, 2, 6))
    original, swapped = M.response_fields(n), M.response_fields(n[..., ::-1, :, :])
    np.testing.assert_allclose(swapped["gamma"], -original["gamma"])
    np.testing.assert_allclose(
        swapped["actor_odd_focus_even"], original["actor_odd_focus_even"]
    )


def test_identical_seed_focus_is_not_implicitly_rotated():
    """No symmetry shortcut negates or learns the second current state."""
    library, weights, config, _, histories, seeds, valid = history_case()
    seeds[1] = seeds[0]
    with torch.no_grad():
        k, v = M.compile_histories(
            histories,
            seeds,
            library["cos"],
            library["sin"],
            valid,
            weights,
            config,
            "cpu",
        )
    np.testing.assert_array_equal(k[:, 0], k[:, 1])
    np.testing.assert_array_equal(v[:, 0], v[:, 1])


def test_incomplete_or_nonfinite_fields_rejected():
    """Reject absent or invalid response support instead of hiding it as zero."""
    with pytest.raises(ValueError):
        M.response_fields(np.zeros((3, 2, 6, 2, 2, 2, 1, 2, 6)))
    n = np.zeros((3, 2, 6, 2, 2, 2, 2, 2, 6))
    n.flat[0] = np.nan
    with pytest.raises(ValueError):
        M.response_fields(n)
