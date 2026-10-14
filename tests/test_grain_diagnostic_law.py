#!/usr/bin/env python3
"""Mechanism controls for explicit diagnostic laws, without a neural run."""

import itertools
from copy import deepcopy

import numpy as np
import pytest

from grain_research.construction import math_diagnostic_law as M

POS, NEG = [[48, 12], [8, 32]], [[12, 48], [32, 8]]


def test_known_population_inverse_is_not_likelihood_lookup():
    assert M.posterior_from_counts(POS, 0) == pytest.approx(6 / 7)
    assert M.posterior_from_counts(POS, 1) == pytest.approx(3 / 11)
    assert M.posterior_from_counts(POS, 0) != pytest.approx(0.8)
    for table, option in itertools.product((POS, NEG), (0, 1)):
        x = np.array(table, float)
        prior = x.sum(axis=1) / x.sum()
        likelihood = x[:, option] / x.sum(axis=1)
        posterior = prior[0] * likelihood[0] / (prior @ likelihood)
        assert M.posterior_from_counts(table, option) == pytest.approx(posterior)


def test_law_reversal_keeps_prior_and_reverses_contrast():
    corners = M.law_cube(POS, NEG)
    assert np.allclose(np.sum(POS, axis=1), np.sum(NEG, axis=1))
    assert corners[1, 1] > corners[0, 1]
    assert corners[1, 0] < corners[0, 0]
    np.testing.assert_allclose(corners[:, 0], corners[::-1, 1])
    coefficients = M.characters(corners)
    assert abs(coefficients[1]) < 1e-15
    assert abs(coefficients[2]) < 1e-15
    assert coefficients[3] > 0


def test_reference_changes_without_event_are_not_personality_evidence():
    prior = np.full((2, 2), 0.6)
    np.testing.assert_allclose(M.characters(prior), [0.6, 0, 0, 0], atol=1e-15)
    fact = np.array([[0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(M.characters(fact), [0.5, 0.5, 0, 0])


def test_all_characters_reconstruct_and_author_integrates():
    x = np.arange(12, dtype=float).reshape(2, 2, 3)
    beta = M.characters(x)
    for i, j in itertools.product(range(2), repeat=2):
        b, c = 2 * i - 1, 2 * j - 1
        np.testing.assert_allclose(
            x[i, j], beta[0] + b * beta[1] + c * beta[2] + b * c * beta[3]
        )
    p_hi, p_lo = np.array([0.25, 0.75]), np.array([0.75, 0.25])
    direct = np.einsum("b,bcq->cq", p_hi - p_lo, x)
    np.testing.assert_allclose(M.author_contrasts(x), direct)


def test_selective_installation_and_interaction():
    k = np.arange(48, dtype=float).reshape(2, 2, 3, 4)
    v, dk, dv = k + 1, k + 2, k + 3
    for mode in ("k", "v", "kv", "noop"):
        out_k, out_v = M.install_reference(k, v, dk, dv, [1], mode)
        np.testing.assert_array_equal(out_k[:, :, [0, 2]], k[:, :, [0, 2]])
        np.testing.assert_array_equal(out_v[:, :, [0, 2]], v[:, :, [0, 2]])
    fields = np.array([1.0, 3.0, 4.0, 9.0])
    parts = M.intervention_parts(fields)
    np.testing.assert_allclose(parts, [2, 3, 3])
    assert parts.sum() == fields[-1] - fields[0]


@pytest.mark.parametrize(
    "table,option", [([[0, 1], [2, 3]], 0), (POS, 2), ([[1.5, 2], [3, 4]], 1)]
)
def test_invalid_counts_are_rejected(table, option):
    with pytest.raises(ValueError):
        M.posterior_from_counts(table, option)


@pytest.mark.parametrize("array", [np.zeros((3, 2)), np.full((2, 2), np.nan)])
def test_incomplete_or_nonfinite_corners_are_rejected(array):
    with pytest.raises(ValueError):
        M.characters(array)


def test_finite_world_named_and_anonymous_targets_are_identical():
    named = M.law_cube(POS, NEG)
    category = M.law_cube(POS, NEG)
    np.testing.assert_array_equal(named, category)
    changed_prior = M.law_cube([[8, 2], [8, 32]], [[2, 8], [32, 8]])
    assert not np.allclose(named, changed_prior)


def toy_rows():
    """Create all registered cells with unequal rule lengths but exact pairs."""
    rows = []
    for bank, family, b, c in itertools.product(range(2), range(6), (-1, 1), (-1, 1)):
        reference = [4, 5, 6, 7] if family % 2 else [4, 5, 6]
        counts = [48, 12] if c == 1 else [12, 48]
        rows.append(
            {
                "id": len(rows),
                "bank": bank,
                "family": family,
                "b": b,
                "c": c,
                "input_ids": [
                    99,
                    21,
                    22 + b,
                    30 + family,
                    *counts,
                    8,
                    19 if family % 2 else 0,
                    66,
                    77,
                    88,
                ],
                "history": list(range(1, 8)),
                "current": [9, 10],
                "reference_slot": [4, 5, 6, 7],
                "reference": reference,
                "padding": [] if family % 2 else [7],
                "action_text": f"A chooses {b} in bank {bank}, family {family}",
            }
        )
    return rows


def test_registered_grid_allows_unequal_trait_lengths_but_not_missing_cells():
    from scripts import run_grain_diagnostic_law_hm141 as R

    rows = toy_rows()
    assert len(R.validate_rows(rows)) == 48
    design = R.fixed_design(rows)
    assert len(design) == 196
    assert sum(d["mode"] == "capture" for d in design) == 48
    assert sum(d["mode"] == "noop" for d in design) == 4
    for entry in design[52:]:
        row, donor = rows[entry["row"]], rows[entry["donor"]]
        assert (row["bank"], row["family"], row["b"]) == (
            donor["bank"],
            donor["family"],
            donor["b"],
        )
        assert row["c"] == -donor["c"]
    with pytest.raises(ValueError):
        R.validate_rows(rows[:-1])
    with pytest.raises(ValueError):
        R.validate_rows(rows[:-1] + [rows[0]])


@pytest.mark.parametrize(
    "damage", ["action", "rule_padding", "nonhistory", "token_bag"]
)
def test_wrong_event_support_or_token_bag_is_rejected(damage):
    from scripts import run_grain_diagnostic_law_hm141 as R

    rows = deepcopy(toy_rows())
    if damage == "action":
        rows[0]["action_text"] = "Different action"
    elif damage == "rule_padding":
        rows[0]["reference"].append(7)
    elif damage == "nonhistory":
        rows[0]["input_ids"][9] = 444
    else:
        rows[0]["input_ids"][4] = 444
    with pytest.raises(ValueError):
        R.validate_rows(rows)


def test_whole_field_accounting_includes_named_category_facts_and_units():
    from scripts import run_grain_diagnostic_law_hm141 as R

    rng = np.random.default_rng(141)
    cube = rng.normal(size=(2, 6, 2, 2, 4, 5, 4))
    manifest = {"spec": {"targets": list("ENACOX")}}
    coefficients, accounts, actions, summary, effects, parts = R.outputs(cube, manifest)
    assert (len(coefficients), len(accounts), len(actions), len(summary)) == (
        960,
        1920,
        120,
        8,
    )
    assert effects.shape == cube.shape
    np.testing.assert_allclose(parts.sum(axis=0), effects[..., 3, :, :])
    assert {row["query"] for row in coefficients} == set(R.QUERIES)
    np.testing.assert_array_equal(
        effects[..., 0, :, :], cube[:, :, :, ::-1, 0] - cube[..., 0, :, :]
    )


def test_reference_interchange_preserves_prefix_and_nonreference_carriers():
    import torch

    from grain_research.construction.math_native_kv_identity import suffix_interchange

    torch.manual_seed(141)
    query = torch.randn(2, 1, 6, 3)
    key, value = torch.randn_like(query), torch.randn_like(query)

    def attention(module, q, k, v, mask, scaling, **kwargs):
        weights = torch.softmax(q @ k.transpose(-1, -2) * scaling, dim=-1)
        return (weights @ v).transpose(1, 2), weights

    native, native_weights = attention(None, query, key, value, None, 1.0)
    donor_k = key[0, :, [1, 2]].clone() + 2
    donor_v = value[0, :, [1, 2]].clone() - 3
    result, weights = suffix_interchange(
        attention,
        query,
        key,
        value,
        None,
        1.0,
        4,
        [1, 2],
        donor_k.numpy(),
        donor_v.numpy(),
        None,
    )
    torch.testing.assert_close(result[:, :4], native[:, :4], rtol=0, atol=0)
    torch.testing.assert_close(
        weights[:, :, :4], native_weights[:, :, :4], rtol=0, atol=0
    )
    assert not torch.equal(result[:, 4:], native[:, 4:])
