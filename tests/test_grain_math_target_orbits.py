#!/usr/bin/env python3
"""Exact finite-group and source-only target-observation tests."""

import json
from pathlib import Path

import numpy as np
import pytest

from grain_research.construction import math_target_orbits as O


def test_characters_are_complete_and_orthogonal():
    basis = O.character_basis(5)
    np.testing.assert_array_equal(basis.T @ basis, 32 * np.eye(32))
    x = np.random.default_rng(10).normal(size=(32, 10))
    result = O.decompose(x)
    assert result["reconstruction_error"] < 1e-14
    assert result["energy_closure_error"] < 1e-13


def test_character_components_transform_by_their_sign():
    f = np.random.default_rng(11).normal(size=(32, 10))
    c = O.decompose(f)["coefficients"]
    for action in (4, 8, 16, 28):
        shifted = f[np.arange(32) ^ action]
        observed = O.decompose(shifted)["coefficients"]
        sign = O.character_basis(5)[action]
        np.testing.assert_allclose(observed, sign[:, None] * c, atol=1e-14)


def test_group_averaging_erases_binding_not_just_noise():
    basis = O.character_basis(5)
    f = basis[:, 5]
    average = (f + f[np.arange(32) ^ 4]) / 2
    np.testing.assert_array_equal(average, 0)
    assert O.decompose(f)["energies"][5] == 1


def test_bound_owner_contrast_is_a_character_interaction():
    f = np.random.default_rng(12).normal(size=32)
    c = O.decompose(f)["coefficients"]
    groups = {
        u: {
            a: [f[m] for m in range(32) if m & 1 == a and (m >> 2) & 1 == u]
            for a in (0, 1)
        }
        for u in (0, 1)
    }
    contrast = (np.mean(groups[0][1]) - np.mean(groups[0][0])) - (
        np.mean(groups[1][1]) - np.mean(groups[1][0])
    )
    assert contrast == pytest.approx(-4 * c[5])


def test_source_fit_excludes_application_and_anchor_is_preserved():
    basis = O.character_basis(5)
    rows = [
        {"family": 0, "role": "source", "values": basis[:, 5]},
        {"family": 1, "role": "application", "values": np.ones(32) * 999},
    ]
    c = O.source_coefficients(rows)
    prediction = O.anchored_completion(0.7, c, [5])
    assert prediction[0] == 0.7
    np.testing.assert_allclose(prediction, 0.7 + basis[:, 5] - basis[0, 5])
    with pytest.raises(ValueError):
        O.source_coefficients(rows + [rows[0]])


def test_new_materials_have_unaliased_same_word_orbits_and_no_truth():
    spec = json.loads(
        (
            Path(__file__).parents[1] / "configs/grain_math_target_orbits_v1.json"
        ).read_text()
    )
    rows = O.material_bank(spec)
    assert len(rows) == 800
    assert len({r["text"] for r in rows}) == 656
    assert sum(r["kind"] == "full" for r in rows) == 512
    assert not any(r["psychological_truth_supplied"] for r in rows)
    assert all(
        r["evidence_status"]["adapted_dejected5"] == "unobserved_direct_self_evidence"
        for r in rows
    )
    other_only = [r for r in rows if r["kind"] == "full" and r["anxiety_owner"] == 1]
    assert all(
        r["evidence_status"]["adapted_na8"] == "unobserved_direct_self_evidence"
        for r in other_only
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_or_incomplete_rejected(bad):
    with pytest.raises(ValueError):
        O.decompose([0, bad])
    with pytest.raises(ValueError):
        O.decompose(np.zeros(31))
