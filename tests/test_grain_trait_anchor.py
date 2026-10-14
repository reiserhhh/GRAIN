#!/usr/bin/env python3
"""Tests of author weighting, source target units and cold composition."""

from copy import deepcopy

import numpy as np
import pytest

from grain_research.construction import math_trait_anchor as A


def law():
    """Make source authors with known finite distributions and mixed targets."""
    p = np.eye(8)
    y = np.column_stack([(np.arange(8) & m) != 0 for m in (1, 2, 4, 3, 5)])
    return A.compile_law(p, y, [0, 0], np.repeat(0.5, 5))


def record(cell):
    """Create one current anonymous coordinate observation."""
    x = np.zeros(15)
    x[0], x[5] = (1 if cell & 1 else -1), (1 if cell & 2 else -1)
    return {
        "author_key": "opaque",
        "events": [
            {"event_id": "e1", "condition": cell // 4, "coordinates": x.tolist()}
        ],
    }


def test_bayes_density_contrast_and_author_reference():
    b = law()
    A.verify_law(b)
    r, pi, psi, rho1, rho0 = [
        np.asarray(b[k])
        for k in ("reference_mass", "prior", "probabilities", "rho1", "rho0")
    ]
    np.testing.assert_allclose(psi - pi, pi * (1 - pi) * (rho1 - rho0) / r[:, None])
    np.testing.assert_allclose(r @ (psi - pi), np.zeros(5), atol=1e-15)


@pytest.mark.parametrize("cell", range(8))
def test_cold_full_choice_within_and_all_character_accounts(cell):
    out = A.cold_author(law(), record(cell))
    np.testing.assert_allclose(out["relative"], np.array(out["choice"]) + out["within"])
    np.testing.assert_allclose(
        out["relative"], np.array(out["walsh_contributions"]).sum(0)
    )
    renamed = record(cell)
    renamed["author_key"] = "another identity"
    assert A.cold_author(law(), renamed)["relative"] == out["relative"]
    assert out["person_posterior"] is False


def test_label_polarity_orients_observation_but_not_person_posterior():
    b = law()
    p = np.eye(8)
    y = np.column_stack([(np.arange(8) & m) != 0 for m in (1, 2, 4, 3, 5)])
    reverse = A.compile_law(p, 1 - y, [0, 0], np.repeat(0.5, 5))
    np.testing.assert_allclose(
        A.cold_author(b, record(7))["relative"],
        -np.array(A.cold_author(reverse, record(7))["relative"]),
    )


def test_missing_source_support_is_not_zero_score():
    p, y = np.eye(8)[:4], np.tile([[0] * 5, [1] * 5], (2, 1))
    b = A.compile_law(p, y, [0, 0], np.repeat(0.5, 5))
    assert A.cold_author(b, record(7))["status"] == "UNOBSERVED_SOURCE_CELLS"
    assert (
        A.cold_author(b, {"author_key": "x", "events": []})["status"]
        == "NO_OBSERVATIONS"
    )


def test_cold_does_not_accept_labels_or_repeated_identity():
    bad = record(7)
    bad["label"] = [0.8] * 5
    with pytest.raises(ValueError, match="excludes"):
        A.cold_author(law(), bad)
    bad = record(7)
    bad["events"] *= 2
    with pytest.raises(ValueError, match="Distinct"):
        A.cold_author(law(), bad)


def test_law_rejects_changed_units_and_nonprobability_features():
    bad = deepcopy(law())
    bad["prior"][0] += 0.1
    with pytest.raises(ValueError):
        A.verify_law(bad)
    with pytest.raises(ValueError):
        A.validate_mass(np.ones((2, 8)))
    with pytest.raises(ValueError):
        A.compile_law(np.eye(8), np.ones((8, 5)), [0, 0], np.repeat(0.5, 5))


def test_character_basis_is_complete_not_a_new_theorem():
    chi = A.characters()
    np.testing.assert_array_equal(chi.T @ chi, 8 * np.eye(8))
