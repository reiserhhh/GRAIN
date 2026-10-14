#!/usr/bin/env python3
"""Synthetic design and exact-contrast tests for HM47; no model invocation."""

import json
from pathlib import Path

import numpy as np
import pytest

from grain_research.construction import math_evaluative_routing as E
from grain_research.construction.math_construct_routing import documents as parent_docs

ROOT = Path(__file__).resolve().parents[1]


def specs():
    """Load declared public behavioural and appraisal material only."""
    parent = json.loads(
        (ROOT / "configs/grain_math_construct_routing_v1.json").read_text()
    )
    spec = json.loads(
        (ROOT / "configs/grain_math_evaluative_routing_v1.json").read_text()
    )
    return parent, spec


def test_material_integrity_and_complete_roles():
    """The appraisal follows unchanged behaviour, before the account boundary."""
    parent, spec = specs()
    old = {r["doc_id"]: r for r in parent_docs(parent)}
    rows = E.documents(parent, spec)
    assert len(rows) == 216
    assert sum(r["role"] == "source" for r in rows) == 72
    assert sum(r["evaluation"] == "none" for r in rows) == 72
    for row in rows:
        previous = old[row["hm46_doc_id"]]
        a, b = row["property_interval"]
        assert row["body"][a:b] == previous["property"]
        assert row["body"][:b] == previous["body"][:b]
        if row["evaluation"] == "none":
            assert row["body"] == previous["body"]
        else:
            c, d = row["evaluation_interval"]
            assert c > b
            assert row["body"][c:d] == spec["evaluation_sentences"][row["evaluation"]]
            assert row["body"][d:] == " End of account."
    controls = [r for r in rows if (r["other"], r["imagined"]) != (0, 0)]
    assert len(controls) == 72
    assert all(r["role"] == "application" and r["wording"] == 0 for r in controls)


def test_exact_design_reconstruction():
    """Every main, interaction and appraisal-presence component is retained."""
    values = np.random.default_rng(47).normal(size=(2, 3, 6))
    coefficients, result = E.decompose(values)
    assert coefficients.shape == (2, 3, 6)
    assert result["reconstruction_max_error"] < 1e-14
    assert result["parseval_error"] < 1e-13


def test_additive_appraisal_does_not_change_behaviour_contrast():
    """Common score shift and changed construct response are different effects."""
    values = np.array([[3, 5, 1], [1, 3, -1]], float)
    _, result = E.decompose(values)
    np.testing.assert_array_equal(result["behaviour_contrast"], [2, 2, 2])
    np.testing.assert_array_equal(result["evaluation_contrast_by_behaviour"], [4, 4])
    assert result["interaction"] == 0


def test_pure_interaction_survives_zero_appraisal_mean():
    """Opposite appraisal responses cannot be cancelled by averaging behaviours."""
    values = np.array([[0, 1, -1], [0, -1, 1]], float)
    _, result = E.decompose(values)
    assert result["interaction"] == 4
    assert np.mean(result["evaluation_contrast_by_behaviour"]) == 0


def test_presence_not_confused_with_evaluation_polarity():
    """Adding a longer reviewer sentence has a separate signed-even contrast."""
    values = np.array([[1, 3, 3], [0, 2, 2]], float)
    _, result = E.decompose(values)
    np.testing.assert_array_equal(result["evaluation_contrast_by_behaviour"], [0, 0])
    np.testing.assert_array_equal(result["appraisal_presence_by_behaviour"], [2, 2])


def test_missing_and_nonfinite_not_completed():
    """Missing experimental cells cannot be filled with constants."""
    with pytest.raises(ValueError):
        E.decompose(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        E.decompose(np.full((2, 3), np.nan))


def test_matrix_scaling_and_zero_boundaries():
    """Scalar modulation can succeed while diagonal-only response is not assumed."""
    a = np.arange(36, dtype=float).reshape(6, 6)
    result = E.matrix_comparison(a, 3 * a)
    assert result["projection"] == pytest.approx(3)
    assert result["residual_squared_share"] == pytest.approx(0)
    zero = E.matrix_comparison(np.zeros((6, 6)), np.eye(6))
    assert zero["projection"] is None
    assert zero["cosine"] is None


def test_carrier_shape_and_zero_norm():
    """Carrier metadata reports undefined relative changes without repairing them."""
    result = E.array_change(np.zeros((2, 3)), np.ones((2, 3)))
    assert result["maximum_absolute_change"] == 1
    assert result["relative_delta_norm"] is None
    with pytest.raises(ValueError):
        E.array_change(np.ones((2, 3)), np.ones((3, 3)))
