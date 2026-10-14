#!/usr/bin/env python3
"""No-network regression tests for the bounded all-plan follow-ups."""

import numpy as np
import pandas as pd
import pytest

import grain_research.dialogue_ledger_generic as generic
from grain_research.open_condition import (
    assignment_variance,
    flip_mixture,
    source_clause_error,
)
from scripts.run_grain_metadata_conditions import fit_cells, group_partition
from scripts.run_grain_open_condition_assignment import metrics


def test_flip_mixture_limits():
    p = np.array([0, 0.2, 1])
    np.testing.assert_allclose(flip_mixture(p, 0), p)
    np.testing.assert_allclose(flip_mixture(p, 0.5), 0.5)
    np.testing.assert_allclose(flip_mixture(p, 0.1), [0.1, 0.26, 0.9])
    with pytest.raises(ValueError):
        flip_mixture(p, 0.6)


def test_assignment_second_moment_when_group_means_equal():
    fitted = {"mean": [0, 0], "covariance": np.eye(2)}
    assert assignment_variance([0.5], fitted)[0] == 0
    assert assignment_variance([0.5], fitted, full_moment=True)[0] == 0.5
    rng = np.random.default_rng(20260914)
    theta = rng.normal(size=(100000, 2))
    z = rng.binomial(1, 0.5, len(theta))
    residual = theta[np.arange(len(theta)), z] - theta.mean(axis=1)
    assert abs(residual.var() - 0.5) < 0.01


def test_source_error_groups_not_duplicate_rows():
    x = np.array([[-2, 1], [-1, 1], [1, 1], [2, 1]])
    z = np.array([-1, -1, 1, 1])
    g = np.array(["a", "b", "c", "d"])
    a = source_clause_error(x, z, g)
    b = source_clause_error(np.repeat(x, 3, axis=0), np.repeat(z, 3), np.repeat(g, 3))
    assert a["groups"] == b["groups"] == 4
    assert a["epsilon"] == b["epsilon"]


def test_empty_metric_is_missing_not_nan():
    assert metrics([], [], [])["rmse"] is None


def test_metadata_fit_and_partition_are_source_only():
    data = pd.DataFrame({"community": ["a"] * 2 + ["b"], "y": [0.0, 1.0, 1.0]})
    fitted = fit_cells(data, ["community"], 0.5, minimum=2)
    assert list(fitted.index) == ["a"]
    assert fitted.loc["a", "probability"] == 0.5
    a = pd.Series(["author1", "author2", "author1"])
    assert group_partition(a).iloc[0] == group_partition(a).iloc[2]


def test_third_person_not_zero_self_score():
    h = [
        {
            "record_id": "r",
            "kind": "new",
            "event_id": "e",
            "value": 0.0,
            "mode": "past",
            "predicate": "travel",
            "condition": None,
            "proposition": {"actor": "other"},
        }
    ]
    assert generic.snapshot(h)["scores"] == {}


def test_scope_preserves_unrelated_negation(monkeypatch):
    f = {
        "actor": "respondent",
        "source_scoped": False,
        "predicate": "enjoy",
        "ambiguous_negation": False,
        "sign": -1,
    }
    monkeypatch.setattr(generic, "frames", lambda *_: [f])
    assert (
        generic.polarity_scope_decision(
            "Yes, I did not enjoy it", [], {"predicate": "travel"}, 1
        )
        == "retain_particle"
    )
    assert (
        generic.polarity_scope_decision(
            "Yes, I did not enjoy it", [], {"predicate": "enjoy"}, 1
        )
        == "same_predicate_conflict"
    )


def test_scope_abstains_on_unresolved_or_repair(monkeypatch):
    monkeypatch.setattr(generic, "frames", lambda *_: [])
    assert (
        generic.polarity_scope_decision("Yes, not really", [], {"predicate": "go"}, 1)
        == "unresolved_negation"
    )
    assert (
        generic.polarity_scope_decision("Yes, actually no", [], {"predicate": "go"}, 1)
        == "repair_requires_scope"
    )
