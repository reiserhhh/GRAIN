#!/usr/bin/env python3
"""Synthetic contracts for relational reading and finite observation bounds."""

import itertools
from collections import Counter

import numpy as np
import pytest

from grain_research.construction import math_relational_appraisal as M


def materials():
    """Make six explicit test clauses, without data or neural dependencies."""
    targets = [f"c{i}" for i in range(6)]
    return {
        "targets": targets,
        "content_families": {
            c: [
                [f"did high {c} example{j}", f"did low {c} example{j}"]
                for j in range(2)
            ]
            for c in targets
        },
    }


def test_documents_scope_and_binding():
    docs = M.documents(materials())
    assert len(docs) == 144
    assert Counter(d["role"] for d in docs) == {"source": 72, "application": 72}
    for d in docs:
        for actor, (start, stop) in d["properties"].items():
            polarity = d["behaviour"] if actor == "A" else 1 - d["behaviour"]
            expected = "high" if polarity == 0 else "low"
            assert d["body"][start:stop].startswith("did " + expected)


def test_raw_word_bag_identical_despite_relations():
    docs = M.documents(materials())
    bags = {}
    for d in docs:
        key = d["family"], d["content"], d["appraisal"] == "none"
        bag = Counter(d["body"].split())
        if key in bags:
            assert bags[key] == bag
        bags[key] = bag


@pytest.mark.parametrize("dimension", (3, 4))
def test_full_characters_and_reconstruction(dimension):
    _, h = M.characters(dimension)
    np.testing.assert_allclose(h @ h.T, np.eye(len(h)) * len(h), atol=0)
    y = np.random.default_rng(48).normal(size=(2,) * dimension)
    result = M.decompose(y)
    assert result["reconstruction_error"] < 1e-14
    assert result["parseval_error"] < 1e-14


def test_bound_and_order_channels_separate():
    y = np.empty((2, 2, 2, 2))
    for b, e, o, r in itertools.product(range(2), repeat=4):
        y[b, e, o, r] = (
            0.5 + 0.2 * (-1) ** (b + r) + 0.1 * (-1) ** (e + r) + 0.03 * (-1) ** (e + o)
        )
    result = M.decompose(y)
    coef = dict(zip(result["bits"], result["coefficients"], strict=True))
    assert coef[(1, 0, 0, 1)] == pytest.approx(0.2)
    assert coef[(0, 1, 0, 1)] == pytest.approx(0.1)
    assert coef[(0, 1, 1, 0)] == pytest.approx(0.03)
    assert result["renaming_rms"] == pytest.approx(0)


def test_renaming_odd_component_not_erased():
    y = np.empty((2, 2, 2, 2))
    for b, e, o, r in itertools.product(range(2), repeat=4):
        y[b, e, o, r] = 0.5 + 0.2 * (-1) ** r
    result = M.decompose(y)
    assert result["name_odd_squared_share"] == pytest.approx(1)
    assert result["renaming_rms"] == pytest.approx(0.4)


def test_bag_bound_loss_identity():
    y = np.array([0.1, 0.2, 0.7, 0.8])
    bound = M.bag_bound(y)
    for constant in (-1, 0.4, 2):
        assert np.mean((y - constant) ** 2) == pytest.approx(
            bound["minimum_mse"] + (constant - bound["oracle_group_mean"]) ** 2
        )


def test_constant_null_and_missing_fail():
    result = M.decompose(np.full((2,) * 4, 0.4))
    assert result["name_odd_squared_share"] is None
    assert M.bag_bound(np.zeros(8))["minimum_mse"] == 0
    with pytest.raises(ValueError):
        M.decompose(np.zeros((2, 2, 1, 2)))
    with pytest.raises(ValueError):
        M.bag_bound([0.1, np.nan, 0.3, 0.4])
