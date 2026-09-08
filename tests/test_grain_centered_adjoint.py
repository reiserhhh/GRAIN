#!/usr/bin/env python3
"""Reference exclusion, conditional quadrature and low-rank field identities."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.centered_adjoint import (
    body_partition,
    center_groups,
    field_account,
    fit_reference_field,
    integrate_nodes,
    read_field,
)


def test_group_covariance_preserves_action_and_constant_field_cancels():
    rng = np.random.default_rng(91)
    labels = np.repeat([0, 1], 6)
    d = center_groups(rng.normal(size=(12, 8)), labels)
    g = rng.normal(size=d.shape)
    result = field_account(d, g, labels)
    assert result["action"] == pytest.approx(result["covariance_action"])
    assert result["constant_field_action"] == pytest.approx(0, abs=1e-12)


def test_three_node_simpson_integrates_cubic_gradient():
    g = {t: np.full((4, 3), 2 * t**3 + t) for t in (0.0, 0.5, 1.0)}
    assert integrate_nodes(g) == pytest.approx(np.ones((4, 3)))


def test_evaluation_gradients_cannot_change_field():
    rng = np.random.default_rng(92)
    labels = np.repeat([0, 1], 5)
    packets = [
        {
            "role": role,
            "labels": labels,
            "displacement": rng.normal(size=(10, 8)),
            "integrated": rng.normal(size=(10, 8)),
        }
        for role in ("reference", "reference", "evaluation")
    ]
    first = fit_reference_field(packets, rank=3)
    packets[-1]["integrated"] *= 1e8
    second = fit_reference_field(packets, rank=3)
    for name in first:
        assert first[name] == pytest.approx(second[name])


def test_low_rank_contraction_equals_full_quadratic_form():
    rng = np.random.default_rng(93)
    labels = np.repeat([0, 1], 4)
    d = center_groups(rng.normal(size=(8, 7)), labels)
    model = {
        "projection": rng.normal(size=(7, 3)),
        "scale": np.array([1, 2, 3.0]),
        "coef": rng.normal(size=(3, 7)),
        "radial": np.array(-0.2),
    }
    result = read_field(d, labels, model)
    matrix = (model["projection"] / model["scale"]) @ model["coef"]
    assert result["rank64_field"] == pytest.approx(np.sum((d @ matrix) * d) / len(d))
    assert result["isotropic_field"] < 0


def test_partition_rejects_overlaps_and_final_query():
    h = np.arange(42.0).reshape(7, 6)
    mask, labels, _, d = body_partition(h, [[1, 2], [4, 5]])
    assert np.flatnonzero(mask).tolist() == [1, 2, 4, 5]
    assert labels.tolist() == [0, 0, 1, 1]
    assert d[:2].sum(0) == pytest.approx(np.zeros(6))
    with pytest.raises(ValueError):
        body_partition(h, [[1, 2], [2, 3]])
    with pytest.raises(ValueError):
        body_partition(h, [[5, 6]])
