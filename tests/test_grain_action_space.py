#!/usr/bin/env python3
"""Finite action-quotient identities and author-excluded field construction."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.action_space import (
    action_diagnostics,
    action_projection,
    fit_action_dictionary,
    read_action_dictionary,
    spectral_dictionary,
)
from grain_research.vector_mechanisms.centered_adjoint import center_groups, read_field


def test_direction_quotient_preserves_action_including_zero_rows():
    rng = np.random.default_rng(601)
    d, g = rng.normal(size=(20, 7)), rng.normal(size=(20, 7))
    d[0] = 0
    a, radial, null = action_projection(d, g)
    assert radial + null == pytest.approx(g)
    assert np.sum(d * radial, axis=1) == pytest.approx(a)
    assert np.sum(d * null, axis=1) == pytest.approx(np.zeros(20), abs=1e-12)
    assert np.sum(radial * null) == pytest.approx(0, abs=1e-12)


def test_symmetric_operator_keeps_signed_actions_and_cancels_skew():
    rng = np.random.default_rng(602)
    old = {
        "projection": rng.normal(size=(15, 3)),
        "scale": np.ones(3),
        "coef": rng.normal(size=(3, 15)),
        "radial": np.array(0.0),
    }
    spectral = spectral_dictionary(old)
    assert len(spectral["original"]) == 6
    assert np.min(spectral["original"]) < 0 < np.max(spectral["original"])
    labels = np.repeat([0, 1], 6)
    d = center_groups(rng.normal(size=(12, 15)), labels)
    score = np.mean((d @ spectral["vectors"]) ** 2 @ spectral["original"])
    assert score == pytest.approx(read_field(d, labels, old)["rank64_field"])


def packets():
    """Construct a known quadratic field plus author-excluded gradient evidence."""
    rng = np.random.default_rng(603)
    result = []
    for role in ("reference", "reference", "evaluation"):
        labels = np.repeat([0, 1], 40)
        d = center_groups(rng.normal(size=(80, 5)), labels)
        gamma = d * np.array([0.5, -0.7, 1.2, 0.1, -0.3])
        result.append(
            {
                "role": role,
                "labels": labels,
                "displacement": d,
                "integrated": gamma / len(d),
                "query": "synthetic",
                "actual": float(np.mean(np.sum(d * gamma, axis=1))),
            }
        )
    return result


def test_same_family_both_losses_recover_known_field_without_check_leakage():
    data = packets()
    basis = {"vectors": np.eye(5), "original": np.zeros(5)}
    model = fit_action_dictionary(data, basis, relative_penalty=1e-8)
    for method in ("gradient_refit", "action_refit"):
        assert model[method] == pytest.approx([0.5, -0.7, 1.2, 0.1, -0.3], abs=1e-7)
    data[-1]["integrated"] *= 1e8
    other = fit_action_dictionary(data, basis, relative_penalty=1e-8)
    for name in model:
        assert model[name] == pytest.approx(other[name])


def test_action_diagnostic_accounts_for_signed_aggregate_error():
    data = packets()
    basis = {"vectors": np.eye(5), "original": np.zeros(5)}
    model = fit_action_dictionary(data, basis)
    p = data[-1]
    scores = read_action_dictionary(p["displacement"], p["labels"], model)
    for row in action_diagnostics(p, model):
        assert row["prediction"] == pytest.approx(scores[row["method"]])
        assert row["diagonal_token_error"] + row[
            "offdiagonal_token_error"
        ] == pytest.approx(row["field_error"] ** 2)
        assert row["prediction"] - row["endpoint"] == pytest.approx(
            row["quadrature_error"] - row["field_error"]
        )
        assert row["null_action_max"] < 1e-12
