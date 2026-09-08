#!/usr/bin/env python3
"""Reference-only conditional fields and explicit comment/layout information."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.centered_adjoint import center_groups
from grain_research.vector_mechanisms.conditioned_action import (
    CONDITIONS,
    fit_conditioned_fields,
    token_conditions,
    token_readouts,
)


def data():
    """Make synthetic reference and evaluation token banks with real group offsets."""
    rng = np.random.default_rng(801)
    packets = []
    for role in ("reference", "reference", "evaluation"):
        h = rng.normal(size=(60, 8))
        labels = np.repeat(np.arange(6), 10)
        h += np.repeat(rng.normal(size=(6, 8)), 10, axis=0)
        d = center_groups(h, labels)
        z = token_conditions(h, labels, np.eye(8))
        g = d * (0.5 + 0.1 * z[:, [0]])
        packets.append(
            {
                "role": role,
                "body": h,
                "displacement": d,
                "labels": labels,
                "integrated": g / len(d),
            }
        )
    return packets


def test_conditions_keep_background_separate_from_layout_and_deviation():
    p = data()[0]
    first = token_conditions(p["body"], p["labels"], np.eye(8))
    second = token_conditions(p["body"] + 2, p["labels"], np.eye(8))
    assert second[:, :8] == pytest.approx(first[:, :8] + 2)
    assert second[:, 8:] == pytest.approx(first[:, 8:])
    assert first[:10, :8] == pytest.approx(np.repeat(first[[0], :8], 10, axis=0))
    assert first[0, 9] == 0 and first[-1, 9] == 1
    assert first[9, 10] == 1 and first[10, 10] == 0


def test_check_authors_cannot_change_conditions_or_coefficients():
    packets = data()
    base = {"vectors": np.eye(8)[:, :2], "action_refit": np.array([0.2, -0.1])}
    model = fit_conditioned_fields(packets, base, np.eye(8))
    packets[-1]["body"] *= 1e8
    packets[-1]["integrated"] *= 1e8
    other = fit_conditioned_fields(packets, base, np.eye(8))
    for key in model:
        assert model[key] == pytest.approx(other[key])


def test_zero_condition_coefficients_reproduce_base_for_any_scoring_material():
    p = data()[-1]
    v = np.eye(8)[:, :2]
    model = {
        "vectors": v,
        "base": np.array([0.2, -0.1]),
        "condition_projection": np.eye(8),
        "condition_mean": np.zeros(12),
        "condition_std": np.ones(12),
    }
    for name, cols in CONDITIONS.items():
        coef = np.zeros((2, 1 + len(cols)))
        coef[:, 0] = model["base"]
        model[f"{name}_coef"] = coef
    scores = token_readouts(p["body"], p["labels"], model)
    for name in CONDITIONS:
        assert scores[name] == pytest.approx(scores["base"])
