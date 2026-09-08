#!/usr/bin/env python3
"""Bidirectional calculus, exact reflection parity and reference exclusion."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.centered_adjoint import center_groups
from grain_research.vector_mechanisms.odd_action import (
    bidirectional_account,
    even_features,
    fit_odd_fields,
    odd_token_readouts,
)


def test_bidirectional_integrals_recover_cubic_and_quartic_parts():
    d = np.array([[0.2, 0.3], [-0.5, -0.4], [0.3, 0.1]])
    labels = np.zeros(3, int)

    # F(X)=sum(X + 2X^2 + 3X^3 + 4X^4), origin zero.
    def grad(x):
        return 1 + 4 * x + 9 * x * x + 16 * x**3

    pos = {t: grad(t * d) for t in (0.0, 0.5, 1.0)}
    neg = {t: grad(-t * d) for t in (0.0, 0.5, 1.0)}
    a = bidirectional_account(d, pos, neg, labels)
    assert a["odd_action"] == pytest.approx(np.sum(d + 3 * d**3))
    assert abs(a["odd_action"]) > 0.01
    assert a["even_action"] == pytest.approx(np.sum(2 * d * d + 4 * d**4))
    assert a["odd_action"] + a["even_action"] == pytest.approx(a["plus_action"])
    assert abs(a["odd_centering_remainder"]) < 1e-12
    assert abs(a["even_centering_remainder"]) < 1e-12


def test_bidirectional_nodes_must_match():
    p = {t: np.ones((4, 3)) for t in (0.0, 0.5, 1.0)}
    n = {t: np.ones((4, 3)) for t in (0.0, 0.25, 0.5, 0.75, 1.0)}
    with pytest.raises(ValueError):
        bidirectional_account(np.ones((4, 3)), p, n, np.zeros(4, int))


def bank():
    """Make nonzero cubic directional actions from centered synthetic states."""
    rng = np.random.default_rng(2020)
    labels = np.repeat([0, 1, 2], 12)
    packets = []
    for role in ("reference", "reference", "evaluation"):
        d = center_groups(rng.normal(size=(36, 8)), labels)
        g = center_groups(d * d, labels) / len(d)
        packets.append(
            {"role": role, "displacement": d, "labels": labels, "integrated_even": g}
        )
    p = np.eye(8)[:, :4]
    return packets, p, np.ones(4)


def test_even_features_and_odd_scores_are_reflection_exact():
    packets, p, scale = bank()
    model = fit_odd_fields(packets, p, scale)
    packet = packets[-1]
    d, labels = packet["displacement"], packet["labels"]
    plus, minus = (
        even_features(d, labels, p, scale),
        even_features(-d, labels, p, scale),
    )
    for name in plus:
        assert plus[name] == pytest.approx(minus[name])
        for group in np.unique(labels):
            assert plus[name][labels == group].sum(0) == pytest.approx(
                np.zeros(plus[name].shape[1]), abs=1e-12
            )
    a, b = odd_token_readouts(d, labels, model), odd_token_readouts(-d, labels, model)
    for name in a:
        assert a[name] == pytest.approx(-b[name])
    assert any(abs(v.mean()) > 1e-4 for v in a.values())


def test_evaluation_gradients_cannot_change_any_field_or_action_weight():
    packets, p, scale = bank()
    a = fit_odd_fields(packets, p, scale)
    packets[-1]["integrated_even"] *= 1e8
    packets[-1]["displacement"] *= 1e8
    b = fit_odd_fields(packets, p, scale)
    for name in a:
        assert a[name] == pytest.approx(b[name])
