#!/usr/bin/env python3
"""Unit tests for score-link versus vector interaction mechanisms."""

import numpy as np
import pytest

from grain_research.construction import math_native_joint as N


def test_gauge_and_clr_identity():
    z = np.random.default_rng(1).normal(size=(8, 10))
    p = N.probability(z)
    np.testing.assert_allclose(N.probability(z + np.arange(8)[:, None]), p)
    np.testing.assert_allclose(
        np.log(p) - np.log(p).mean(1, keepdims=True), N.center_logits(z)
    )


def test_additive_logits_can_have_nonlinear_score_interaction():
    z = np.array([[0, -2], [0, -1], [0, -1], [0, 0]], float)
    d = N.interaction(z, [0, 1])
    np.testing.assert_allclose(d["logit_interaction"], 0)
    assert abs(d["total"]) > 0.01
    assert d["representation_action"] == 0


def test_complete_interaction_closure_and_integral():
    z = np.random.default_rng(2).normal(size=(4, 10))
    values = np.linspace(0, 1, 10)
    d = N.interaction(z, values)
    assert d["total"] == pytest.approx(d["link_only"] + d["representation_action"])
    assert N.response_integral(
        d["additive_logits"], d["logit_interaction"], values
    ) == pytest.approx(d["representation_action"], abs=1e-12)


def test_kl_bregman_and_fisher_psd():
    x, y = np.random.default_rng(3).normal(size=(2, 10))
    d = N.categorical_geometry(x, y)
    assert d["kl"] >= 0
    assert d["fisher_energy"] >= 0
    assert d["kl"] == pytest.approx(d["bregman"], abs=1e-12)
    q = N.probability(y)
    f = np.diag(q) - np.outer(q, q)
    assert np.linalg.eigvalsh(f).min() > -1e-12
    assert np.max(abs(f @ np.ones(10))) < 1e-12


def test_source_fit_does_not_use_cold_targets():
    def row(f, role, effect):
        return {
            "family": f,
            "role": role,
            "frame": 0,
            "actor": "self",
            "incidence": 0,
            "logit_interaction": np.array([effect, -effect]),
            "total": effect,
        }

    rows = [row(0, "source", 1), row(1, "source", 3), row(2, "application", 99)]
    fit = N.fit_joint_source(rows)
    np.testing.assert_array_equal(fit["global_logits"], [2, -2])
    assert fit["source_families"] == 2
    with pytest.raises(ValueError):
        N.fit_joint_source(rows + [rows[0]])


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_reject_nonfinite(bad):
    with pytest.raises(ValueError):
        N.probability([0, bad])


def test_reject_incomplete_or_mismatched_tables():
    with pytest.raises(ValueError):
        N.interaction(np.ones((3, 10)), np.linspace(0, 1, 10))
    with pytest.raises(ValueError):
        N.score(np.ones(10), [0, 1])


def test_source_application_uncertainty_zero_when_exact():
    rows, units = [], []
    values = [0, 1]
    effect = np.array([0.2, -0.2])
    for f in (0, 1):
        rows.append(
            {
                "family": f,
                "role": "source",
                "frame": 0,
                "actor": "self",
                "incidence": 0,
                "logit_interaction": effect,
                "total": 0,
            }
        )
    for f in (2, 3):
        z = np.array([0.0, 0.0])
        units.append(
            {
                "family": f,
                "role": "application",
                "key": (0, "self", 0),
                "baseline": z,
                "actual": float(N.score(z + effect, values)),
                "values": values,
            }
        )
    ci = N.family_interval(rows, units, draws=100)
    assert ci[0] == pytest.approx(ci[1])
    assert ci[0] > 0
