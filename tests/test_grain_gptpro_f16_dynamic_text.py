#!/usr/bin/env python3
"""Mathematical and input-contract checks for controlled dynamic text scoring."""

import copy
import hashlib

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from grain_research.gptpro_foundation_v3 import f16_dynamic_text as d
from grain_research.gptpro_foundation_v3.f16_t1_text import (
    event_probabilities,
    oracle_channel,
)


def bundle():
    """Small nonsingular source-only instrument without hidden material labels."""
    e = np.full((2, 2, 4), 0.05)
    for b in (0, 1):
        for q in (0, 1):
            e[b, q, 2 * b + q] = 0.85
    text = "I ask for advice."
    material = {
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "pole_readout": 0.9,
        "nli_qualification": 0.9,
        "grain_qualification": 0.8,
    }
    return {
        "version": d.VERSION,
        "scorer_code_hash": d.code_hash(),
        "world_contract": d.CONTRACT,
        "future_conditions": d.PLAN,
        "source_models": [
            {
                "prior": {
                    "mu": [0.0, 0.4, 0.5],
                    "Sigma": np.diag([1.0, 0.3, 0.1]).tolist(),
                },
                "log_sigma": np.log([0.5, 0.5]).tolist(),
            }
        ],
        "source_components": 1,
        "qmc_power": 5,
        "seed": 7,
        "channel_draws": {k: [e.tolist()] for k in ("hybrid_joint", "nli_joint")},
        "materials": {"a": material},
        "reference": {k: {"mu": 0.0, "sigma": 1.0} for k in ("T2", "T3")},
    }


def raw(time=3):
    """One author with observed events at zero and two; requested clock is separate."""
    return {
        "author_id": "u",
        "target_time": time,
        "world_contract": d.CONTRACT,
        "history": [
            {
                "record_id": str(t),
                "op": "new",
                "recorded_at": t,
                "observation": {
                    "author_id": "u",
                    "event_id": str(t),
                    "t": t,
                    "z": (-1) ** t,
                    "material_id": "a",
                },
            }
            for t in (0, 2)
        ],
    }


@pytest.mark.parametrize("c", range(4))
def test_one_update_matches_adaptive_quadrature(c):
    e = np.asarray(bundle()["channel_draws"]["hybrid_joint"][0])
    m, v, offset, r = -0.3, 1.2, 0.7, 0.25

    def f(x):
        return (
            norm.pdf(x, m, np.sqrt(v))
            * event_probabilities(np.array([x + offset]), r, e)[0, c]
        )

    z = quad(f, -np.inf, np.inf, epsabs=1e-11)[0]
    em = quad(lambda x: x * f(x), -np.inf, np.inf, epsabs=1e-11)[0] / z
    ev = quad(lambda x: (x - em) ** 2 * f(x), -np.inf, np.inf, epsabs=1e-11)[0] / z
    got = d.probit_update(
        np.array([[m]]), np.array([[v]]), np.array([[offset]]), r, e, [c]
    )
    np.testing.assert_allclose([x.item() for x in got], [em, ev, np.log(z)], atol=1e-10)


def test_uninformative_channel_preserves_state_and_parameter_information():
    e = np.full((2, 2, 4), 0.25)
    p = np.array([[0.1, 0.5, 0.3], [1.0, -0.4, 1.0]])
    z = np.array([[1.0, -1, 1, -1]])
    m, v, ll = d.filter_nodes(
        p,
        z,
        np.ones_like(z),
        np.ones_like(z),
        np.ones_like(z, bool),
        np.log([0.5, 0.5]),
        e,
        "hybrid_joint",
        3,
    )
    np.testing.assert_allclose(m, 0, atol=1e-14)
    np.testing.assert_allclose(v[:, 0], 0.25 / (1 - (0.99 * np.tanh(p[:, 2])) ** 2))
    np.testing.assert_allclose(ll, 4 * np.log(0.25))


def test_future_joint_covariance_matches_loading_matrix_and_mean_formula():
    p = np.array([[0.2, 0.4, 0.5]])
    m, v = np.array([[0.7]]), np.array([[0.3]])
    mean, cov = d.future_moments(p, m, v, np.log([0.5, 0.6]), d.PLAN)
    rho = 0.99 * np.tanh(0.5)
    a = np.zeros((5, 9))
    for j in range(5):
        a[j, 0] = rho**j
        for k in range(1, j + 1):
            a[j, k] = rho ** (j - k)
        if j:
            a[j, 4 + j] = 1
    expected = a @ np.diag([0.3] + [0.25] * 4 + [0.36] * 4) @ a.T
    np.testing.assert_allclose(cov[0, 0], expected, atol=1e-14)
    g = sum(rho**k for k in range(1, 5)) / 4
    expected_var = (
        0.3 * g * g
        + 0.25 / 16 * sum(sum(rho**k for k in range(5 - j)) ** 2 for j in range(1, 5))
        + 0.36 / 4
    )
    np.testing.assert_allclose(d.TARGETS[1] @ cov[0, 0] @ d.TARGETS[1], expected_var)
    np.testing.assert_allclose(mean[0, 0] @ d.TARGETS[1], 0.2 + 0.7 * g)
    assert not np.isclose(np.trace(cov[0, 0, 1:, 1:]) / 16, expected_var)


def test_missing_clock_steps_propagate_and_empty_is_prior():
    b, x = bundle(), raw(3)
    first = d.cold_scores([x], b)[0]
    later = d.cold_scores([dict(x, target_time=7)], b)[0]
    assert first["target_time"] == 3 and later["target_time"] == 7
    assert abs(later["targets"]["T2"]["score"]) < abs(first["targets"]["T2"]["score"])
    empty = d.cold_scores([dict(x, history=[])], b)[0]
    assert empty["prior_only"] and empty["observed_events"] == 0
    assert empty["targets"]["T2"]["score"] == 0
    assert empty["targets"]["T2"]["variance"] > 0


def test_gaussian_filter_matches_dense_conditioning():
    p = np.array([[0.3, -0.2, 0.6]])
    z = np.array([[1.0, 0, -1, 0, 0]])
    y = np.array([[0.4, 0, -0.7, 0, 0]])
    mask = np.array([[1, 0, 1, 0, 0]], bool)
    m, v, _ = d.filter_nodes(
        p,
        z,
        y,
        np.ones_like(z),
        mask,
        np.log([0.5, 0.4]),
        None,
        "eligible_continuous",
        4,
    )
    times, rho = np.array([0, 2]), 0.99 * np.tanh(0.6)
    stationary = 0.25 / (1 - rho * rho)
    covariance = stationary * rho ** np.abs(
        times[:, None] - times[None, :]
    ) + 0.16 * np.eye(2)
    cross = stationary * rho ** (4 - times)
    resid = y[0, times] - 0.3 + 0.2 * z[0, times]
    np.testing.assert_allclose(m[0, 0], cross @ np.linalg.solve(covariance, resid))
    np.testing.assert_allclose(
        v[0, 0], stationary - cross @ np.linalg.solve(covariance, cross)
    )


def test_alias_retraction_and_batch_id_invariance():
    b, x = bundle(), raw()
    expected = d.cold_scores([x], b)[0]
    alias = copy.deepcopy(x)
    alias["history"].append(
        {"record_id": "alias", "op": "alias", "parent_id": "2", "recorded_at": 3}
    )
    assert d.cold_scores([alias], b)[0] == expected
    removed = copy.deepcopy(alias)
    removed["history"].append(
        {"record_id": "remove", "op": "retract", "parent_id": "2", "recorded_at": 3}
    )
    result = d.cold_scores([removed], b)[0]
    assert result["stale_annotations"] == ["alias"]
    direct = dict(x, history=x["history"][:1])
    assert result["targets"] == d.cold_scores([direct], b)[0]["targets"]
    renamed = copy.deepcopy(x)
    renamed["author_id"] = "other"
    for op in renamed["history"]:
        op["observation"]["author_id"] = "other"
    np.testing.assert_allclose(
        d.cold_scores([renamed, x], b)[1]["joint_mean"],
        expected["joint_mean"],
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "mutation", ["truth", "future", "wrong_author", "duplicate", "labels_in_bundle"]
)
def test_reject_invalid_inputs(mutation):
    b, x = bundle(), raw()
    row = x["history"][0]["observation"]
    if mutation == "truth":
        row["y"] = 1
    elif mutation == "future":
        x["history"][-1]["recorded_at"] = 4
    elif mutation == "wrong_author":
        row["author_id"] = "other"
    elif mutation == "duplicate":
        x["history"][1]["observation"]["t"] = 0
    else:
        b["materials"]["a"]["qualification"] = 1
    with pytest.raises(ValueError):
        d.cold_scores([x], b)


def test_oracle_unqualified_has_no_state_information():
    e = oracle_channel()
    m, v, ll = d.probit_update(
        np.array([[0.4]]), np.array([[0.7]]), np.array([[0.5]]), 0.25, e, [0]
    )
    np.testing.assert_allclose(m, 0.4)
    np.testing.assert_allclose(v, 0.7)
    np.testing.assert_allclose(ll, np.log(0.25))
