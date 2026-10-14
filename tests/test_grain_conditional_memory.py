#!/usr/bin/env python3
"""Exact observation contracts, hidden-path summation and safe cold interfaces."""

import copy
from itertools import product

import numpy as np
import pytest
from scipy.special import logsumexp

from grain_research.construction import math_author_belief as B
from grain_research.construction import math_conditional_memory as M


def bundle():
    """Asymmetric positive arithmetic fixture, not psychological discovery data."""
    rng = np.random.default_rng(282)
    return {
        "alpha0": rng.uniform(0.1, 3, (3, 8)),
        "alpha": rng.uniform(0.1, 5, (3, 8, 8)),
        "count0": np.zeros((3, 8)),
        "count": np.zeros((3, 8, 8)),
        "base0": np.zeros(8),
        "base": np.zeros((8, 8)),
        "codes": np.array([[0, 12, 31], [31, 0, 12]]),
    }


@pytest.mark.parametrize("view", M.VIEWS)
def test_forward_matches_all_short_hidden_paths(view):
    b, path = bundle(), np.array([0, 5, 2])
    initial, transition = M.fixed_rows(b)
    masks = M.observation_masks(path, view)
    total = np.zeros(3)
    for x in product(range(8), repeat=3):
        if all(masks[t, a] for t, a in enumerate(x)):
            total += (
                initial[:, x[0]] * transition[:, x[0], x[1]] * transition[:, x[1], x[2]]
            )
    np.testing.assert_allclose(
        M.forward_log_likelihood(b, path, view)[-1], np.log(total)
    )


def test_full_equals_old_fixed_model_at_every_prefix():
    b, path = bundle(), np.array([0, 5, 2, 7, 1, 2])
    full = M.forward_log_likelihood(b, path, "full")
    for t in range(1, len(path) + 1):
        np.testing.assert_allclose(
            full[t], B.component_log_likelihood(b, [path[:t]], False)
        )
    np.testing.assert_allclose(full[0], 0)


@pytest.mark.parametrize("view", ["condition", "expression"])
def test_hidden_old_bits_never_leak_into_coarsened_likelihood(view):
    b, path = bundle(), np.array([0, 5, 2, 7])
    changed = path ^ (3 if view == "condition" else 4)
    np.testing.assert_array_equal(
        M.forward_log_likelihood(b, path, view),
        M.forward_log_likelihood(b, changed, view),
    )
    assert (
        np.linalg.norm(
            M.forward_log_likelihood(b, path, "full")
            - M.forward_log_likelihood(b, changed, "full")
        )
        > 0.01
    )


def test_chain_rule_not_product_of_marginal_views():
    fields = M.evidence_fields(bundle(), np.array([0, 5, 2, 7]))
    ell = M.contract_log_likelihoods(fields)
    np.testing.assert_allclose(ell[2] + ell[4], ell[1])
    np.testing.assert_allclose(ell[3] + ell[5], ell[1])
    weights = M.posterior_logweights(ell)
    np.testing.assert_allclose(logsumexp(weights, axis=-1), 0, atol=1e-15)
    np.testing.assert_allclose(
        M.posterior_logweights(weights[2] + ell[4]), weights[1], atol=1e-14
    )
    assert np.linalg.norm(M.posterior_logweights(ell[2] + ell[3]) - weights[1]) > 0.01
    assert np.linalg.norm(weights[4] - weights[1]) > 0.01


def test_late_starts_anew_and_keeps_complete_profiles():
    b, early, late = bundle(), np.array([0, 5, 2]), np.array([7, 1, 0])
    ell = M.contract_log_likelihoods(M.evidence_fields(b, early))[4, -1]
    logw = M.posterior_logweights(ell)
    row = M.observe_late(b, late, logw)
    initial, _ = M.fixed_rows(b)
    np.testing.assert_allclose(row["predictions"][0], np.exp(logw) @ initial)
    total = ell + B.component_log_likelihood(b, [late], False)
    np.testing.assert_allclose(
        row["profiles"][:, -1],
        B.profile(np.exp(M.posterior_logweights(total)), b["codes"]),
    )
    np.testing.assert_allclose(row["profiles"].sum(-1), 1)


def test_anchors_change_profiles_not_observational_likelihoods():
    b, path = bundle(), np.array([0, 5, 2])
    changed = {**b, "codes": b["codes"][:, ::-1]}
    np.testing.assert_array_equal(
        M.evidence_fields(b, path), M.evidence_fields(changed, path)
    )


def record():
    """Two identified episodes in strict publication order."""
    return {
        "author_key": "new",
        "episodes": [
            {"events": [{"event_id": "a", "timestamp": 1, "cell": 0}]},
            {"events": [{"event_id": "b", "timestamp": 2, "cell": 5}]},
        ],
    }


def test_cold_returns_all_contracts_and_rejects_hidden_labels():
    r = record()
    assert set(M.cold(bundle(), r)["contracts"]) == set(M.CONTRACTS)
    for key in ("label", "duplicate", "time", "episodes"):
        bad = copy.deepcopy(r)
        if key == "label":
            bad["big5"] = [1] * 5
        elif key == "duplicate":
            bad["episodes"][1]["events"][0]["event_id"] = "a"
        elif key == "time":
            bad["episodes"][1]["events"][0]["timestamp"] = 0
        else:
            bad["episodes"] = bad["episodes"][:1]
        with pytest.raises(ValueError):
            M.cold(bundle(), bad)


@pytest.mark.parametrize("value", [np.array([0.5]), np.array([8]), np.array([])])
def test_invalid_observation_support_is_rejected(value):
    with pytest.raises(ValueError):
        M.evidence_fields(bundle(), value)
