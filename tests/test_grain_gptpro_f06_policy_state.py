"""Timing, selection-likelihood and truncation checks for F06 interventions."""

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f06_policy_state import (
    gaussian_policy_filter,
    grid_policy_filter,
    observed_arrays,
    policy_sensitivity,
    simulate_policy_pair,
)


def example(policy="hidden", beta=0.4, events=8):
    """Fixed exogenous stream for comparable timing checks."""
    rng = np.random.default_rng(15)
    return simulate_policy_pair(
        0.2,
        0.5,
        0.6,
        beta,
        0.1,
        rng.normal(0, 0.5, events),
        rng.normal(0, 0.5, events),
        rng.random(events),
        policy,
    )


def test_grid_matches_scalar_gaussian_when_policy_is_ignorable():
    rows = example()["observed"]
    exact = gaussian_policy_filter(rows, 0.2, 0.5, 0.6, 0.4)
    grid = grid_policy_filter(rows, 0.2, 0.5, 0.6, 0.4, (None, 0.0))
    assert grid["status"] == "OK"
    np.testing.assert_allclose(
        np.asarray(grid["state_mean"])[:, 0], exact["state_mean"], atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(grid["loglik"][0], exact["loglik"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        grid["loglik"][1], exact["loglik"] - len(rows) * np.log(2), atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        np.asarray(grid["state_mean"])[:, 1], exact["state_mean"], atol=1e-10, rtol=0
    )


def test_last_action_changes_selection_but_not_past_physical_state():
    rows = example()["observed"]
    changed = [dict(row) for row in rows]
    changed[-1]["h"] = 1 - rows[-1]["h"]
    left = gaussian_policy_filter(rows, 0.2, 0.5, 0.6, 0.4)
    right = gaussian_policy_filter(changed, 0.2, 0.5, 0.6, 0.4)
    assert left == right
    first = grid_policy_filter(rows, 0.2, 0.5, 0.6, 0.4, (3.0,))
    second = grid_policy_filter(changed, 0.2, 0.5, 0.6, 0.4, (3.0,))
    assert abs(first["loglik"][0] - second["loglik"][0]) > 0.01
    np.testing.assert_allclose(
        first["before_action_state_mean"], second["before_action_state_mean"]
    )
    assert abs(first["state_mean"][-1][0] - second["state_mean"][-1][0]) > 0.01


def test_zero_actions_leave_beta_likelihood_flat():
    rows = example("zero")["observed"]
    results = [
        gaussian_policy_filter(rows, 0.2, 0.5, 0.6, beta)["loglik"]
        for beta in (-0.5, 0, 0.4, 1)
    ]
    assert len(set(results)) == 1


def test_recorded_policy_has_no_extra_latent_selection_factor():
    rows = example("recorded")["observed"]
    left = gaussian_policy_filter(rows, 0.2, 0.5, 0.6, 0.4, policy="recorded")
    assert left == gaussian_policy_filter(rows, 0.2, 0.5, 0.6, 0.4)
    wrong = [dict(row) for row in rows]
    wrong[-1]["h"] = 1 - wrong[-1]["h"]
    assert (
        gaussian_policy_filter(wrong, 0.2, 0.5, 0.6, 0.4, policy="recorded")["status"]
        == "IMPOSSIBLE_RECORDED_POLICY"
    )


def test_counterfactual_difference_obeys_intervention_recurrence():
    for policy in ("zero", "recorded", "hidden"):
        result = example(policy)
        delta = np.asarray(result["delta_state"])
        h = np.array([r["h"] for r in result["observed"]])
        assert delta[0] == 0
        np.testing.assert_allclose(
            delta[1:], 0.6 * delta[:-1] + 0.4 * h[:-1], atol=1e-14
        )
        assert example(policy, beta=0)["sum_squared_perturbation"] == 0


def test_input_prohibits_truth_and_duplicate_events():
    rows = example()["observed"]
    with pytest.raises(ValueError, match="duplicate"):
        observed_arrays([rows[0], rows[0]])
    with pytest.raises(ValueError, match="observed event"):
        observed_arrays([{**rows[0], "true_state": 0}])


def test_grid_expands_once_and_exposes_unresolved_tail_loss():
    rows = example("zero")["observed"][:1]
    expanded = policy_sensitivity(rows, 0.2, 0.5, 0.95, 0, (None,), q=1)
    assert expanded["status"] == "OK"
    assert len(expanded["grid_attempts"]) == 2
    assert expanded["grid_attempts"][0]["status"] == "GRID_INADEQUATE"
    failed = policy_sensitivity(rows, 0.2, 0.5, 0.9999, 0, (None,))
    assert failed["status"] == "GRID_INADEQUATE"
    assert len(failed["grid_attempts"]) == 2
    assert failed["max_outside_mass"] > 0.001


def test_hidden_action_update_matches_continuous_quadrature():
    from scipy.integrate import quad
    from scipy.special import expit
    from scipy.stats import norm

    row = {
        "author_id": "a",
        "event_id": "a:0",
        "role": "source",
        "t": 0,
        "z": 1.0,
        "y": 0.7,
        "h": 1,
    }
    prior_v = 0.25 / (1 - 0.6**2)
    posterior_v = 1 / (1 / prior_v + 1 / 0.25)
    denominator = quad(
        lambda s: norm.pdf(s, scale=np.sqrt(posterior_v)) * expit(3 * s),
        -np.inf,
        np.inf,
    )[0]
    numerator = quad(
        lambda s: s * norm.pdf(s, scale=np.sqrt(posterior_v)) * expit(3 * s),
        -np.inf,
        np.inf,
    )[0]
    result = grid_policy_filter([row], 0.2, 0.5, 0.6, 0.4, (3.0,))
    expected_loglik = norm.logpdf(0, scale=np.sqrt(prior_v + 0.25)) + np.log(
        denominator
    )
    np.testing.assert_allclose(
        result["state_mean"][0][0], numerator / denominator, atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(result["loglik"][0], expected_loglik, atol=1e-10, rtol=0)
