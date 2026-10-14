"""Exact Bayes, information and distinct policy behavior checks."""

import numpy as np

from grain_research.gptpro_foundation_v3.f06_family_policies import (
    LAWS,
    acquisition_values,
    entropy,
    policy_action,
    posterior_update,
    run_policy,
)


def test_information_gain_matches_enumerated_expected_entropy():
    w = np.arange(1.0, 10.0)
    w /= w.sum()
    actual = acquisition_values(w, "expected_entropy_gain")
    expected = []
    for action in range(8):
        p = LAWS[action] @ w
        expected.append(
            entropy(w)
            - p * entropy(posterior_update(w, action, 1))
            - (1 - p) * entropy(posterior_update(w, action, 0))
        )
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_epistemic_not_total_bernoulli_variance():
    w = np.zeros(9)
    w[0] = 1
    np.testing.assert_array_equal(
        acquisition_values(w, "epistemic_variance"), np.zeros(8)
    )
    assert np.all((LAWS @ w) * (1 - LAWS @ w) > 0)


def test_direct_likelihood_replay_and_final_update():
    uniforms = np.random.default_rng(42).random(24)
    path = run_policy(4, uniforms, np.arange(24) % 8, "coverage")
    product = np.ones(9) / 9
    for row in path:
        product *= LAWS[row["action"]] if row["response"] else 1 - LAWS[row["action"]]
    product /= product.sum()
    np.testing.assert_allclose(path[-1]["posterior"], product, atol=1e-14)
    np.testing.assert_allclose(path[-1]["entropy_after_update"], entropy(product))


def test_distinct_actions_and_lexical_ties():
    w = np.ones(9) / 9
    assert policy_action(w, "fixed_corner", 5) == 0
    assert policy_action(w, "coverage", 5) == 5
    assert policy_action(w, "random", 5, 3) == 3
    for policy in ("epistemic_variance", "expected_entropy_gain"):
        scores = acquisition_values(w, policy)
        selected = policy_action(w, policy, 0)
        assert (
            selected
            == np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-14, rtol=0))[0]
        )


def test_posterior_ties_do_not_depend_on_roundoff():
    from grain_research.gptpro_foundation_v3.f06_family_policies import lexical_maximum

    assert lexical_maximum([1 / 3, 1 / 3 + 1e-16, 1 / 3]) == 0
    assert lexical_maximum([0.2, 0.5, 0.3]) == 1


def test_fixed_corner_cannot_identify_axis():
    path = run_policy(
        7, np.random.default_rng(15).random(24), np.zeros(24, dtype=int), "fixed_corner"
    )
    for row in path:
        families = np.asarray(row["posterior"]).reshape(3, 3).sum(axis=1)
        np.testing.assert_allclose(families, np.ones(3) / 3, atol=1e-14)


def test_symmetric_laws_make_adaptive_orderings_equivalent():
    """Equal paths follow complementary Bernoulli laws, not copied policies."""
    rng = np.random.default_rng(20260915)
    conditional_variance = LAWS * (1 - LAWS)
    np.testing.assert_allclose(
        conditional_variance, np.tile(conditional_variance[0], (8, 1)), atol=1e-14
    )
    for _ in range(100):
        w = rng.dirichlet(np.ones(9))
        assert policy_action(w, "epistemic_variance", 0) == policy_action(
            w, "expected_entropy_gain", 0
        )
        assert not np.allclose(
            acquisition_values(w, "epistemic_variance"),
            acquisition_values(w, "expected_entropy_gain"),
        )
