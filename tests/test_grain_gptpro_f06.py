#!/usr/bin/env python3
"""F06 adaptive-policy checks."""

import numpy as np

from grain_research.gptpro_foundation_v3.f06 import CONDITIONS, FAMILIES, choose_action, family_probability, update_posterior


def test_family_probabilities_and_posterior_normalize():
    candidates = [(family, beta) for family in FAMILIES for beta in (.5, 1.0, 1.5)]
    weights = np.full(len(candidates), 1.0 / len(candidates))
    assert all(0.0 < family_probability(family, beta, condition) < 1.0 for family, beta in candidates for condition in CONDITIONS)
    updated = update_posterior(weights, candidates, CONDITIONS[0], 1)
    np.testing.assert_allclose(np.sum(updated), 1.0, atol=1e-12)


def test_fixed_policy_has_no_hidden_label_access():
    candidates = [(family, 1.0) for family in FAMILIES]
    weights = np.full(len(candidates), 1.0 / len(candidates))
    assert choose_action(weights, candidates, "fixed", np.random.default_rng(1)) == CONDITIONS[0]
