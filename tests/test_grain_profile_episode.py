#!/usr/bin/env python3
"""Profile-preserving episode transport and explicit hierarchical likelihoods."""

import copy

import numpy as np
import pytest
from scipy.special import logsumexp

from grain_research.construction import math_author_belief as B
from grain_research.construction import math_conditional_memory as C
from grain_research.construction import math_profile_episode as M
from tests.test_grain_conditional_memory import bundle, record


def grouped_bundle():
    """Make nontrivial groups in a small positive arithmetic fixture."""
    b = bundle()
    b["codes"] = np.array([[0, 0, 31], [0, 31, 31]])
    return b


@pytest.mark.parametrize("assignment", [0, 1])
def test_selected_whole_profile_preserved_not_other_assignment(assignment):
    b = grouped_bundle()
    w = np.log(np.array([0.15, 0.55, 0.30]))
    projected = M.episode_logweights(b, w, assignment)
    before = B.profile(np.exp(w), b["codes"])
    after = B.profile(np.exp(projected), b["codes"])
    np.testing.assert_allclose(before[assignment], after[assignment])
    assert np.linalg.norm(before[1 - assignment] - after[1 - assignment]) > 0.01
    np.testing.assert_allclose(
        M.episode_logweights(b, projected, assignment), projected, atol=1e-14
    )


def test_uniform_prior_and_singleton_profiles_are_exact_nulls():
    b = bundle()
    w = np.log(np.array([0.2, 0.5, 0.3]))
    np.testing.assert_allclose(M.episode_logweights(b, w, 0), w)
    other = grouped_bundle()
    uniform = np.full(3, -np.log(3))
    np.testing.assert_allclose(M.episode_logweights(other, uniform, 0), uniform)


def test_shared_profile_law_is_product_of_episode_mixtures_not_kernel_means():
    b = grouped_bundle()
    early, late = np.array([0, 5, 2]), np.array([7, 1, 0])
    ell0 = B.component_log_likelihood(b, [early], False)
    ell1 = B.component_log_likelihood(b, [late], False)
    fine = C.posterior_logweights(ell0)
    projected = M.episode_logweights(b, fine, 0)
    row = C.observe_late(b, late, projected)
    numerator = 0.0
    denominator = np.exp(ell0).mean()
    for group in (0, 31):
        ix = b["codes"][0] == group
        numerator += ix.mean() * np.exp(ell0[ix]).mean() * np.exp(ell1[ix]).mean()
    np.testing.assert_allclose(row["logp"].sum(), np.log(numerator / denominator))
    mean_bundle = {
        **b,
        "alpha": b["alpha"].mean(0, keepdims=True),
        "alpha0": b["alpha0"].mean(0, keepdims=True),
    }
    assert (
        abs(
            B.component_log_likelihood(mean_bundle, [late], False)[0]
            - np.log(np.exp(ell1).mean())
        )
        > 0.001
    )


def test_missing_source_profile_kept_as_unsupported_not_neutral_component():
    b = grouped_bundle()
    bank = M.profile_bank(b, 0)
    assert bank["counts"][1] == 0
    w = M.episode_logweights(b, np.log(np.array([0.2, 0.5, 0.3])), 0)
    assert np.isfinite(w).all()
    assert B.profile(np.exp(w), b["codes"])[0, 1] == 0


def test_coherent_full_and_group_laws_have_same_first_episode():
    b = grouped_bundle()
    early = np.array([0, 5, 2])
    ell = B.component_log_likelihood(b, [early], False)
    grouped_marginal = 0.0
    for group in (0, 31):
        ix = b["codes"][0] == group
        grouped_marginal += ix.mean() * np.exp(ell[ix]).mean()
    np.testing.assert_allclose(grouped_marginal, np.exp(ell).mean())
    projected = M.episode_logweights(b, C.posterior_logweights(ell), 0)
    np.testing.assert_allclose(logsumexp(projected), 0, atol=1e-14)


def test_cold_all_models_have_explicit_selected_anchor_and_support():
    b, r = grouped_bundle(), record()
    output = M.cold(b, r)
    rows = output["models"]
    assert set(rows) == set(M.MODELS)
    full = np.array(rows["full"]["boundary_all_anchor_profiles"])
    actual = np.array(rows["shared_actual_profile"]["boundary_all_anchor_profiles"])
    np.testing.assert_allclose(full[0], actual[0])
    assert rows["shared_actual_profile"]["source_supported_profiles"][1] is False
    bad = copy.deepcopy(r)
    bad["big5"] = [0] * 5
    with pytest.raises(ValueError):
        M.cold(b, bad)


@pytest.mark.parametrize("assignment", [2, -1, True])
def test_unknown_anchor_contract_rejected(assignment):
    with pytest.raises(ValueError):
        M.episode_logweights(grouped_bundle(), np.full(3, -np.log(3)), assignment)
