#!/usr/bin/env python3
"""Mathematical toy checks, not empirical evidence of psychological validity."""

import numpy as np

from scripts.check_grain_math_seminar_ii_20260908 import (
    distribution_witness,
    finite_nonlinearity_witness,
    fisher_witness,
    inverse_witness,
    observational_equivalence_witness,
    query_null_witness,
    repeated_representation_witness,
    shared_aggregator_witness,
    symmetry_witness,
    synergy_witness,
)


def test_fisher_minimizes_local_cost_for_the_same_target():
    w = fisher_witness()
    assert np.allclose(w["target_changes"], [0.001, 0.001])
    assert w["euclidean_to_optimal_cost_ratio"] > 1.0
    assert w["relative_quadratic_error"] < 0.01


def test_query_null_is_relative_to_which_queries_are_asked():
    w = query_null_witness()
    assert w["two_query_response"] == [0.0, 0.0]
    assert w["three_query_response"] == [0.0, 0.0, 1.0]
    assert w["direction_norm"] == 1.0


def test_unjustified_invariance_erases_the_role_observable():
    w = symmetry_witness()
    assert w["fixed_role_scores"] == [1.0, -1.0]
    assert w["role_swap_group_average"] == 0.0
    assert w["coordinate_and_dual_role_transformed_score"] == 1.0


def test_synergy_is_not_marginal_correlation_or_noncommutativity():
    w = synergy_witness()
    assert w["marginal_correlations"] == [0.0, 0.0]
    assert w["marginal_information_bits"] == [0.0, 0.0]
    assert w["joint_information_bits"] == 1.0
    assert w["mixed_edit_difference"] == 1.0
    assert w["input_edit_commutator_norm"] == 0.0


def test_mean_pooling_is_not_distribution_identification():
    w = distribution_witness()
    assert w["ordinary_means"] == [0.0, 0.0]
    assert w["gaussian_kernel_mmd_squared"] > 0.3


def test_score_equivalence_does_not_determine_the_mechanism():
    w = observational_equivalence_witness()
    assert w["observational_mse"] == 0.0
    assert w["same_edit_model_responses"] == [1.0, 0.0]


def test_generative_likelihood_is_not_its_own_inverse():
    w = inverse_witness()
    assert np.allclose(w["posteriors_for_same_text_event"], [0.8, 4 / 13])


def test_rank_two_attribution_does_not_require_query_dependent_aggregation():
    w = shared_aggregator_witness()
    assert w["query_independent_aggregate"] == [1.0, 1.0]
    assert w["attribution_rank"] == 2


def test_schur_scalar_conclusion_does_not_apply_to_repeated_components():
    w = repeated_representation_witness()
    assert w["commutator_norm"] == 0.0
    assert w["repeated_component_off_diagonal"] == 2.0
    assert np.isclose(w["transport_determinant"], 3.0)


def test_zero_local_sensitivity_is_not_finite_invariance():
    w = finite_nonlinearity_witness()
    assert w["local_gradient"] == [0.0, 0.0]
    assert w["linear_prediction"] == 0.0
    assert w["finite_response"] == 1.0
