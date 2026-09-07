"""Finite mechanism possibilities, not experimental confirmation on people."""

from math import exp

import pytest

from scripts.check_grain_innovation_witnesses import (
    posterior_from_likelihoods,
    seminar_challenges,
    witnesses,
)


def test_full_factorial_finds_binding_while_main_effects_miss_it():
    data = witnesses()["query_role_interaction"]
    assert data["cells"] == 8
    assert data["main_evidence"] == data["main_role"] == 0
    assert data["main_query"] == 3
    assert data["triple_interaction"] == 2


def test_local_paired_action_can_survive_global_cancellation():
    data = witnesses()["paired_action_without_global_axis"]
    assert data["mean_vector_norm"] < 1e-12
    assert data["mean_global_action"] == pytest.approx(0, abs=1e-12)
    assert data["mean_local_action"] == pytest.approx(1)
    assert data["quarter_turn_mismatched_action"] == pytest.approx(0, abs=1e-12)


def test_noncommutativity_can_be_simple_recency():
    data = witnesses()["order_without_reasoning"]
    assert data["positive_then_negative"] == -0.5
    assert data["negative_then_positive"] == 0.5
    assert data["order_difference"] == -1


def test_interactions_can_carry_all_information_without_marginal_covariance():
    data = witnesses()["synergy_without_marginal_signal"]
    assert data["covariance_with_x"] == data["covariance_with_y"] == 0
    assert data["product_readout_mse"] == 0
    assert data["additive_least_squares_mse"] == pytest.approx(1)


def test_context_and_reference_have_distinct_roles():
    data = witnesses()["condition_and_reference"]
    assert data["voluntary_same_action_posterior"] == pytest.approx(0.8)
    assert data["compulsory_same_action_posterior"] == pytest.approx(0.5)
    assert data["voluntary_rare_group_prior_posterior"] == pytest.approx(4 / 13)
    assert (
        data["voluntary_log_likelihood_ratio"] > data["compulsory_log_likelihood_ratio"]
    )


@pytest.mark.parametrize("values", [(0, 0.2, 0.5), (0.8, 2, 0.5), (0.8, 0.2, 1)])
def test_probability_contract(values):
    with pytest.raises(ValueError, match="Likelihoods"):
        posterior_from_likelihoods(*values)


def test_four_cell_average_does_not_identify_psychological_content():
    data = seminar_challenges()["factorial_mean_not_unique_semantics"]
    assert data["four_cell_mean"] == pytest.approx(0.15)
    assert data["nuisance_world_semantic_component"] == 0
    assert data["observable_max_difference"] == 0


def test_order_cycle_rejects_only_the_declared_additive_scale():
    data = seminar_challenges()["order_cycle_not_unique_updating"]
    assert data["linear_margin"] == 0
    expected = (
        1 / (1 + exp(-5)) - 2 / (1 + exp(-4)) - 1 / (1 + exp(-1)) + 2 / (1 + exp(-2))
    )
    assert data["sigmoid_probability"] == pytest.approx(expected)
    assert data["static_cubic_margin"] == 12


def test_mass_competition_can_look_like_negative_reference_comparison():
    data = seminar_challenges()["negative_context_slope_without_reference_comparison"]
    assert data["outputs"][0] > data["outputs"][1] > data["outputs"][2]
    assert data["derivative_at_zero"] == pytest.approx(-1 / 9)
    assert data["cancelling_assimilation_plus_contrast"] == [0, 0, 0]


def test_budget_counts_are_not_reported_as_neural_forwards():
    data = seminar_challenges()["literal_budget_arithmetic"]
    assert data["two_sites_four_directions_three_doses_two_signs"] == 4608
    assert data["plus_baselines_and_secondary_and_direction_examples"] == 5428


def test_balanced_probe_responses_recover_declared_gradient_kernel():
    data = seminar_challenges()["coded_probe_functional_kernel"]
    assert data["identity_max_error"] < 1e-12
    assert data["same_function_signature_max_error"] == 0
    assert data["kernel"][0][1] == pytest.approx(5 / 3)
    assert data["kernel"][0][2] == pytest.approx(0, abs=1e-12)


def test_joint_minus_edge_does_not_equal_an_indirect_effect():
    data = seminar_challenges()["patch_subtraction_is_not_mediation"]
    assert data["joint_minus_edge"] != data["other_only"] - data["base"]
    assert (
        data["interaction"]
        == data["joint"] - data["edge_only"] - data["other_only"] + data["base"]
    )


def test_equal_baselines_can_hide_context_dependent_sensitivity():
    data = seminar_challenges()["context_can_change_sensitivity_at_equal_baseline"]
    assert data["baseline_at_zero_evidence"][0] == data["baseline_at_zero_evidence"][1]
    assert data["evidence_sensitivities"][0] != data["evidence_sensitivities"][1]
