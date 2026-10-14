#!/usr/bin/env python3
"""Mathematical invariants and estimator/truth isolation for paired readouts."""

import numpy as np
import pytest

from grain_research.construction import math_response as M


def test_gls_dual_same_information():
    rng = np.random.default_rng(71)
    matrix = rng.normal(size=(12, 4))
    ell = np.array([1.0, 0, 0.5, 0])
    dual = M.target_dual(matrix, ell)
    np.testing.assert_allclose(matrix.T @ dual["w"], ell, atol=1e-12)
    np.testing.assert_allclose(dual["w"], M.gls_target(matrix, ell), atol=1e-12)


def test_infeasible_target_has_kernel_witness():
    d = np.array([1.0, 2, -1])
    matrix = np.column_stack([d, d])
    ell = np.array([1.0, 0])
    assert M.target_dual(matrix, ell)["status"] == "unresolved"
    witness = M.infeasibility_witness(matrix, ell)
    assert witness["exists"] and witness["map_residual"] < 1e-12
    assert abs(witness["target_contrast"]) > 0.5


def test_near_collision_is_not_exact_impossibility():
    matrix = np.array([[1.0, 1], [0, 1e-6]])
    dual = M.target_dual(matrix, np.array([1.0, 0]))
    assert dual["status"] == "identified"
    assert dual["noise_variance"] > 1e11
    assert not M.infeasibility_witness(matrix, np.array([1.0, 0]))["exists"]


def test_walsh_design_and_eligibility():
    bits = np.array(list(M.product((-1, 1), repeat=4)))
    q, names = M.effect_design(bits)
    np.testing.assert_allclose(q.T @ q, np.eye(15) * 16)
    y = M.text_targets(bits)
    assert np.all(y[bits[:, 2] == -1] == 0)
    assert names.index("0*2") >= 0


@pytest.mark.parametrize("world", M.WORLDS)
def test_finite_roles_are_separated_and_truth_not_in_view(world):
    views, truth = M.finite_world(71, 0.6, world)
    assert not (set(views["source"].group) & set(views["application"].group))
    assert not (set(views["source"].group) & set(views["validation"].group))
    assert "M" not in vars(views["source"])
    assert "theta" not in vars(views["source"])
    assert truth["M"].shape == (64, 4)


def test_family_offset_does_not_change_forward_columns():
    rng = np.random.default_rng(17)
    q = np.tile(np.array(list(M.product((-1, 1), repeat=2))), (4, 1))
    g = np.repeat(np.arange(4), 4)
    x = q @ rng.normal(size=(2, 7))
    shifted = x + np.repeat(rng.normal(size=(4, 7)), 4, axis=0)
    np.testing.assert_allclose(
        M.fit_loadings(x, q, g)[0], M.fit_loadings(shifted, q, g)[0], atol=1e-12
    )


def test_fit_does_not_read_test_vectors_or_targets():
    rng = np.random.default_rng(91)
    bits = np.tile(np.array(list(M.product((-1, 1), repeat=4))), (6, 1))
    q, _ = M.effect_design(bits)
    x = q @ rng.normal(size=(15, 30)) + rng.normal(scale=0.02, size=(96, 30))
    group = np.repeat(np.arange(6), 16)
    roles = np.repeat(np.array(["source"] * 3 + ["validation"] * 2 + ["test"]), 16)
    first, _ = M.fit_text_models(x, bits, group, roles)
    x[roles == "test"] = 1e8
    bits[roles == "test"] *= -1
    second, _ = M.fit_text_models(x, bits, group, roles)
    for arm in first:
        np.testing.assert_array_equal(first[arm]["w"], second[arm]["w"])


def test_joint_unit_is_not_additive():
    bits = np.array(list(M.product((-1, 1), repeat=4)))
    y = M.text_targets(bits)
    effects = M.family_effects(y, bits, np.zeros(16, int), 0)[0]
    assert effects["target_gain"] == 2
    assert effects["other_target_gain"] == 0
    assert effects["joint_pole_owner"] == 2
    assert effects["style_rms"] == 0


def test_cold_score_no_label_or_id_dependency():
    model = {"w": np.eye(2), "mu": np.ones(2), "intercept": np.zeros(2)}
    np.testing.assert_array_equal(M.score_bundle(model, np.array([[2.0, 3]])), [[1, 2]])


def test_family_bootstrap_reproducible():
    values = np.array([1.0, 2, 3, 4])
    assert M.bootstrap_interval(values) == M.bootstrap_interval(values)
    assert M.bootstrap_interval(np.ones(4)) == [1, 1]


def test_strong_single_actions_joint_calibration_not_probability_product():
    bits = np.tile(np.array(list(M.product((-1, 1), repeat=4))), (6, 1))
    group = np.repeat(np.arange(6), 16)
    roles = np.repeat(np.array(["source"] * 3 + ["validation"] * 2 + ["test"]), 16)
    # No joint columns in the observation; joint meaning must be constructed.
    x = bits[:, [0, 2, 3]].astype(float)
    joint, _ = M.fit_binding_model(x, bits, group, roles, True)
    additive, _ = M.fit_binding_model(x, bits, group, roles, False)
    held = roles == "test"
    truth = M.text_targets(bits[held])
    assert np.mean((M.score_bundle(joint, x[held]) - truth) ** 2) < 1e-4
    assert np.mean((M.score_bundle(additive, x[held]) - truth) ** 2) > 0.2
    # Raw factor coordinates are signed real values, not calibrated probabilities.
    assert np.any(x < 0)


def test_binding_fit_cannot_select_using_confirmation():
    rng = np.random.default_rng(37)
    bits = np.tile(np.array(list(M.product((-1, 1), repeat=4))), (6, 1))
    group = np.repeat(np.arange(6), 16)
    roles = np.repeat(
        np.array(["source"] * 3 + ["validation"] * 2 + ["confirmation"]), 16
    )
    x = bits[:, [0, 2, 3]].astype(float) + rng.normal(scale=0.03, size=(96, 3))
    first, _ = M.fit_binding_model(x, bits, group, roles, True)
    x[roles == "confirmation"] = 10000
    bits[roles == "confirmation"] *= -1
    second, _ = M.fit_binding_model(x, bits, group, roles, True)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_dynamics_metric_and_normal_rotation_counterexample():
    rows, fits = M.dynamics_probe()
    assert max(r["gauge_error"] for r in rows) < 1e-11
    rotation = [r for r in rows if r["system"] == "normal_rotation"]
    assert rotation[0]["response"] == 0 and abs(rotation[1]["response"]) > 0
    assert max(r["gain"] for r in rotation) == 1
    assert max(r["gain"] for r in rows if r["system"] == "nonnormal") > 8
    assert fits["nonnormal"]["A_error"] < 0.02
