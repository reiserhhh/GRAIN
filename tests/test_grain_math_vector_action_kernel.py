"""Synthetic tests of the full-vector observation, source Gram law and geometry."""

import numpy as np
import pytest

from grain_research.construction import math_ordered_moments as O
from grain_research.construction import math_vector_action_kernel as K


def test_projected_full_observation_matches_original_typed_moments():
    rng = np.random.default_rng(15)
    frame = np.linalg.qr(rng.normal(size=(7, 2)))[0]
    body = {name: rng.normal(size=(1 + (i % 4), 7)) for i, name in enumerate(O.SPANS)}
    full = K.full_observation(body)
    assert full.shape == (84,)
    np.testing.assert_allclose(
        K.projected_observation(full, frame), O.observation(body, frame), atol=1e-14
    )


def test_fixed_shape_rejects_query_and_unmatched_hidden_width():
    body = {name: np.ones((2, 5)) for name in O.SPANS}
    body["Q"] = np.ones((1, 5))
    with pytest.raises(ValueError):
        K.full_observation(body)
    del body["Q"]
    body["A_property"] = np.ones((2, 6))
    with pytest.raises(ValueError):
        K.full_observation(body)


def test_full_direction_outside_original_rank2_remains_visible():
    frame = np.eye(5)[:, :2]
    body = {name: np.zeros((2, 5)) for name in O.SPANS}
    original = K.full_observation(body)
    body["A_property"][:, 4] = [1, -1]
    delta = K.full_observation(body) - original
    assert np.linalg.norm(delta) > 0
    np.testing.assert_array_equal(K.projected_observation(delta, frame), np.zeros(24))


def test_weighted_dual_matches_independent_augmented_primal():
    rng = np.random.default_rng(151)
    x = rng.normal(size=(23, 8))
    y = rng.normal(size=23)
    families = [0] * 4 + [1] * 19
    function = K.fit_action(x, y, families)
    weights = O.family_weights(families)
    z = x / function["scale"]
    a = np.vstack((np.sqrt(weights[:, None]) * z, np.eye(8)))
    target = np.r_[np.sqrt(weights) * y, np.zeros(8)]
    alternative = np.linalg.lstsq(a, target, rcond=None)[0] / function["scale"]
    np.testing.assert_allclose(function["coefficient"], alternative, atol=1e-14)
    new = rng.normal(size=(7, 8))
    np.testing.assert_allclose(
        K.apply_action(new, function["coefficient"]),
        K.kernel_prediction(new, x, function["weighted_dual"], function["scale"]),
        atol=1e-14,
    )


def test_global_source_rms_and_kernel_are_hidden_isometry_invariant():
    rng = np.random.default_rng(152)
    x = rng.normal(size=(19, 12, 5))
    rotation = np.linalg.qr(rng.normal(size=(5, 5)))[0]
    y = rng.normal(size=19)
    families = [0] * 6 + [1] * 13
    a = K.fit_action(x.reshape(19, -1), y, families)
    b = K.fit_action((x @ rotation).reshape(19, -1), y, families)
    np.testing.assert_allclose(a["scale"], b["scale"], atol=1e-14)
    np.testing.assert_allclose(a["weighted_dual"], b["weighted_dual"], atol=1e-14)
    new = rng.normal(size=(3, 12, 5))
    np.testing.assert_allclose(
        K.apply_action(new.reshape(3, -1), a["coefficient"]),
        K.apply_action((new @ rotation).reshape(3, -1), b["coefficient"]),
        atol=1e-14,
    )


def test_psd_and_rank_diagnostic_accepts_redundant_source_rows():
    x = np.tile([1.0, -2.0], (8, 1))
    function = K.fit_action(x, np.ones(8), [0] * 8)
    assert function["numerical_source_rank"] == 1
    assert function["gram_eigen_min"] >= -function["gram_rank_tolerance"]
    assert function["system_residual_max"] < 1e-12


@pytest.mark.parametrize("width", [0, 3])
def test_identical_state_zero_and_nonexistent_observation_action(width):
    x = np.zeros((6, width))
    function = K.fit_action(x, np.ones(6), [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(
        K.apply_action(np.zeros((4, width)), function["coefficient"]), np.zeros(4)
    )
    assert function["numerical_source_rank"] == 0


def test_frozen_action_is_not_restandardized_by_application_batch():
    function = K.fit_action(np.eye(3), np.array([0.1, 0.4, -0.2]), [0, 0, 1])
    alone = K.apply_action(np.array([[3.0, 2.0, 1.0]]), function["coefficient"])[0]
    enlarged = np.array([[3.0, 2.0, 1.0], [999.0, -111.0, 47.0]])
    assert K.apply_action(enlarged, function["coefficient"])[0] == pytest.approx(alone)


@pytest.mark.parametrize(
    "x,y", [(np.ones((3, 2)), np.ones(2)), (np.array([[np.nan], [0]]), np.ones(2))]
)
def test_invalid_source_data_fails(x, y):
    with pytest.raises(ValueError):
        K.fit_action(x, y, [0] * len(x))


def test_prepare_refuses_existing_packet(tmp_path):
    from scripts.run_grain_math_vector_action_kernel import prepare

    with pytest.raises(FileExistsError):
        prepare(tmp_path)
