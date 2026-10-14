"""Synthetic coverage of ordered encoding, source fitting and zero-action structure."""

import numpy as np
import pytest

from grain_research.construction import math_ordered_moments as M


@pytest.mark.parametrize("tokens", [1, 2, 3, 4])
def test_complete_encoding_reconstructs_full_token_matrix(tokens):
    h = np.random.default_rng(tokens).normal(size=(tokens, 23))
    moments = M.ordered_moments(h, tokens - 1)
    np.testing.assert_allclose(M.reconstruct_moments(moments, tokens), h, atol=1e-14)


@pytest.mark.parametrize("tokens", [1, 2, 3, 4])
def test_reversal_parity_and_single_token_odd_zero(tokens):
    h = np.random.default_rng(14).normal(size=(tokens, 7))
    original, reversed_m = M.ordered_moments(h), M.ordered_moments(h[::-1])
    np.testing.assert_allclose(
        reversed_m, original * (-1.0) ** np.arange(4)[:, None], atol=1e-15
    )
    if tokens == 1:
        np.testing.assert_array_equal(original[1:], np.zeros((3, 7)))


def test_mean_collision_is_separated_by_odd_position_channel():
    h = np.array([[1, 2], [3, 4]], float)
    a, b = M.ordered_moments(h), M.ordered_moments(h[::-1])
    np.testing.assert_array_equal(a[0], b[0])
    assert np.linalg.norm(a[1] - b[1]) > 0


def test_known_length_required_for_complete_reconstruction():
    with pytest.raises(ValueError):
        M.reconstruct_moments(np.zeros((2, 3)), 4)


@pytest.mark.parametrize("h", [np.zeros(3), np.zeros((0, 3)), np.array([[np.nan]])])
def test_bad_token_states_fail(h):
    with pytest.raises(ValueError):
        M.ordered_moments(h)


def test_fixed_feature_shapes_and_no_question_input():
    frame = np.eye(5)[:, :2]
    bodies = {name: np.ones((2, 5)) for name in M.SPANS}
    observation = M.observation(bodies, frame)
    assert observation.shape == (24,)
    assert [len(M.feature_indices(arm)) for arm in M.ARMS] == [0, 12, 16, 24]
    np.testing.assert_array_equal(observation[12:16], np.zeros(4))
    bodies["Q"] = np.zeros((1, 5))
    with pytest.raises(ValueError):
        M.observation(bodies, frame)


def test_family_balancing_equalizes_loss_mass():
    families = np.array([0, 1, 1, 1])
    weights = M.family_weights(families)
    assert weights.sum() == pytest.approx(4)
    assert weights[families == 0].sum() == pytest.approx(weights[families == 1].sum())


def test_source_ridge_matches_weighted_augmented_least_squares():
    rng = np.random.default_rng(141)
    x = rng.normal(size=(30, 4))
    x[:, 3] = 0
    y = x[:, :3] @ np.array([0.2, -0.3, 0.1])
    families = [0] * 10 + [1] * 20
    function = M.compile_action(x, y, families)
    z = x / function["scale"]
    weights = M.family_weights(families)
    a = np.vstack((np.sqrt(weights[:, None]) * z, np.eye(4)))
    target = np.r_[np.sqrt(weights) * y, np.zeros(4)]
    alternative = np.linalg.lstsq(a, target, rcond=None)[0]
    np.testing.assert_allclose(function["coefficient"], alternative, atol=1e-14)
    assert function["active"] == [True, True, True, False]
    np.testing.assert_array_equal(
        M.apply_action(np.zeros((3, 4)), function), np.zeros(3)
    )


def test_nonzero_constant_direction_is_not_lost_by_std_centering():
    x = np.ones((10, 1))
    function = M.compile_action(x, np.full(10, 0.2), [0] * 10)
    assert function["active"] == [True]
    assert M.apply_action(x, function).mean() > 0


def test_zero_observation_has_exact_zero_action():
    function = M.compile_action(np.empty((4, 0)), np.ones(4), [0, 0, 1, 1])
    np.testing.assert_array_equal(
        M.apply_action(np.empty((5, 0)), function), np.zeros(5)
    )


def test_frozen_scoring_does_not_depend_on_application_batch():
    x, y = np.eye(3), np.array([0.2, 0.1, -0.1])
    function = M.compile_action(x, y, [0, 0, 1])
    held = np.array([[3.0, 1.0, -2.0]])
    alone = M.apply_action(held, function)[0]
    enlarged = np.vstack((held, [[999.0, -999.0, 42.0]]))
    assert M.apply_action(enlarged, function)[0] == pytest.approx(alone)


def test_prepare_refuses_existing_packet(tmp_path):
    from scripts.run_grain_math_ordered_moments import prepare

    with pytest.raises(FileExistsError):
        prepare(tmp_path)
