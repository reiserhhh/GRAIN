#!/usr/bin/env python3
"""Small mathematical witnesses for choice/response and learned design rules."""

import numpy as np
import pytest
from scipy.special import expit

from grain_research.choice_design import (
    anchored_cells,
    choice_coordinates,
    choose_allocation,
    design_gain,
    learn_cell_moments,
    linear_cell_score,
)


def moments():
    """Fixed positive two-cell moments with nonzero prior means."""
    return {
        "mean": np.array([0.1, -0.2]),
        "covariance": np.array([[0.4, 0.15], [0.15, 0.3]]),
        "noise": np.array([0.7, 0.9]),
    }


def test_fixed_assignment_has_no_choice_score():
    score, count = choice_coordinates([], [])
    assert np.isnan(score).all()
    assert (count == 0).all()


def test_opportunity_offset_is_exact_and_missing_tail_is_null():
    raw, counts = choice_coordinates([-1, -1, -1], [1, 1, -1])
    adjusted, _ = choice_coordinates([-1, -1, -1], [1, 1, -1], [1.2, -0.3, 8.0])
    np.testing.assert_allclose(adjusted[:2], raw[:2] - [1.2, -0.3])
    assert np.isnan(adjusted[2])
    assert counts.tolist() == [3, 3, 0]


def test_unknown_opportunity_nonidentifiability():
    gamma, offset, shift = np.array([0.5, -0.2]), np.array([1.0, 0.7]), 2.3
    np.testing.assert_allclose(
        expit(gamma + offset), expit((gamma + shift) + (offset - shift))
    )


def test_response_retains_level_and_contrast_not_frequency_mix():
    z = np.r_[np.full(9, -1), 1]
    y = 0.2 + 0.4 * z
    score, _ = anchored_cells(z, y)
    np.testing.assert_allclose(score, [0.2, 0.4])
    assert not np.isclose(y.mean(), score[0])


def test_absent_response_cell_is_not_filled_with_zero():
    score, counts = anchored_cells([-1, -1], [0.2, 0.4])
    assert np.isnan(score).all()
    assert counts.tolist() == [2, 0]


def test_source_moments_remove_finite_event_noise():
    rng = np.random.default_rng(12)
    truth = rng.multivariate_normal([0.1, -0.2], [[0.4, 0.15], [0.15, 0.3]], size=6000)
    pilot = (
        truth[..., None]
        + rng.normal(size=(6000, 2, 20)) * np.sqrt([0.7, 0.9])[None, :, None]
    )
    fit = learn_cell_moments(pilot)
    np.testing.assert_allclose(fit["covariance"], moments()["covariance"], atol=0.025)
    np.testing.assert_allclose(fit["noise"], moments()["noise"], atol=0.01)


def test_score_matches_gaussian_block_conditioning():
    z, y = np.array([-1, -1, 1]), np.array([0.4, 0.7, -0.5])
    m = moments()
    x = np.eye(2)[(z == 1).astype(int)]
    covariance = m["covariance"]
    k = (
        covariance
        @ x.T
        @ np.linalg.inv(
            x @ covariance @ x.T + np.diag(m["noise"][(z == 1).astype(int)])
        )
    )
    expected = m["mean"] + k @ (y - x @ m["mean"])
    fit = linear_cell_score(z, y, ["a", "b", "c"], m)
    np.testing.assert_allclose(fit["score"], expected)
    np.testing.assert_allclose(fit["covariance"], covariance - k @ x @ covariance)
    np.testing.assert_allclose(fit["score"], fit["observed"] + fit["borrowed"])
    np.testing.assert_allclose(
        fit["data_weights"] + fit["prior_weights"], np.eye(2), atol=1e-15
    )


def test_perfect_repetition_adds_zero_information():
    first = linear_cell_score([-1, 1], [0.2, 0.7], ["a", "b"], moments())
    repeated = linear_cell_score(
        [-1, 1, -1, 1], [0.2, 0.7, 0.2, 0.7], ["a", "b", "a", "b"], moments()
    )
    np.testing.assert_allclose(first["score"], repeated["score"])
    np.testing.assert_allclose(first["covariance"], repeated["covariance"])
    assert repeated["unique_events"] == 2 and repeated["reports"] == 4
    np.testing.assert_equal(
        design_gain(first["covariance"], [0.5, 0.5], [0.7, 0.9], repeated=True), [0, 0]
    )


def test_conflicting_repetition_requires_revision():
    with pytest.raises(ValueError, match="revision"):
        linear_cell_score([-1, -1], [0.2, 0.3], ["a", "a"], moments())


def test_information_gain_is_actual_covariance_difference():
    m = moments()
    target = np.array([-0.5, 0.5])
    gain = design_gain(m["covariance"], target, m["noise"])
    for index, condition in enumerate((-1, 1)):
        fit = linear_cell_score([condition], [0.1], ["new"], m)
        np.testing.assert_allclose(
            gain[index], target @ (m["covariance"] - fit["covariance"]) @ target
        )


def test_design_does_not_accept_future_outcomes():
    counts, trace = choose_allocation(moments(), [0, 1], 16)
    assert counts.sum() == 16 and counts.min() >= 2
    assert counts[1] > counts[0]
    assert len(trace) == 12


def test_one_cell_borrowing_is_explicit_not_direct_observation():
    fit = linear_cell_score([1, 1], [0.9, 0.8], ["a", "b"], moments())
    assert fit["counts"][0] == 0
    assert fit["data_weights"][0, 1] > 0
    assert fit["data_weights"][0, 0] == 0
    assert fit["covariance"][0, 0] > 0


@pytest.mark.parametrize(
    "root,tail", [([0], [1]), ([1, -1], [1]), ([1], [float("nan")])]
)
def test_choice_invalid_inputs(root, tail):
    with pytest.raises(ValueError):
        choice_coordinates(root, tail)


def test_source_pilot_rejects_missing_or_degenerate_values():
    with pytest.raises(ValueError):
        learn_cell_moments(np.zeros((5, 2, 4)))
    with pytest.raises(ValueError):
        learn_cell_moments(np.full((5, 2, 4), np.nan))
