"""Synthetic tests for PA01's exact action policy."""

import numpy as np

from grain_research.measurement_discovery.adaptive_discrimination import (
    expected_information_gain,
    update_posterior,
)


def test_information_gain_and_posterior():
    candidates = [("M1_axis_1", .5), ("M2_axis_2", .5)]
    weights = np.array([.5, .5])
    assert expected_information_gain(weights, candidates, (1, 1, 1)) >= 0
    posterior = update_posterior(weights, candidates, (1, 1, 1), 1)
    assert np.isclose(posterior.sum(), 1)


def test_invalid_action_rejected():
    from grain_research.measurement_discovery.adaptive_discrimination import choose_action

    try:
        choose_action(np.array([1.0]), [("M1_axis_1", .5)], "bad", np.random.default_rng(1))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid action should fail")
