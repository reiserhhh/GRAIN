#!/usr/bin/env python3
"""F07 openness channel checks."""

import numpy as np

from grain_research.gptpro_foundation_v3.f07 import channel_derivative, channel_prob, fisher_information


def test_probability_and_derivative_are_bounded():
    for world in ("U", "D", "P", "flat"):
        p = channel_prob(.3, .4, world)
        assert 0.0 < p < 1.0
        if world == "flat":
            assert channel_derivative(.3, .4, world) == 0.0


def test_fisher_is_nonnegative_and_derivative_matches_numeric():
    h = 1e-5
    for world in ("U", "D", "P"):
        numerical = (channel_prob(.3 + h, .4, world) - channel_prob(.3 - h, .4, world)) / (2 * h)
        np.testing.assert_allclose(numerical, channel_derivative(.3, .4, world), atol=1e-5)
        assert fisher_information(.3, [-1.0, 0.0, 1.0], world) >= 0.0
