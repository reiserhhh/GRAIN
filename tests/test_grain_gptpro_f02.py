#!/usr/bin/env python3
"""F02 finite-world checks."""

import itertools

import pytest

from grain_research.gptpro_foundation_v3.f02 import c1_probability, c2_next_probability, future_distribution


def test_c1_exact_probabilities():
    assert c1_probability(1, 1) == pytest.approx(0.75)
    assert c1_probability(-1, 1) == pytest.approx(0.25)


def test_c2_future_distribution_normalizes_and_diverges():
    for history in itertools.product((0, 1), repeat=2):
        assert sum(future_distribution(history, 6, "xor").values()) == pytest.approx(1.0)
    assert future_distribution((0, 1), 2, "xor") != future_distribution((0, 1), 2, "iid")


def test_c2_transition_is_second_order():
    assert c2_next_probability(0, 1, 1, "xor") == pytest.approx(0.75)
    assert c2_next_probability(0, 1, 0, "xor") == pytest.approx(0.25)
