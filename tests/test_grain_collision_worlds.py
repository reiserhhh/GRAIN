"""Exact tests for the finite GP01 collision worlds."""

import numpy as np

from grain_research.measurement_discovery.collision_worlds import (
    c1_law,
    c2_lag3_equal_probability,
    c2_triple_law,
    c3_exposure_bounds,
    distance_chain,
)


def test_c1_marginal_and_condition_differ():
    rows = c1_law()
    assert np.mean([row["p_y1"] for row in rows if row["type"] == -1]) == 0.5
    assert rows[0]["p_y1"] != rows[1]["p_y1"]


def test_c2_triples_and_lag3():
    assert np.isclose(sum(c2_triple_law("xor").values()), 1)
    assert c2_triple_law("xor")["000"] == 0.1875
    assert c2_triple_law("xor")["001"] == 0.0625
    assert c2_lag3_equal_probability("xor") == 0.625
    assert c2_lag3_equal_probability("iid") == 0.5


def test_c3_same_observed_rate_different_exposure():
    assert {row["p_selected"] for row in c3_exposure_bounds()} == {0.4}
    assert c3_exposure_bounds()[0]["q"] != c3_exposure_bounds()[1]["q"]


def test_distance_threshold_is_not_transitive():
    result = distance_chain()
    assert result["ab"] <= 1 and result["bc"] <= 1 and result["ac"] > 1
    assert not result["transitive_at_epsilon"]
