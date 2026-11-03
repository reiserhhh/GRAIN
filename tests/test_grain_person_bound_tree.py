#!/usr/bin/env python3
"""Designed observation aliases,exact tree coordinates and qualified cold records."""

import numpy as np
import pytest
from scipy.special import logsumexp

from grain_research.construction import math_person_bound_response as O
from grain_research.construction import math_person_bound_tree as M
from test_grain_person_bound_response import cold_fixture


def test_tree_is_exact_and_gauge_invariant():
    z = np.arange(24.0).reshape(8, 3) / 3
    fields = M.tree_coordinates(z)
    recovered = M.tree_log_probabilities(fields["direction"], fields["assessability"])
    assert recovered == pytest.approx(z - logsumexp(z, -1, keepdims=True))
    shifted = M.tree_coordinates(z + np.arange(8)[:, None] * 1000)
    assert shifted["direction"] == pytest.approx(fields["direction"])
    assert shifted["assessability"] == pytest.approx(fields["assessability"])


def test_binary_alias_does_not_identify_availability_or_support():
    worlds = []
    for available, support in ((0.5, 0.8), (0.8, 0.5)):
        worlds.append(
            np.array([available * (1 - support), available * support, 1 - available])
        )
    assert worlds[0][1] == pytest.approx(worlds[1][1])
    assert not np.allclose(worlds[0], worlds[1])
    a, b = (M.tree_coordinates(np.log(w)) for w in worlds)
    assert a["direction"] != pytest.approx(b["direction"])
    assert a["assessability"] != pytest.approx(b["assessability"])


def test_probability_tree_keeps_both_coordinates_independent():
    d, a = np.array([-3.0, 0, 3.0]), np.array([1.0, 1.0, 1.0])
    fields = M.tree_coordinates(M.tree_log_probabilities(d, a))
    assert fields["direction"] == pytest.approx(d)
    assert fields["assessability"] == pytest.approx(a)
    fields = M.tree_coordinates(M.tree_log_probabilities(a, d))
    assert fields["direction"] == pytest.approx(a)
    assert fields["assessability"] == pytest.approx(d)


def test_extreme_finite_logits_are_not_clipped():
    z = np.array([[-1000.0, 0, 1000], [1000.0, -1000, 0]])
    fields = M.tree_coordinates(z)
    recovered = M.tree_log_probabilities(fields["direction"], fields["assessability"])
    assert np.isfinite(recovered).all()
    assert recovered == pytest.approx(fields["log_probabilities"])


def test_roles_do_not_confuse_availability_with_direction():
    value = np.array([o * t + 3 * d for o, t, d in O.SIGNS]).reshape(1, 2, 2, 2)
    fields = M.tree_coordinates(M.tree_log_probabilities(np.zeros_like(value), value))
    assert O.character_accounts(fields["direction"])[0, 3] == pytest.approx(0)
    assert O.character_accounts(fields["assessability"])[0, 3] == pytest.approx(1)


@pytest.mark.parametrize("bad", [np.zeros((2, 2)), np.full((2, 3), np.nan)])
def test_failed_or_binary_observations_cannot_be_filled(bad):
    with pytest.raises(ValueError):
        M.tree_coordinates(bad)


def test_cold_source_direction_and_true_event_ownership():
    record = cold_fixture()
    record["version"] = "HM288-v1"
    source = np.asarray(record["source_components"])
    gamma = source[0, 2, :, 0]
    direction = np.zeros((48, 2, 2, 2, 50))
    for o, t, d in np.ndindex(2, 2, 2):
        direction[:, o, t, d] = ((2 * o - 1) * (2 * t - 1) * gamma)[:, None]
    record["item_logits"] = M.tree_log_probabilities(
        direction, np.zeros_like(direction)
    ).tolist()
    result = M.cold(record)
    value = np.asarray(result["channels"]["direction"]["global_common"])
    assert value[0, 2, 3, 0] == pytest.approx(np.mean(gamma**2))
    available = np.asarray(result["channels"]["assessability"]["global_common"])
    assert np.max(abs(available)) < 1e-12
    record["cases"][2]["writer_comments"][0] = record["cases"][0]["writer_comments"][0]
    with pytest.raises(ValueError, match="ownership"):
        M.cold(record)


def test_cold_rejects_application_labels():
    record = cold_fixture()
    record["version"] = "HM288-v1"
    record["application_labels"] = [1] * 24
    with pytest.raises(ValueError, match="registered"):
        M.cold(record)


def test_native_companion_rejects_binary_before_model_use():
    with pytest.raises(ValueError, match="three-category"):
        M.native_items(None, [], [0, 1], "cpu")


def test_question_preserves_complete_item_and_writer():
    text = M.question("Worry about things.", 1)
    assert '"I worry about things."' in text
    assert "writer R" in text
    assert "2 if they are insufficient" in text
