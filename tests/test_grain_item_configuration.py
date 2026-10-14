#!/usr/bin/env python3
"""Fixed item projections,source orientation and conditional contact fixtures."""

from itertools import permutations

import numpy as np
import pytest

from grain_research.construction import math_criterion_components as C
from grain_research.construction import math_item_configuration as M
from grain_research.construction import math_output_observability as O


def bank():
    """Unequal poles avoid accidentally assuming five/five items."""
    return [
        {"item": j, "target": j // 10, "pole": 1 if j % 10 < 6 else -1}
        for j in range(50)
    ]


def test_unequal_pole_key_and_orthogonal_complement():
    response = np.arange(150.0).reshape(3, 50)
    z, v, ids = M.item_frame(response, bank(), 0)
    pieces = M.split_frame(z, v)
    assert z @ v == pytest.approx(response @ O.key_matrix(bank())[0])
    assert pieces[0] == pytest.approx(pieces[1] + pieces[2])
    assert pieces[2] @ v == pytest.approx(np.zeros(3), abs=1e-14)
    assert len(ids) == 10
    k = M.kernels(z, z[::-1], v)
    assert k[0] == pytest.approx(k[1] + k[2])


def test_zero_average_configuration_can_have_oriented_shared_pattern():
    v = np.ones(2) / np.sqrt(2)
    s, direction = np.array([-1.0, 1]), np.array([1.0, -1])
    z = s[:, None] * direction
    k = M.kernels(z, z, v)
    assert O.cross_author_accounts(s, s, k[1])["distinct_authors"] == pytest.approx(0)
    assert O.cross_author_accounts(s, s, k[2])["distinct_authors"] > 0
    patterns = M.signed_patterns(s, M.split_frame(z, v))
    assert patterns[2] == pytest.approx(direction)
    assert M.signed_patterns(-s, M.split_frame(z, v)) == pytest.approx(-patterns)
    assert O.cross_author_accounts(-s, -s, k[2])["distinct_authors"] == pytest.approx(
        O.cross_author_accounts(s, s, k[2])["distinct_authors"]
    )
    assert O.cross_author_accounts(s, s, M.kernels(z, -z, v)[2])["distinct_authors"] < 0


def test_exact_common_assignment_mean_is_not_assumed_zero():
    y = np.array([0.0, 0, 1, 1])
    groups = np.zeros(4, int)
    a = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [2, 1, 0, 0.0]])
    b = a[::-1]
    z0 = np.array([[1, 2], [2, -1], [1, 0.0]])
    z1 = np.array([[0, 1], [2, 3], [-1, 0.0]])
    k = M.kernels(z0, z1, np.ones(2) / np.sqrt(2))[2]
    observed = []
    for ix in permutations(range(4)):
        e = y[list(ix)] - 0.5
        observed.append(
            O.cross_author_accounts(
                O.centered_scores(a @ e), O.centered_scores(b @ e), k
            )["distinct_authors"]
        )
    expected = O.orbit_mean(a, b, C.orbit_covariance(y, groups), k)
    assert expected == pytest.approx(np.mean(observed))
    assert abs(expected) > 1e-4


def record():
    """Provide disjoint finite observations,not simulated psychological outcomes."""
    rng = np.random.default_rng(10)
    return {
        "version": "HM286-v1",
        "author_keys": ["a", "b", "c"],
        "left_event_ids": [[f"l{u}a", f"l{u}b"] for u in range(3)],
        "right_event_ids": [[f"r{u}a", f"r{u}b"] for u in range(3)],
        "source_readings": rng.normal(size=(2, 3, 5)).tolist(),
        "item_joint_responses": rng.normal(size=(2, 3, 50)).tolist(),
        "item_bank": bank(),
    }


def test_cold_signed_profile_preserves_source_orientation():
    data = record()
    original = M.cold(data)
    data["source_readings"] = (-np.asarray(data["source_readings"])).tolist()
    changed = M.cold(data)
    assert np.asarray(changed["signed_source_query_item_patterns"]) == pytest.approx(
        -np.asarray(original["signed_source_query_item_patterns"])
    )
    assert np.asarray(changed["distinct_author_source_query_parts"]) == pytest.approx(
        np.asarray(original["distinct_author_source_query_parts"])
    )


@pytest.mark.parametrize(
    "bad", ["duplicate_event", "label", "missing_item", "nonfinite", "wrong_views"]
)
def test_cold_rejects_unqualified_observations(bad):
    data = record()
    if bad == "duplicate_event":
        data["right_event_ids"][0][0] = data["left_event_ids"][0][0]
    elif bad == "label":
        data["application_trait"] = 1
    elif bad == "missing_item":
        data["item_bank"].pop()
    elif bad == "wrong_views":
        data["left_event_ids"].append(data["right_event_ids"].pop())
    else:
        data["source_readings"][0][0][0] = float("nan")
    with pytest.raises(ValueError):
        M.cold(data)
