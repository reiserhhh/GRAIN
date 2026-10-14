#!/usr/bin/env python3
"""Synthetic checks for HM86; no model, personal data or target scores."""

import itertools

import numpy as np
import pytest

from grain_research.construction.math_query_evidence import (
    directions,
    event_means,
    evidence_laws,
    finite_action,
    points,
    reading,
)


def test_pairing_and_conditional_identity():
    rng = np.random.default_rng(91)
    r = reading(
        rng.normal(size=(9, 2, 3)),
        rng.normal(size=(9, 2, 3)),
        rng.normal(size=(4, 3)),
        [[1, 2], [4, 5, 6]],
    )
    np.testing.assert_allclose(
        r["actual"], r["event_uniform"] + r["pairing"], atol=1e-14
    )
    np.testing.assert_allclose(
        r["conditional"],
        r["conditional_uniform"] + r["conditional_pairing"],
        atol=1e-14,
    )
    assert np.all((r["body_mass"] > 0) & (r["body_mass"] < 1))


def test_variable_length_action_identity():
    rng = np.random.default_rng(4)
    rows = [
        reading(
            rng.normal(size=(n, 1, 2)),
            rng.normal(size=(n, 1, 2)),
            rng.normal(size=(2, 2)),
            groups,
        )
        for n, groups in [(7, [[0, 1], [3, 4]]), (9, [[0, 1, 2], [5, 6, 7]])]
    ]
    a = finite_action(*rows)
    np.testing.assert_allclose(
        a["actual"], a["value"] + a["allocation"] + a["pairing"], atol=1e-14
    )


def test_permutation_expectation_is_event_uniform():
    k = np.array([[[1.0]], [[2.0]], [[-1.0]], [[0.0]]])
    v = np.array([[[0.4]], [[-0.2]], [[0.6]], [[0.1]]])
    groups = [[0, 1], [2, 3]]
    base = reading(k, v, np.array([[0.7]]), groups)
    observations = []
    for first, second in itertools.product(
        itertools.permutations(groups[0]), itertools.permutations(groups[1])
    ):
        observations.append(
            reading(k, v[list(first + second)], np.array([[0.7]]), groups)["actual"]
        )
    np.testing.assert_allclose(
        np.mean(observations, axis=0), base["event_uniform"], atol=1e-14
    )


def test_source_is_actor_balanced_and_query_free():
    a = np.zeros((6, 2, 2, 4, 3))
    a[:, 1, :, :, 0] = 1
    a[:, :, 0, :, 1] = 10
    d, norm = directions(a)
    np.testing.assert_allclose(norm, 2)
    np.testing.assert_allclose(d[:, :, 1:], 0)
    np.testing.assert_allclose(np.linalg.norm(d.reshape(6, -1), axis=1), 1)


def test_author_law_and_character_identity():
    p, n = evidence_laws()
    np.testing.assert_allclose(p.sum(), 1)
    np.testing.assert_allclose((p - n) @ points(), 0)
    rng = np.random.default_rng(31)
    f = rng.normal(size=(16, 2, 3))
    char = points()[:, 1] * points()[:, 2]
    np.testing.assert_allclose(
        np.einsum("i,ihd->hd", p - n, f),
        np.einsum("i,ihd->hd", char / 16, f),
        atol=1e-14,
    )


@pytest.mark.parametrize("groups", [[], [[0, 1], [1, 2]], [[9]], [[0.1]], [[1, 1]]])
def test_invalid_event_groups_fail_closed(groups):
    with pytest.raises(ValueError):
        event_means(np.ones((5, 1, 2)), groups, 2)


def test_zero_calibration_does_not_get_rescued():
    with pytest.raises(ValueError):
        directions(np.zeros((6, 2, 2, 4, 3)))
