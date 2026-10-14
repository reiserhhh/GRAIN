#!/usr/bin/env python3
"""Counterworld, qualification and independent marginal-polytope checks."""

import itertools

import numpy as np
import pytest

from grain_research.construction import math_event_graph_observation as M


def toy():
    """A nonconstant relation-sensitive kernel with known third coefficient."""
    triple = [0, 1, 2]
    kernel = (
        0.2
        + 0.1 * M.SIGNS[:, 0]
        + 0.3 * M.SIGNS[:, 0] * M.SIGNS[:, 1]
        + 0.7 * M.SIGNS.prod(1)
    )
    lookup = {}
    for bank, (i, signs) in itertools.product(range(2), enumerate(M.SIGNS)):
        levels = tuple((signs + 1).tolist() + [1, 1, 1])
        lookup[bank, levels] = np.array([kernel[i] + 0.6 * bank])
    return triple, kernel, lookup


@pytest.mark.parametrize(
    "kind", ("episode_order", "reverse", "slot_shuffle", "packet_13", "duplicates")
)
def test_same_graph_different_deliveries(kind):
    """Packaging does not become an author difference, even with exact aliases."""
    triple, _, lookup = toy()
    rows = M.design_graph(0, 0, triple, 1)
    delivered = M.package(rows, kind, 113)
    assert M.digest(rows) == M.digest(delivered)
    assert len(M.observation(delivered, lookup)["episodes"]) == 40
    np.testing.assert_allclose(
        M.observation(delivered, lookup)["score"], [0.48], atol=1e-14
    )


def test_atomic_equality_but_real_relation_difference():
    """Exact same atoms and all pair moments permit different triple relations."""
    triple, _, lookup = toy()
    low, high = [M.design_graph(0, 0, triple, law) for law in (-1, 1)]
    assert M.digest(low, True) == M.digest(high, True)
    assert M.digest(low) != M.digest(high)
    np.testing.assert_allclose(
        M.observation(high, lookup)["score"] - M.observation(low, lookup)["score"],
        [0.56],
    )


def test_invalid_coupling_depends_on_delivery():
    """Encounter-order zipping manufactures a different joint distribution."""
    triple, _, lookup = toy()
    rows = M.design_graph(0, 0, triple, 1)
    original = M.order_zip_observation(rows, lookup)
    shuffled = M.order_zip_observation(M.package(rows, "slot_shuffle", 113), lookup)
    assert abs(float(original[0] - shuffled[0])) > 0.01


def test_unknown_links_and_actors_are_not_zero():
    """No event-link or subject inference is smuggled into direct joint scores."""
    triple, _, lookup = toy()
    rows = M.design_graph(0, 0, triple, 1)
    for field in ("episode_id", "actor", "condition", "state_id"):
        result = M.observation([{**r, field: None} for r in rows], lookup)
        assert result["score"] is None
        assert result["unresolved"]


def test_conflicting_alias_and_episode_are_errors():
    """Different values or conditions are not silently averaged as duplicates."""
    triple, _, lookup = toy()
    rows = M.design_graph(0, 0, triple, 1)
    with pytest.raises(ValueError, match="Conflicting reports"):
        M.observation(rows + [{**rows[0], "level": 2}], lookup)
    rows[0] = {**rows[0], "condition": 1}
    with pytest.raises(ValueError, match="conflicting condition"):
        M.observation(rows, lookup)


def test_other_actor_does_not_change_qualified_score():
    """Other-actor episodes stay excluded rather than become author evidence."""
    triple, _, lookup = toy()
    rows = M.design_graph(0, 0, triple, 1)
    other = [
        {
            **r,
            "atom_id": "B:" + r["atom_id"],
            "episode_id": "B:e0",
            "actor": "B",
            "qualification": "other_report",
        }
        for r in rows[:6]
    ]
    result = M.observation(rows + other, lookup)
    assert len(result["excluded"]) == 6
    np.testing.assert_allclose(result["score"], [0.48])


def test_choice_and_new_episodes_are_preserved():
    """Independent repeated episodes can change choice; report duplicates cannot."""
    triple, _, lookup = toy()
    rows = M.design_graph(0, 0, triple, 1)
    extra = [
        {
            **r,
            "atom_id": "new:" + r["atom_id"],
            "episode_id": "new:" + r["episode_id"],
            "condition": 1,
        }
        for r in rows
    ]
    result = M.observation(rows + extra, lookup)
    assert result["choice"] == {0: 0.5, 1: 0.5}
    np.testing.assert_allclose(result["score"], [0.78])


def test_pair_bounds_and_complementary_relation():
    """LP agrees with the explicit one-dimensional coupling family."""
    _, kernel, _ = toy()
    pair = M.identified_bounds(kernel, 2)
    np.testing.assert_allclose([pair["lower"], pair["upper"]], [-0.5, 0.9])
    assert pair["rank"] == 7
    for law in (-1, 1):
        full = M.identified_bounds(kernel, 3, third=0.4 * law)
        assert full["rank"] == 8
        np.testing.assert_allclose(
            [full["lower"], full["upper"]], [0.2 + 0.28 * law] * 2
        )
    null = M.identified_bounds(kernel - 0.7 * M.SIGNS.prod(1), 2)
    np.testing.assert_allclose([null["lower"], null["upper"]], [0.2, 0.2])


def test_certificates_have_attainable_primal_and_dual():
    """Witnesses prove sharpness, not just a numerical optimizer success flag."""
    _, kernel, _ = toy()
    result = M.identified_bounds(kernel, 1)
    a, b = np.array(result["a"]), np.array(result["b"])
    for direction, w in zip((1, -1), result["witnesses"], strict=True):
        p, dual = np.array(w["p"]), np.array(w["dual"])
        assert np.min(p) >= -1e-12
        np.testing.assert_allclose(a @ p, b, atol=1e-12)
        assert np.min(direction * kernel - a.T @ dual) >= -1e-12
        np.testing.assert_allclose(direction * kernel @ p, b @ dual, atol=1e-12)


def test_out_of_support_is_not_fabricated():
    """A partial or undeclared unit cannot receive a convenient fake zero."""
    triple, _, lookup = toy()
    rows = M.design_graph(0, 0, triple, 1)
    result = M.observation(rows[:5], lookup)
    assert result["score"] is None
    assert len(result["unresolved"]) == 5
    with pytest.raises(ValueError, match="outside"):
        M.observation(rows, {})
