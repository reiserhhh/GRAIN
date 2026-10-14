#!/usr/bin/env python3
"""Finite event-field identities and contrasting mechanism worlds."""

import numpy as np
import pytest

from grain_research.construction import math_event_field as M


def test_decomposition_is_lossless_and_energy_preserving():
    a = np.random.default_rng(1).normal(size=(2, 8, 3, 9))
    d = M.decompose(a)
    np.testing.assert_allclose(d["even"] + d["odd"], a[0], atol=1e-14)
    np.testing.assert_allclose(d["even"] - d["odd"], a[1], atol=1e-14)
    np.testing.assert_allclose(
        (a * a).sum() / 2, (d["even"] ** 2).sum() + (d["odd"] ** 2).sum()
    )
    swapped = M.decompose(a[::-1])
    np.testing.assert_allclose(swapped["even"], d["even"])
    np.testing.assert_allclose(swapped["odd"], -d["odd"])


def test_event_field_preserves_identity_where_pool_forgets_it():
    field = np.arange(24).reshape(8, 3)
    np.testing.assert_allclose(field.mean(1), np.roll(field, 1, axis=1).mean(1))
    assert not np.array_equal(field, np.roll(field, 1, axis=1))


def test_terminal_only_and_event_relation_worlds_are_distinguished():
    events = np.eye(3)
    assert not np.array_equal(events[0], events[-1])
    orders = [events, events[::-1]]
    endpoints = np.stack([v[-1] for v in orders])
    aligned = np.stack([v if i == 0 else v[::-1] for i, v in enumerate(orders)])
    assert np.linalg.norm(M.decompose(endpoints)["odd"]) > 0
    np.testing.assert_array_equal(M.decompose(aligned)["odd"], 0)


def test_profile_observation_commutes_with_order_decomposition():
    p = np.random.default_rng(4).dirichlet(np.ones(8), size=2)
    h = np.random.default_rng(2).normal(size=(2, 8, 3, 7))
    a = M.decompose(h)
    d = M.decompose(np.stack([M.observe(p, v) for v in h]))
    for name in M.MODES:
        np.testing.assert_allclose(M.observe(p, a[name]), d[name], atol=1e-14)


def test_all_boolean_characters_are_orthogonal_and_complete():
    w = np.vstack((np.ones(8), M.characters()))
    np.testing.assert_array_equal(w @ w.T, 8 * np.eye(8))


def test_actual_span_alignment_and_actor_retention():
    body = "\n".join(f"On occasion {e}, Person A did work." for e in M.EVENTS[::-1])
    spans = M.positions(body, body, [(i, i + 1) for i in range(len(body))])
    h = np.arange(len(body) * 2).reshape(len(body), 2)
    result = M.extract(h, spans)
    assert set(spans) == set(M.EVENTS)
    assert all(s["actor"] == "A" for s in spans.values())
    np.testing.assert_array_equal(result["event_end"][0], result["body_end"])
    np.testing.assert_allclose(
        result["pooled_event_mean"], result["event_mean"].mean(0)
    )


def test_zero_contrast_is_not_a_fake_perfect_direction():
    row = M.relation_row(np.zeros((3, 5)), np.ones((3, 5)))
    assert row["direction_cosine"] is None
    assert row["wrong_slot_cosine"] is None


@pytest.mark.parametrize("bad", [np.zeros((1, 8, 3)), np.full((2, 8, 3), np.nan)])
def test_incomplete_nonfinite_orders_rejected(bad):
    with pytest.raises(ValueError):
        M.decompose(bad)


def test_invalid_probabilities_rejected():
    with pytest.raises(ValueError):
        M.observe(np.ones((2, 8)), np.ones((8, 3, 5)))


def test_missing_event_address_rejected():
    with pytest.raises(ValueError):
        M.positions(
            "On occasion K1, Person A acted.", "On occasion K1, Person A acted.", []
        )
