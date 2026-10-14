#!/usr/bin/env python3
"""Exact worlds distinguish Markov laws,heterogeneity and target assignment."""

import numpy as np
import pytest

from grain_research.construction import math_observed_path_memory as M


def product(a, b):
    """Return a two-state A,B,C product with an equiprobable middle state."""
    return np.einsum("i,j,k->ijk", [a, 1 - a], [0.5, 0.5], [b, 1 - b])


def test_markov_and_generic_memory_are_not_target_assignment():
    y = np.array([[0] * 5, [1] * 5])
    for p in (
        product(0.4, 0.7),
        np.array([[[0.2, 0.05], [0.2, 0.05]], [[0.05, 0.2], [0.05, 0.2]]]),
    ):
        profiles, laws, _ = M.construct(np.stack([p, p]), y)
        np.testing.assert_allclose(laws[2], laws[3], atol=1e-10)
        np.testing.assert_allclose(
            M.fields(profiles, laws)[3] - M.fields(profiles, laws)[2], 0, atol=1e-10
        )
    assert M.divergence(laws[3], laws[0]) > 0


def test_mixture_creates_apparent_memory_without_individual_memory():
    p = np.stack([product(0.1, 0.1), product(0.9, 0.9)] * 2)
    y = np.array([[0] * 5, [0] * 5, [1] * 5, [1] * 5])
    _, laws, _ = M.construct(p, y)
    assert M.divergence(laws[3], laws[0]) > 0.1
    np.testing.assert_allclose(laws[1], laws[3], atol=1e-12)


def test_target_parity_survives_adjacent_and_pooled_constraints():
    p = np.zeros((2, 2, 2, 2))
    for g in range(2):
        for a in range(2):
            for b in range(2):
                p[g, a, b, a ^ g] = 0.25
    profiles, laws, _ = M.construct(p, np.array([[0] * 5, [1] * 5]))
    assert max(M.margin_errors(laws).values()) < 1e-10
    field = M.fields(profiles, laws)
    assert np.max(abs(field[3] - field[2])) == pytest.approx(0.5)


def test_triples_are_not_independent_records_and_bags_do_not_fix_paths():
    x, y = [0, 0, 1, 1, 0], [0, 1, 0, 1, 0]
    np.testing.assert_array_equal(np.bincount(x), np.bincount(y))
    assert not np.allclose(M.triple_mass(x, 2), M.triple_mass(y, 2))
    assert M.triple_mass(x, 2).sum() == pytest.approx(1)


def test_cold_identity_chronology_and_no_outcomes():
    record = {
        "author_key": "anonymous",
        "events": [
            {"event_id": str(i), "cell": i % 2, "timestamp": float(i)} for i in range(4)
        ],
    }
    readout = np.full((2, 4, 5, 2, 2, 2), 0.5)
    assert M.cold_record(readout, record)["overlapping_triples"] == 2
    with pytest.raises(ValueError):
        M.cold_record(readout, record | {"label": 1})
    repeated = {"author_key": "x", "events": record["events"] + [record["events"][0]]}
    with pytest.raises(ValueError):
        M.cold_record(readout, repeated)
    reversed_record = {"author_key": "x", "events": record["events"][::-1]}
    with pytest.raises(ValueError):
        M.cold_record(readout, reversed_record)


def test_missing_source_cell_is_not_neutral():
    readout = np.full((2, 4, 5, 2, 2, 2), 0.5)
    readout[..., 0, 1, 0] = np.nan
    record = {
        "author_key": "x",
        "events": [
            {"event_id": str(i), "cell": i % 2, "timestamp": i} for i in range(3)
        ],
    }
    value = M.cold_record(readout, record)
    assert not np.asarray(value["supported"]).any()
    assert value["scores"][0][0][0] is None


def test_author_and_event_rename_do_not_change_score():
    r = {
        "author_key": "old",
        "events": [
            {"event_id": str(i), "cell": i % 2, "timestamp": i} for i in range(3)
        ],
    }
    s = {
        "author_key": "new",
        "events": [v | {"event_id": "new" + v["event_id"]} for v in r["events"]],
    }
    field = np.full((2, 4, 5, 2, 2, 2), 0.25)
    assert M.cold_record(field, r)["scores"] == M.cold_record(field, s)["scores"]


def test_no_common_support_cannot_be_reported_as_zero_relation():
    field = np.full((5, 2, 2, 2), np.nan)
    inner, cosine, coverage, count = M.recurrence(
        field, field, np.full((2, 2, 2), 0.125)
    )
    assert np.isnan(inner).all() and np.isnan(cosine).all()
    assert coverage == 0 and count == 0
