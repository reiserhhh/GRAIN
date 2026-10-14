#!/usr/bin/env python3
"""Finite source-law and SAME-event bridge sanity checks, not psychometric proof."""

import numpy as np
import pytest

from grain_research.construction import math_source_meaning_bridge as M


def law():
    """Make a centered, distinct, bounded five-field test law."""
    r = np.full(8, 1 / 8)
    x = np.arange(8)
    w = (
        np.column_stack(
            [
                np.prod(
                    [
                        np.where(x & (1 << b), 1, -1)
                        for b in range(3)
                        if mask & (1 << b)
                    ],
                    axis=0,
                )
                for mask in (1, 2, 3, 4, 5)
            ]
        )
        * 0.1
    )
    return r, 0.5 + w, np.full(5, 0.5)


def test_direct_event_mean_and_cold_batch_invariance():
    """Source anchoring agrees with event sums and does not depend on batch peers."""
    r, psi, pi = law()
    f = M.fields(r, psi, pi)
    p = M.event_mass(np.array([0, 0, 2, 7]), ["a", "b", "c", "d"])
    score = M.cold_scores(p[None], r, f)["full"]
    expected = (psi - pi)[[0, 0, 2, 7]].mean(0)
    np.testing.assert_allclose(score[0], expected, atol=1e-15)
    np.testing.assert_array_equal(
        score[0], M.cold_scores(np.stack((p, r)), r, f)["full"][0]
    )


def test_shared_unique_and_choice_within_preserved():
    """Both source-only decompositions retain the entire original field."""
    r, psi, pi = law()
    f = M.fields(r, psi, pi)
    np.testing.assert_allclose(f["full"], f["choice"] + f["within"], atol=1e-15)
    np.testing.assert_allclose(f["full"], f["unique"] + f["shared"], atol=1e-15)
    product = f["full"].T @ (r[:, None] * f["unique"])
    np.testing.assert_allclose(product - np.diag(np.diag(product)), 0, atol=1e-15)
    assert M.gram(f["full"], r)["rank"] == 5


def test_aliased_fields_are_not_five_independent_meanings():
    """A fully coincident source family has no target-unique field."""
    r, psi, pi = law()
    psi[:] = psi[:, :1]
    f = M.fields(r, psi, pi)
    assert np.max(abs(f["unique"])) < 1e-15
    assert M.gram(f["full"], r)["rank"] == 1


def test_matched_conditions_zero_choice_not_whole_evidence():
    """Condition-matched edits retain within-condition target differences."""
    r, psi, pi = law()
    f = M.fields(r, psi, pi)
    cells = np.array([0, 1, 4, 5, 2, 3, 6, 7, 0, 1, 4, 5])
    keep_s = np.array([i for i in range(12) if i not in (0, 2, 4, 6)])
    keep_c = np.array([i for i in range(12) if i not in (1, 3, 5, 7)])
    p = np.stack(
        [M.event_mass(cells[ix], [str(i) for i in ix]) for ix in (keep_s, keep_c)]
    )
    scores = M.cold_scores(p, r, f)
    np.testing.assert_allclose(scores["choice"][0] - scores["choice"][1], 0, atol=1e-15)
    assert np.linalg.norm(scores["within"][0] - scores["within"][1]) > 0


def test_crossed_tensor_orientation_and_negative_contacts():
    """Meaning,reader andsource axes cannot be silently interchanged."""
    x = np.zeros((3, 5))
    t = np.zeros((3, 5, 5))
    x[:, 4] = [1, 2, -1]
    t[:, 1, 3] = [-1, -2, 1]
    b = M.bridge(x, t)
    assert b["contact"][4, 1, 3] == -2
    assert b["cosine"][4, 1, 3] == pytest.approx(-1)
    assert not b["defined"][0, 0, 0]
    assert np.count_nonzero(b["contact"]) == 1


@pytest.mark.parametrize(
    "cells,ids",
    [
        (np.array([0, 1]), ["same", "same"]),
        (np.array([0, 8]), ["a", "b"]),
        (np.array([], dtype=int), []),
    ],
)
def test_bad_observation_rejected(cells, ids):
    """Missing/duplicated/unsupported events are not filled as neutral scores."""
    with pytest.raises(ValueError):
        M.event_mass(cells, ids)


def test_missing_source_cell_rejected():
    """No unavailable source region is silently assigned zero meaning."""
    r, psi, pi = law()
    r[0] = 0
    r[1] += 1 / 8
    with pytest.raises(ValueError):
        M.fields(r, psi, pi)
