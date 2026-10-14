#!/usr/bin/env python3
"""Positive controls and failure worlds for cross-target mechanism diagnostics."""

import itertools

import numpy as np
import pytest

from grain_research.construction import math_cross_target_operator as M
from scripts.verify_grain_math_cross_target_operator import independently_construct


def synthetic_rows():
    """Make a complete factorial world with no model calls or human labels."""
    rows = []
    docs = [
        ("atom", "single_event", f, p, -1, c)
        for f, p, c in itertools.product((0, 1), (0, 1), M.TARGETS)
    ]
    docs += [
        ("account", r, f, p, o, c)
        for r, f, p, o, c in itertools.product(
            M.REGIMES, (0, 1), (0, 1), (0, 1), M.TARGETS
        )
    ]
    for parent, (role, regime, f, p, o, c) in enumerate(docs):
        for q in M.TARGETS:
            slope = (2 if c == q else 0.1) * (1 + 0.1 * f) + (
                0.2 * o if role == "account" else 0
            )
            logits = np.linspace(0, (2 * p - 1) * slope, 10).tolist()
            rows.append(
                {
                    "unit": len(rows),
                    "parent_unit": parent,
                    "role": role,
                    "regime": regime,
                    "family": f,
                    "polarity": p,
                    "order": o,
                    "content": c,
                    "query": q,
                    "candidate_logits": logits,
                    "candidate_mass": 0.8,
                    **M.decode(logits),
                }
            )
    return rows


def test_additive_node_world_has_no_cycle_or_selective_matching():
    """Additive target and behavior biases cannot establish target-specific interaction."""
    d = np.arange(6)[:, None] + np.arange(6)[None, :] * 2
    g, c = M.graph_split(d)
    np.testing.assert_allclose(d, g)
    np.testing.assert_allclose(c, 0, atol=1e-14)
    _, _, assignment = M.assignment(d)
    assert assignment["tied_assignments"] == 720
    assert max(abs(r["contrast"]) for r in M.rectangles(d)) == 0


def test_rank_one_can_have_cycles_without_multiple_response_modes():
    """Cycle energy alone must not be mistaken for a rich multi-target operator."""
    d = np.outer(np.arange(1, 7), np.arange(6, 0, -1))
    assert np.linalg.norm(M.graph_split(d)[1]) > 0
    _, s = M.spectrum(d)
    assert s["rank1_tail_share"] < 1e-28
    assert s["numerical_rank"] == 1


def test_selective_identity_accepts_true_multi_target_world():
    """The procedure must find a genuinely selective world, not reject everything."""
    d = np.eye(6)
    _, s = M.spectrum(d)
    _, _, a = M.assignment(d)
    assert s["numerical_rank"] == 6
    assert s["rank1_energy_share"] == pytest.approx(1 / 6)
    assert a["named_rank"] == 1 and a["tied_assignments"] == 1
    assert M.describe(d, d)["cycle_energy_share"] == pytest.approx(5 / 6)


def test_constant_and_zero_do_not_get_false_selectivity():
    """No-response and common-response controls retain undefined or tied diagnostics."""
    for d in (np.zeros((6, 6)), np.full((6, 6), 0.3)):
        assert M.assignment(d)[2]["tied_assignments"] == 720
        assert M.describe(d, d)["named_diagonal_excess"] == pytest.approx(0)
    zero = M.describe(np.zeros((6, 6)), np.zeros((6, 6)))
    assert zero["rank1_energy_share"] is None
    assert zero["shape_remainder_share"] is None
    assert M.cosine(np.zeros((6, 6)), np.eye(6)) is None


def test_roundoff_operator_is_not_perfect_recovery():
    """Balanced contrast cancellation cannot produce a meaningful cosine or factor rank."""
    d = np.eye(6) * 1e-15
    assert M.spectrum(d)[1]["numerical_rank"] == 0
    assert M.cosine(d, d) is None
    assert M.describe(d, np.eye(6))["cycle_energy_share"] is None


def test_graph_split_is_independent_incidence_projection():
    """Orthogonal graph projection and double-centering give the same finite object."""
    d = np.arange(36).reshape(6, 6) ** 2 / 20 + np.eye(6)
    g, c = M.graph_split(d)
    b = M.incidence()
    independent = (b @ np.linalg.lstsq(b, d.ravel(), rcond=None)[0]).reshape(6, 6)
    np.testing.assert_allclose(g, independent, atol=1e-12)
    np.testing.assert_allclose(b.T @ c.ravel(), 0, atol=1e-12)
    assert abs(np.sum(g * c)) < 1e-10
    assert np.linalg.matrix_rank(b) == 11


def test_simultaneous_relabeling_preserves_geometry_and_named_assignment():
    """Coordinate naming does not create or remove an interaction mechanism."""
    d = np.eye(6) + np.arange(6)[:, None] * 0.1
    p = [2, 4, 0, 5, 1, 3]
    renamed = d[np.ix_(p, p)]
    assert M.assignment(d)[2]["named_rank"] == M.assignment(renamed)[2]["named_rank"]
    np.testing.assert_allclose(M.spectrum(d)[0], M.spectrum(renamed)[0], atol=1e-14)
    np.testing.assert_allclose(
        M.graph_split(d)[1][np.ix_(p, p)], M.graph_split(renamed)[1]
    )


@pytest.mark.parametrize(
    "kind", ("missing", "duplicate", "mismatch", "nonfinite", "wrong_parent")
)
def test_bad_factorial_observations_fail_closed(kind):
    """Incomplete data cannot be silently treated as an observed zero response."""
    rows = synthetic_rows()
    if kind == "missing":
        rows.pop()
    elif kind == "duplicate":
        rows.append(rows[0].copy())
    elif kind == "mismatch":
        rows[0]["score"] += 0.2
    elif kind == "nonfinite":
        rows[0]["candidate_logits"][1] = np.nan
    else:
        rows[1]["parent_unit"] = 999
    with pytest.raises(ValueError):
        M.validate_observations(rows)


@pytest.mark.parametrize("metric", M.UNITS)
def test_full_roster_and_independent_cell_contrasts(metric):
    """All planned operators are computed, including null updates and account orders."""
    rows = synthetic_rows()
    operators = M.matrices(rows, metric)
    alternate = independently_construct(rows, M.TARGETS, M.REGIMES, metric)
    assert len(operators) == 50 and set(operators) == set(alternate)
    for key in operators:
        np.testing.assert_allclose(operators[key], alternate[key], atol=1e-14)
    assert len(M.rectangles(operators["atom/pooled"])) == 225
    assert len(M.family_comparisons(operators)) == 8


def test_shape_projection_distinguishes_gain_and_coupling_change():
    """Scaling one operator is not the same as adding a new selective response mode."""
    a = np.eye(6)
    assert M.projection(2 * a, a)["atomic_scale"] == 2
    assert M.projection(2 * a, a)["shape_remainder_share"] == 0
    changed = 2 * a + np.diag(np.arange(6) - 2.5)
    assert M.projection(changed, a)["shape_remainder_share"] > 0


def test_probability_and_logodds_units_remain_separate():
    """Adding a logit gauge leaves both units invariant, but they are not linear copies."""
    z = np.arange(10, dtype=float) * 0.1
    assert M.decode(z) == pytest.approx(M.decode(z + 100))
    with pytest.raises(ValueError):
        M.decode(np.zeros(9))
    assert M.decode(z)["score"] != M.decode(z)["log_odds"]
