#!/usr/bin/env python3
"""F02-04 adverse inputs: single state, reversed order, same mean/different history."""

import numpy as np

from grain_research.gptpro_foundation_v3.f02 import generate_world
from grain_research.gptpro_foundation_v3.f02_recovery import (
    fit_observed_source,
    sequential_predictions,
)
from grain_research.measurement_discovery.estimators import (
    author_table,
    fit_b1,
    predict_one,
)

COUNTS = {"source": 30, "validation": 0, "application": 0}


def rows_of(values, author="u_a", role="source", zs=None):
    """Minimal ordered records for one author."""
    return [
        {
            "author_id": author,
            "event_id": f"{author}_{t}",
            "t": t,
            "observation": int(v),
            "role": role,
            "condition": {"z": (zs[t] if zs else 1)},
        }
        for t, v in enumerate(values)
    ]


def c2_fit():
    records, _ = generate_world("c2_xor", 20260915, events=48, counts=COUNTS)
    return fit_observed_source(records, "c2_xor")


def test_single_state_source_fit_is_finite_and_uninformed_states_stay_half():
    fitted = fit_b1({"u_a": rows_of([0] * 20)})
    p = fitted["source_p"]
    assert np.all(np.isfinite(p))
    assert p[0] == 1 / (18 + 2)
    assert np.allclose(p[1:], 0.5)
    table = author_table(rows_of([1] * 12))
    assert table[3] == (10 + 1) / (10 + 2) and np.allclose(table[:3], 0.5)


def test_single_state_prefix_predictions_stay_inside_unit_interval():
    fitted = c2_fit()
    prefix = rows_of([1] * 12, role="application")
    for candidate in ("b1", "b2", "m"):
        p = predict_one(candidate, (1, 1), prefix, fitted)
        assert 1e-6 <= p <= 1 - 1e-6


def test_same_mean_different_history_separates_history_models_from_mean():
    fitted = c2_fit()
    alternating = rows_of([0, 1] * 6, author="u_alt", role="application")
    blocked = rows_of([0] * 6 + [1] * 6, author="u_blk", role="application")
    assert np.mean([r["observation"] for r in alternating]) == np.mean(
        [r["observation"] for r in blocked]
    )
    # the same final state (0,1) is not guaranteed; use a common next state
    state = (0, 1)
    b2 = [predict_one("b2", state, h, fitted) for h in (alternating, blocked)]
    m = [predict_one("m", state, h, fitted) for h in (alternating, blocked)]
    assert abs(b2[0] - b2[1]) > 1e-3
    assert abs(m[0] - m[1]) > 1e-3
    assert predict_one("b1", state, alternating, fitted) == predict_one(
        "b1", state, blocked, fitted
    )


def test_reversed_order_changes_history_models_not_mean_or_condition_cells():
    values = [0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    forward = rows_of(values, author="u_f", role="application")
    backward = rows_of(values[::-1], author="u_b", role="application")
    fitted = c2_fit()
    state = (1, 1)
    assert (
        abs(
            predict_one("b2", state, forward, fitted)
            - predict_one("b2", state, backward, fitted)
        )
        > 1e-3
    )
    zs = [1, -1] * 6
    c1_source, _ = generate_world("c1", 20260915, events=48, counts=COUNTS)
    c1_fit = fit_observed_source(c1_source, "c1")
    tail = [0, 1]
    fwd = rows_of(values + tail, author="u_f", role="application", zs=zs + [1, -1])
    bwd = rows_of(
        values[::-1] + tail, author="u_b", role="application", zs=zs[::-1] + [1, -1]
    )
    pf = sequential_predictions(fwd, c1_fit, prefix=12)
    pb = sequential_predictions(bwd, c1_fit, prefix=12)
    first = {
        c: next(p["p"] for p in pf if p["t"] == 12 and p["candidate"] == c)
        for c in ("personal_mean", "personal_condition")
    }
    second = {
        c: next(p["p"] for p in pb if p["t"] == 12 and p["candidate"] == c)
        for c in ("personal_mean", "personal_condition")
    }
    assert first == second
