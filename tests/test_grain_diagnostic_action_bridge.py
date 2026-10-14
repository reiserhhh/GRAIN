#!/usr/bin/env python3
"""Synthetic implementation checks, not personality-validity evidence."""

from copy import deepcopy

import numpy as np
import pytest

from grain_research.construction import math_diagnostic_action_bridge as M
from grain_research.construction import math_relational_diagnostic_update as R


@pytest.fixture
def case():
    """All32 supported profiles and two addressed observation windows."""
    rng = np.random.default_rng(214)
    bank = R.compile_bank(
        rng.normal(size=(64, 2, 3)) / 5, np.repeat(np.arange(32), 2), 1
    )
    record = {
        "frame_key": "synthetic",
        "windows": [
            [
                {
                    "event_id": f"{v}:{t}",
                    "vector": rng.normal(size=3).tolist(),
                    "condition": t % 2,
                }
                for t in range(12)
            ]
            for v in (0, 1)
        ],
        "edit": {
            "view": 0,
            "selected": [f"0:{t}" for t in range(4)],
            "control": [f"0:{t}" for t in range(4, 8)],
        },
    }
    return bank, record


def test_literal_all_laws(case):
    """Direct distance/product enumeration agrees in the registered units."""
    bank, record = case
    points, view = M.edit_points(record, 3)
    np.testing.assert_allclose(
        M.readings(bank, points, view), M.literal(bank, points, view), atol=2e-14
    )


def test_hm209_pairing_replayed(case):
    """Paired and product readings have exactly the established source meaning."""
    bank, record = case
    points, view = M.edit_points(record, 3)
    np.testing.assert_allclose(
        M.readings(bank, points, view)[:, 1:],
        R.read(bank, points)["log_odds"],
        atol=2e-14,
    )


def test_companion_single_law_invariance(case):
    """A single-window scorer does not silently inspect a companion window."""
    bank, record = case
    points, view = M.edit_points(record, 3)
    changed = points.copy()
    changed[:, 1] += 3
    np.testing.assert_array_equal(
        M.readings(bank, points, view)[:, 0], M.readings(bank, changed, view)[:, 0]
    )


def test_analytic_single_gradient(case):
    """Finite perturbations check the fixed analytic field, without tuning it."""
    bank, record = case
    points, view = M.edit_points(record, 3)
    step = np.array([0.2, -0.1, 0.4])
    high, low = points[:1].copy(), points[:1].copy()
    high[:, view] += 1e-5 * step
    low[:, view] -= 1e-5 * step
    finite = (
        M.readings(bank, high, view)[0, 0] - M.readings(bank, low, view)[0, 0]
    ) / 2e-5
    np.testing.assert_allclose(
        finite, M.single_field(bank, points[:1, view], view)[0] @ step, atol=1e-10
    )


def test_reverse_edit_sign(case):
    """Finite actions reverse sign without modifying source coefficients."""
    bank, record = case
    other = deepcopy(record)
    other["edit"]["selected"], other["edit"]["control"] = (
        other["edit"]["control"],
        other["edit"]["selected"],
    )
    np.testing.assert_allclose(
        M.cold(bank, other)["finite_actions"],
        -np.array(M.cold(bank, record)["finite_actions"]),
        atol=0,
    )


@pytest.mark.parametrize("bad", ["labels", "duplicate", "foreign", "overlap", "empty"])
def test_invalid_event_protocol(case, bad):
    """Private labels, fake addresses and duplicate evidence cannot enter cold input."""
    bank, record = case
    row = deepcopy(record)
    if bad == "labels":
        row["labels"] = [1] * 5
    elif bad == "duplicate":
        row["windows"][1][0]["event_id"] = "0:0"
    elif bad == "foreign":
        row["edit"]["selected"][0] = "unseen"
    elif bad == "overlap":
        row["edit"]["control"][0] = "0:0"
    else:
        row["windows"][1] = []
    with pytest.raises(ValueError):
        M.cold(bank, row)
