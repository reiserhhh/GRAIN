#!/usr/bin/env python3
"""Distinguish cardinal nonadditivity from globally cyclic ordinal comparison."""

import json

import numpy as np
import pytest

from grain_research.construction import math_author_comparison_geometry as M
from scripts import run_grain_comparison_observable_contract_hm228 as C


def run_ordinal(tmp_path, monkeypatch, values):
    """Supply finite synthetic graph observations without data or model access."""
    edges = M.graph()
    monkeypatch.setattr(
        C.R,
        "check",
        lambda *args: {
            "config": {"observable_contract": {"ordinal_robust_margin": 1e-4}}
        },
    )
    monkeypatch.setattr(C.R, "prepared", lambda _: {"edges": edges})
    monkeypatch.setattr(
        C.H, "selected", lambda *args: {"odd": np.repeat(values[:, None], 5, axis=1)}
    )
    C.ordinal(tmp_path)
    return json.loads((tmp_path / "ordinal_feasibility.json").read_text())["checks"]


def test_nonlinear_cardinal_curl_still_permits_order(tmp_path, monkeypatch):
    """Cubic differences are cyclic in magnitude but preserve scalar ordering."""
    edges = M.graph()
    difference = M.incidence(edges, 24) @ np.arange(24.0)
    nonlinear = difference**3
    assert np.max(abs(M.cycles(edges, 24) @ nonlinear)) > 0
    checks = run_ordinal(tmp_path, monkeypatch, nonlinear)
    assert all(c["strict_scalar_order_feasible_on_observed_edges"] for c in checks)


@pytest.mark.parametrize("closing", [0.1, 1e-5])
def test_long_cycle_missed_by_triangle_test(tmp_path, monkeypatch, closing):
    """A full ring can be ordinal-cyclic without any local triangle reversal."""
    edges = M.graph()
    lookup = {tuple(edge): i for i, edge in enumerate(edges)}
    d = M.incidence(edges, 24) @ np.arange(24.0)
    d[lookup[(0, 23)]] = closing
    for u in range(0, 24, 3):
        triangle = [
            d[lookup[(u, u + 1)]],
            d[lookup[(u + 1, u + 2)]],
            -d[lookup[(u, u + 2)]],
        ]
        assert not (
            np.all(np.asarray(triangle) > 0) or np.all(np.asarray(triangle) < 0)
        )
    checks = run_ordinal(tmp_path, monkeypatch, d)
    exact, robust = checks[:2]
    assert not exact["strict_scalar_order_feasible_on_observed_edges"]
    assert len(exact["cyclic_components"][0]) == 24
    assert robust["strict_scalar_order_feasible_on_observed_edges"] == (closing < 1e-4)


def test_null_order_does_not_create_personality_evidence(tmp_path, monkeypatch):
    """No strict constraints is compatibility, not detection of author variation."""
    checks = run_ordinal(tmp_path, monkeypatch, np.zeros(32))
    assert all(c["directed_edges"] == 0 for c in checks)
    assert all(c["not_personality_validity"] for c in checks)
