#!/usr/bin/env python3
"""Completion-based coverage is not outcome-based sample selection."""

import pytest

from scripts.audit_grain_companion_partial_hm243 import coverage, require_terminal


def test_incomplete_corner_is_not_imputed():
    rows, keys = coverage(7)
    assert len(rows) == 48
    assert keys == [(0, 0)]
    assert rows[1] == {"author": 0, "view": 1, "corners": 3, "complete": False}
    assert sum(x["corners"] for x in rows) == 7


def test_complete_views_keep_order_and_missing_authors():
    rows, keys = coverage(49)
    assert len(keys) == 12
    assert len({u for u, _ in keys}) == 6
    assert rows[12]["corners"] == 1
    assert sum(x["complete"] for x in rows) == 12


def test_completion_bounds_are_not_silently_changed():
    for completed in (-1, 193):
        with pytest.raises(ValueError):
            coverage(completed)
    assert len(coverage(192)[1]) == 48


def test_partial_audit_cannot_read_live_or_prepared_outcomes():
    for state in ("RUNNING", "PREPARED", "COLLECTED"):
        with pytest.raises(RuntimeError):
            require_terminal({"state": state}, 123)


def test_terminal_state_alone_does_not_prove_pid_stopped(monkeypatch):
    monkeypatch.setattr(
        "scripts.audit_grain_companion_partial_hm243.os.kill", lambda *_: None
    )
    with pytest.raises(RuntimeError, match="still present"):
        require_terminal({"state": "BUDGET_STOP"}, 123)


def test_terminal_missing_pid_is_safe_without_restart(monkeypatch):
    def absent(*_):
        raise ProcessLookupError

    monkeypatch.setattr("scripts.audit_grain_companion_partial_hm243.os.kill", absent)
    require_terminal({"state": "BUDGET_STOP"}, 123)
