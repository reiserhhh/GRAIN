#!/usr/bin/env python3
"""The replay adapter must compare results, not mask verification failures."""

import json

import pytest

from scripts.verify_grain_observer_order_hm130_readonly import existing_report_sink


def test_final_report_compared_without_rewriting(tmp_path):
    path = tmp_path / "verification.json"
    path.write_text(json.dumps({"proof": 42}))
    before = path.read_bytes(), path.stat().st_mtime_ns
    sink, calls = existing_report_sink(tmp_path)
    sink(path, {"proof": 42})
    assert calls == [True]
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    with pytest.raises(ValueError):
        sink(path, {"proof": 42})


def test_changed_result_and_other_writes_rejected(tmp_path):
    path = tmp_path / "verification.json"
    path.write_text(json.dumps({"proof": 42}))
    sink, calls = existing_report_sink(tmp_path)
    with pytest.raises(ValueError):
        sink(path, {"proof": 0})
    with pytest.raises(ValueError):
        sink(tmp_path / "new_report.json", {"proof": 42})
    assert not calls
    assert not (tmp_path / "new_report.json").exists()
