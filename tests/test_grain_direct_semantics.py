#!/usr/bin/env python3
"""Frozen semantic cases, recoding and fail-closed GPU ownership checks."""

import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from grain_research.vector_mechanisms.direct_semantics import panel, prepare, summarize
from grain_research.vector_mechanisms.direct_traits import BIG5, tokenize_prompt
from scripts.qualify_grain_direct_traits_semantics import require_idle_history
from tests.test_grain_direct_traits import Tokenizer


def test_full_panel_and_exact_primary_template():
    rows = panel()
    assert len(rows) == 90 and len({c["case_id"] for c in rows}) == 90
    assert sum(c["polarity"] == -1 for c in rows) == 18
    tok = Tokenizer()
    for c in rows:
        ids, a = prepare(c, tok)
        if c["variant"] == "primary":
            expected, _, _ = tokenize_prompt([c["text"]], c["task"], tok)
            assert ids == expected
        elif c["task"] in BIG5:
            np.testing.assert_allclose(a["values"], 1 - np.arange(10) / 9)
        else:
            assert a["values"] == [0, 1]
    bad = rows[0] | {"text": "a post-hoc replacement"}
    with pytest.raises(ValueError, match="Unregistered"):
        prepare(bad, tok)


def test_sensitivity_is_recoded_and_neutral_is_not_targeted():
    rows = [
        c | {"conditional_score": {-1: 0.93, 0: 0.2, 1: 0.8}[c["polarity"]]}
        for c in panel()
    ]
    contrasts, tasks = summarize(rows)
    assert len(contrasts) == 36 and len(tasks) == 9
    assert all(r["positive"] for r in contrasts)
    assert all(r["mean_absolute_recoded_counter_change"] == 0 for r in tasks)
    assert all(r["primary_no_evidence_score"] == 0.93 for r in tasks)
    with pytest.raises(ValueError, match="grid"):
        summarize(rows[:-1])
    with pytest.raises(ValueError, match="grid"):
        summarize(list(reversed(rows)))
    bad = copy.deepcopy(rows)
    bad[0]["conditional_score"] = float("nan")
    with pytest.raises(ValueError, match="readout"):
        summarize(bad)
    for r in rows:
        if r["polarity"] >= 0:
            r["conditional_score"] = 0.6
    assert not any(r["positive"] for r in summarize(rows)[0])


def test_history_idle_gate_rejects_live_lock_running_and_unknown(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "psutil", SimpleNamespace(pid_exists=lambda pid: pid == 42)
    )
    p = tmp_path / "queue_status.json"
    p.write_text(json.dumps({"status": "RUNNING", "pid": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="idle"):
        require_idle_history(tmp_path)
    p.write_text(json.dumps({"status": "COMPLETE", "pid": 42}), encoding="utf-8")
    with pytest.raises(ValueError, match="alive"):
        require_idle_history(tmp_path)
    p.write_text(
        json.dumps({"status": "PAUSED_BUDGET_BETWEEN_SHARDS", "pid": 1}),
        encoding="utf-8",
    )
    (tmp_path / "queue.lock").touch()
    with pytest.raises(ValueError, match="idle"):
        require_idle_history(tmp_path)
    (tmp_path / "queue.lock").unlink()
    assert require_idle_history(tmp_path)["status"] == "PAUSED_BUDGET_BETWEEN_SHARDS"
    with pytest.raises(ValueError):
        require_idle_history(None)
