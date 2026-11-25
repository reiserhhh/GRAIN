#!/usr/bin/env python3
"""Run the examples as a reader would, using the canonical public namespace."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(module, example):
    raw = subprocess.check_output(
        [sys.executable, "-m", f"grain_core.{module}", f"examples/{example}.json"],
        cwd=ROOT,
        text=True,
    )
    return json.loads(raw)


def test_event_quickstart():
    result = run("event_response", "event_response")
    assert result["events"][0]["status"] == "support"
    assert result["cards"][0]["reference_rate_bounds"] == [0.75, 1.0]
    assert not result["later_checks"]


def test_view_quickstart():
    result = run("dual_view_increment", "dual_view_increment")
    assert result["simple_lag"] == pytest.approx([-0.4, 2.0])
    assert result["full"] == pytest.approx([0.8, 1.2])
    assert result["vs_simple_lag"] == "changes_threshold_decision"


@pytest.mark.parametrize(
    ("index", "suffix"),
    [("README.md", ""), ("README.ja.md", "_JA"), ("README.zh.md", "_ZH")],
)
def test_localized_entrypoints(index, suffix):
    manual = f"docs/GRAIN_EVENT_AND_VIEW_INCREMENT{suffix}.md"
    assert manual in (ROOT / index).read_text()
    content = (ROOT / manual).read_text()
    for module in ("event_response", "dual_view_increment"):
        assert f"python -m grain_core.{module} examples/{module}.json" in content
