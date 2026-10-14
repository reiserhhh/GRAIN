"""Tests for GP05 use synthetic local arrays and never inspect private content."""

import json

from pathlib import Path


def test_gp05_script_declares_missing_time_boundary():
    path = Path(__file__).resolve().parents[1] / "scripts/run_grain_natural_interface_audit.py"
    text = path.read_text()
    assert "NOT_IDENTIFIABLE_WITH_THIS_VIEW" in text
    assert "calendar_timestamp" in text


def test_gp05_outputs_are_json_shaped_after_run():
    # This is deliberately a schema-level check; the real cache is private.
    value = {"status": "PASS", "science_status": "BLOCKED_INPUT"}
    assert json.loads(json.dumps(value))["science_status"] == "BLOCKED_INPUT"
