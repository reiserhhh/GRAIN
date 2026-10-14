#!/usr/bin/env python3
"""Ensure the D01 inventory exports metadata, not participant values."""

import json

import pytest

import scripts.inventory_grain_interpretation_evidence as module


def test_inventory_keeps_private_values_out(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "scores.csv").write_text("participant,score\nPRIVATE_AUTHOR,4.321\n")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "VM", source)
    monkeypatch.setattr(
        module,
        "INPUTS",
        [
            ("criterion", "scores.csv", "validation only"),
            ("missing", "missing.csv", "pending"),
        ],
    )
    # The manifest records the real script; provide a synthetic root-relative path.
    monkeypatch.setattr(module, "__file__", str(source / "scores.csv"))
    output = tmp_path / "output"
    module.inventory(output)
    rows = json.loads((output / "source_inventory.json").read_text())
    assert rows[0]["rows"] == 1 and rows[0]["columns"] == ["participant", "score"]
    assert rows[1]["exists"] is False
    for path in output.glob("*.json"):
        assert "PRIVATE_AUTHOR" not in path.read_text()
        assert "4.321" not in path.read_text()
    with pytest.raises(FileExistsError):
        module.inventory(output)
