#!/usr/bin/env python3
"""F00-01 contract and replay tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grain_research.gptpro_foundation_v3.f00 import build_manifest, function_inventory

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent / "project persona"


def test_declared_source_manifest_is_complete():
    manifest = build_manifest(PARENT, ROOT)
    assert len(manifest["source_documents"]) == 5
    assert all(row["exists"] for row in manifest["inputs"])


def test_inventory_is_static_and_documented():
    rows = function_inventory(ROOT / "grain_research/measurement_discovery")
    assert rows
    assert all("function" in row and "path" in row for row in rows)


def test_b1_reference_values_are_not_silently_changed():
    summary = json.loads((ROOT / "results/grain_measurement_discovery_v1/gp03/r1/summary.json").read_text())
    oracle = {row["split"]: row["brier"] for row in summary["metrics"] if row["channel"] == "oracle"}
    assert oracle == {"validation": pytest.approx(0.19343257893885188), "application": pytest.approx(0.18910631495521518)}
