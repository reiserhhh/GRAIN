"""R34 target-structure inventory: structure, OPEN status, dry-run labelling, builder signatures exist."""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
D = ROOT / "results/grain_gptpro_foundation_v3/F16/target_structure_inventory_r1"


def test_inventory_structure_and_open_status():
    inv = json.loads((D / "inventory.json").read_text())
    assert set(inv["targets"]) == {"T1", "T2", "T3"} and "OPEN" in inv["status"]
    for t in inv["targets"].values():
        for key in ("estimand", "object_class", "structural_object", "gauge_reference", "target_appropriate_comparisons", "missing"):
            assert t.get(key), key
    assert any(".80" in r for r in inv["rules"]) and any("not structural re-estimation" in r.lower() for r in inv["rules"])


def test_dry_run_labelled_and_no_metrics():
    dr = json.loads((D / "dry_run_feasibility.json").read_text())
    assert dr["label"].startswith("DRY RUN FEASIBILITY ONLY") and dr["neural_calls"] == 0
    assert not any(k in json.dumps(dr) for k in ("agreement", "distance", "ratio"))


def test_builder_functions_are_subset_capable():
    from grain_research.gptpro_foundation_v3 import f05_recovery, f16_t1_source, f16_t1_text
    assert "selected" in inspect.signature(f16_t1_source.fit_source).parameters
    assert list(inspect.signature(f05_recovery.fit_source_dynamics).parameters) == ["rows"]
    assert "rows" in inspect.signature(f16_t1_text.channel_counts).parameters
