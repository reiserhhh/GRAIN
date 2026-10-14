"""Current F00-01/F00-05 revisions and dated integrity snapshots (packet 13 Part A3/A4)."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3 import revision_stage as rs  # noqa: E402

F00 = ROOT / "results/grain_gptpro_foundation_v3/F00"


def test_baseline_actual_replay_and_corrected_legacy_claim():
    b = json.loads((F00 / "baseline_inventory_r2/baseline_inventory.json").read_text())
    assert b["source_and_plan_hashes"]["all_match"] and b["missing_sources"] == []
    leg = b["legacy_roots"]["gp_pa_pv_files_vs_20260915_snapshot"]
    assert leg["files"] == leg["match"] == 80
    r = b["b1_replay"]
    assert r["status"] == "REPLAYED_EXACT" and r["metrics_csv_byte_identical"] and r["scores_csv_byte_identical"] and r["legacy_gp03_unchanged"]
    assert r["reader_joblib_byte_identical"] is False            # recorded honestly: model bytes differ, predictions identical
    old = json.loads((F00 / "r1/F00-01/summary.json").read_text())["replay"]
    assert old["recomputed_prediction_file"] is False and "run into a temporary directory" in old["note"]
    assert (F00 / "baseline_inventory_r2/b1_replay/text_channel_scores.csv").exists()


def test_routing_covers_every_open_gap():
    acc = json.loads((F00 / "repair_routing_r2/acceptance.json").read_text())
    assert acc["every_open_gap_has_responsible_card"] and acc["gaps_without_card"] == [] and acc["no_old_long_runs_restarted"]
    rows = list(csv.DictReader((F00 / "repair_routing_r2/repair_routing.csv").open()))
    assert len(rows) == acc["rows"] == 64
    assert (F00 / "repair_routing_r2_failed_attempt1/FAILURE.md").exists()


def test_snapshots_are_labelled_current_only_and_skip_manifested_or_private_roots():
    idx = json.loads((F00 / "integrity_snapshots_r1/index.json").read_text())
    assert len(idx["snapshots"]) == 120
    assert {"F13/binary_source_full_r2", "F13/recovery3_r1", "F00/stage_orchestrator_r1", "F00/common_record_adapter_r2"} <= {s["root"] for s in idx["snapshots"]}
    assert any(s["root"] == "R17/natural_sensitivity_r1" and s["reason"].startswith("private") for s in idx["skipped"])
    for s in idx["snapshots"]:
        snap = json.loads(rs.snapshot_path(s["root"]).read_text())
        assert snap["historical_runtime_manifest_available"] is False and "not provenance" in snap["meaning"]
        assert not rs.manifest_entries(rs.resolve_root(s["root"]))
    res = rs.integrity("F13/recovery3_r1", None)
    assert res["integrity"] == "OK_CURRENT_SNAPSHOT" and res["integrity_basis"].startswith("current_snapshot_")


def test_snapshot_detects_change_missing_and_new_files(tmp_path, monkeypatch):
    root = tmp_path / "res" / "X" / "y_r1"
    root.mkdir(parents=True)
    (root / "a.txt").write_text("a\n")
    (root / "b.txt").write_text("b\n")
    snaps = tmp_path / "snaps"
    snaps.mkdir()
    monkeypatch.setattr(rs, "SNAPSHOTS", snaps)
    key = str(root)
    rs.snapshot_path(key).parent.mkdir(parents=True, exist_ok=True)
    rs.snapshot_path(key).write_text(json.dumps({"snapshot_date": "2026-09-26", "files": {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in ("a.txt", "b.txt")},
                                                 "historical_runtime_manifest_available": False}))
    assert rs.integrity(key, None)["integrity"] == "OK_CURRENT_SNAPSHOT"
    (root / "a.txt").write_text("changed\n")
    (root / "b.txt").unlink()
    (root / "c.txt").write_text("new\n")
    probs = " ".join(rs.integrity(key, None)["problems"])
    assert "CHANGED a.txt" in probs and "MISSING b.txt" in probs and "NEW_AFTER_SNAPSHOT c.txt" in probs


def test_readiness_separates_provenance_integrity_and_science():
    reg = rs.load_registry()
    r = rs.root_readiness(reg, "F13/recovery3_r1")
    assert r["provenance_basis"].startswith("none") and r["integrity_basis"].startswith("current_snapshot_") and "scientific_status" in r
    r2 = rs.root_readiness(reg, "F05/common_shock_episodes_r1")
    assert r2["provenance_basis"] == "runtime_manifest"
