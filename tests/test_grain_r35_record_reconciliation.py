"""R35 packet-18 record reconciliation: typed, hash-pinned test re-records and documentation records; failures never reconciled."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3 import revision_stage as rs  # noqa: E402


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture
def world(tmp_path):
    root = tmp_path / "rootL"; (root / "common_record_r1").mkdir(parents=True)
    (root / "decision.md").write_text("# d\n")
    (root / "common_record_r1" / "tests.txt").write_text("......   [100%]\n6 passed in 0.5s\n")
    (root / "output_manifest.json").write_text(json.dumps({"files": [{"path": "decision.md", "sha256": _sha(root / "decision.md")}]}))
    rec = tmp_path / "rerecord.tests.txt"; rec.write_text("exit_code=0\ncommand=x\nrun_at=2026\n---\n6 passed\n")
    entry = {"kind": "CURRENT_REVISION", "curation": "T", "evidence": {"decision": str(root / "decision.md")}, "relations": [], "amendments": [], "depends_on_revisions": []}
    reg = {"schema": "t", "cards": {}, "adapters": {}, "revisions": {str(root): entry}}
    return {"reg": reg, "root": root, "rec": rec, "entry": entry}


def test_without_reconciliation_legacy_format_is_not_ready(world):
    assert rs.root_readiness(world["reg"], str(world["root"]))["state"] != rs.READY


def test_valid_rerecord_reconciles_with_typed_basis(world):
    world["entry"]["test_record_reconciliation"] = {"path": str(world["rec"]), "sha256": _sha(world["rec"]), "recorded_at": "2026"}
    rd = rs.root_readiness(world["reg"], str(world["root"]))
    assert rd["state"] == rs.READY and "reconciled" in rd["test_record_basis"]


def test_changed_rerecord_refuses(world):
    world["entry"]["test_record_reconciliation"] = {"path": str(world["rec"]), "sha256": _sha(world["rec"])}
    world["rec"].write_text("exit_code=0\ncommand=y\n")
    assert rs.root_readiness(world["reg"], str(world["root"]))["state"] != rs.READY


def test_failing_legacy_never_reconciled(world):
    (world["root"] / "common_record_r1" / "tests.txt").write_text("..F.  [100%]\n1 failed, 3 passed\n")
    world["entry"]["test_record_reconciliation"] = {"path": str(world["rec"]), "sha256": _sha(world["rec"])}
    assert rs.root_readiness(world["reg"], str(world["root"]))["state"] != rs.READY


def test_failing_rerecord_refuses(world):
    world["rec"].write_text("exit_code=1\ncommand=x\n---\n1 failed\n")
    world["entry"]["test_record_reconciliation"] = {"path": str(world["rec"]), "sha256": _sha(world["rec"])}
    assert rs.root_readiness(world["reg"], str(world["root"]))["state"] != rs.READY


def test_real_registry_after_reconciliation():
    reg = rs.load_registry()
    for k in ("P05/native_action_r1", "P05/c2_venue_dependence_r1"):
        rd = rs.root_readiness(reg, k)
        assert rd["state"] == rs.READY and rd.get("test_record_basis", "").startswith("legacy format reconciled")
    for k in ("F13/recovery2_r1", "P05/contrast_ledger_r1", "F16/construction_briefs_r2"):
        assert rs.root_readiness(reg, k)["state"] == rs.READY
    assert rs.root_readiness(reg, "F00/common_record_adapter_r1")["state"] == "NOT_READY_SUPERSEDED"
