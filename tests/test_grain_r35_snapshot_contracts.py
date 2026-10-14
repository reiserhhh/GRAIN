"""R35 packet-17: current integrity snapshots are typed LIMITED inputs; contracts must permit them; live hashes checked."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3 import revision_stage as rs  # noqa: E402


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class Child:
    def __init__(self):
        self.calls = 0

    def __call__(self, cmd, **kw):
        self.calls += 1
        (Path(cmd[-1]) / "adapter_result.json").write_text(json.dumps({"status": "PASS"}))

        class Done:
            returncode, stderr = 0, ""
        return Done()


@pytest.fixture
def world(tmp_path, monkeypatch):
    child = Child()
    monkeypatch.setattr(rs.subprocess, "run", child)
    snaps = tmp_path / "snapshots"; snaps.mkdir()
    monkeypatch.setattr(rs, "SNAPSHOTS", snaps)
    snap_root = tmp_path / "snapA"; snap_root.mkdir()
    (snap_root / "data.csv").write_text("x\n1\n"); (snap_root / "decision.md").write_text("# d\n")
    rs.snapshot_path(str(snap_root)).write_text(json.dumps({"root": str(snap_root), "snapshot_date": "2026-09-26", "kind": "CURRENT_INTEGRITY_SNAPSHOT",
        "historical_runtime_manifest_available": False, "excluded_prefixes": [],
        "files": {n: _sha((snap_root / n).read_bytes()) for n in ("data.csv", "decision.md")}}))
    run_root = tmp_path / "runB"; run_root.mkdir()
    (run_root / "decision.md").write_text("# r\n")
    (run_root / "output_manifest.json").write_text(json.dumps({"files": [{"path": "decision.md", "sha256": _sha(b"# r\n")}]}))
    code = tmp_path / "code.py"; code.write_text("print(1)\n")
    auth = tmp_path / "auth.md"; auth.write_text("authorized\n")

    def entry(root, deps=(), kind="CURRENT_REVISION"):
        return {"kind": kind, "curation": "T", "evidence": {"decision": str(root / "decision.md")}, "relations": [], "amendments": [], "depends_on_revisions": list(deps)}
    reg = {"schema": "t", "cards": {}, "revisions": {str(snap_root): entry(snap_root), str(run_root): entry(run_root, [str(snap_root)])}, "adapters": {}}

    def adapter(root, requires, permitted=None):
        ad = {"root": str(root), "mode": "execute", "dependency_mode": "scientific", "compute_cap_seconds": 10, "authorization_sha256": _sha(auth.read_bytes()),
              "inputs": [{"role": "data", "path": str(snap_root / "data.csv"), "kind": "file", "sha256": _sha((snap_root / "data.csv").read_bytes())}],
              "code": [{"path": str(code), "sha256": _sha(code.read_bytes())}], "requires_ready": [str(r) for r in requires]}
        if permitted is not None:
            ad["evidence_levels_permitted"] = permitted
        return ad
    return {"reg": reg, "snap": snap_root, "run": run_root, "auth": auth, "child": child, "tmp": tmp_path, "adapter": adapter}


def launch(w, ad, out="out"):
    w["reg"]["adapters"]["a"] = ad
    return rs.run_adapter(w["reg"], "a", w["tmp"] / out, w["auth"], 10)


def refused(w, ad, out="out"):
    w["reg"]["adapters"]["a"] = ad
    with pytest.raises(rs.StageRefusal) as e:
        rs.run_adapter(w["reg"], "a", w["tmp"] / out, w["auth"], 10)
    assert w["child"].calls == 0
    return str(e.value)


def test_permitted_snapshot_contract_launches_with_typed_level(world):
    res = launch(world, world["adapter"](world["snap"], [world["snap"]], ["runtime_manifest", "current_snapshot_limited"]))
    assert world["child"].calls == 1 and res["status"] == "PASS" and res["grants_readiness"] is False
    assert res["preflight"]["evidence_levels"] == {str(world["snap"]): "current_snapshot_limited"}
    rd = rs.root_readiness(world["reg"], str(world["snap"]))
    assert rd["state"] == rs.READY and rd["evidence_level_own"] == "current_snapshot_limited" and rd["provenance_basis"].startswith("none")


def test_changed_or_new_file_after_snapshot_fails_live_check(world):
    (world["snap"] / "data.csv").write_text("x\n2\n")  # adapter pin re-taken so only the live snapshot check can catch it
    msg = refused(world, world["adapter"](world["snap"], [world["snap"]], ["runtime_manifest", "current_snapshot_limited"]))
    assert "NOT_READY" in msg and rs.evidence_level(world["reg"], str(world["snap"])) == "none"
    (world["snap"] / "data.csv").write_text("x\n1\n"); (world["snap"] / "extra.txt").write_text("late\n")
    assert rs.integrity(str(world["snap"]), world["reg"]["revisions"][str(world["snap"])])["integrity"] == "FAIL"


def test_runtime_required_contract_refuses_snapshot_by_default(world):
    msg = refused(world, world["adapter"](world["snap"], [world["snap"]]))          # no declaration -> runtime_manifest only
    assert "does not permit evidence level current_snapshot_limited" in msg
    msg = refused(world, world["adapter"](world["snap"], [world["snap"]], ["runtime_manifest"]))
    assert "does not permit evidence level current_snapshot_limited" in msg


def test_rejected_root_refused_even_if_snapshot_permitted(world):
    world["reg"]["revisions"][str(world["snap"])]["kind"] = "REJECTED_LEGACY"
    msg = refused(world, world["adapter"](world["snap"], [world["snap"]], ["runtime_manifest", "current_snapshot_limited"]))
    assert "NOT_READY_REJECTED_LEGACY" in msg


def test_weakest_level_propagates_through_dependencies(world):
    assert rs.evidence_level(world["reg"], str(world["run"])) == "current_snapshot_limited"
    msg = refused(world, world["adapter"](world["run"], [world["run"]], ["runtime_manifest"]))
    assert f"does not permit evidence level current_snapshot_limited for {world['run']}" in msg


def test_real_registry_contracts_declare_levels():
    reg = rs.load_registry()
    assert all("evidence_levels_permitted" in ad for ad in reg["adapters"].values())
    assert reg["adapters"]["f16_sealed_cold_subset"]["evidence_levels_permitted"] == ["runtime_manifest"]
    assert "current_snapshot_limited" in reg["adapters"]["r24_r1_canonical_replay"]["evidence_levels_permitted"]
    assert rs.evidence_level(reg, "F13/personal_law_r1") == "current_snapshot_limited"   # via snapshot dependencies
    assert rs.evidence_level(reg, "F16/serialize_full_r1") == "runtime_manifest"


def test_twelve_row_reconciliation_matches_live_registry():
    """Every root recorded in the snapshot is still current with the recorded readiness/level; later live roots were
    declared by packet 17 itself (after the snapshot), never silently."""
    import csv as _csv
    OUTR = ROOT / "results/grain_gptpro_foundation_v3/F00/r35_twelve_row_reconciliation_r1"
    rec = json.loads((OUTR / "reconciliation.json").read_text())
    rows = list(_csv.DictReader((OUTR / "reconciliation_roots.csv").open()))
    reg = rs.load_registry()
    assert rec["rows"] == ["R05", "R06", "R09", "R10", "R12", "R18", "R24", "R26", "R27", "R31", "R34", "A02"]
    # Packet 18 Part D reconciled these roots on purpose (F00/record_reconciliation_r1); every other recorded root must be unchanged.
    later = {"P05/native_action_r1": rs.READY, "P05/c2_venue_dependence_r1": rs.READY, "F13/recovery2_r1": rs.READY,
             "P05/contrast_ledger_r1": rs.READY, "F16/construction_briefs_r2": rs.READY, "F00/common_record_adapter_r1": "NOT_READY_SUPERSEDED"}
    for r in rows:
        assert reg["revisions"][r["root"]]["kind"] == "CURRENT_REVISION"
        assert rs.root_readiness(reg, r["root"])["state"] == later.get(r["root"], r["readiness"])
        assert rs.evidence_level(reg, r["root"]) == r["evidence_level_closure"]
    for R, s_ in rec["summary"].items():
        recorded = {r["root"] for r in rows if r["requirement"] == R}
        assert s_["current_roots"] == len(recorded)
        live = {k for k, v in reg["revisions"].items() if R in v.get("requirements", []) and v["kind"] == "CURRENT_REVISION"}
        # later additions must be declared by a later packet/supervisor authority (packets 17-34, supervisors 2042/2143/2245/2347/0048/0149/0251/0352/0452/0553/0654/0754/0855/0955/1056/1157, seminars 19-23), or be the packet-19 verified A02 mapping
        a02_mapping = {"F05/invariants_r1", "F04/information_full_r1"}
        assert recorded <= live and all(reg["revisions"][k]["authority"] in ("PACKET_17", "PACKET_18", "PACKET_19", "SUPERVISOR_2042", "PACKET_20", "SUPERVISOR_2143", "PACKET_21", "SUPERVISOR_2245", "PACKET_22", "SUPERVISOR_2347", "PACKET_23", "SUPERVISOR_0048", "PACKET_24", "SUPERVISOR_0149", "SEMINAR_19", "PACKET_25", "SUPERVISOR_0251", "PACKET_26", "SUPERVISOR_0352", "SEMINAR_20", "PACKET_27", "SUPERVISOR_0452", "PACKET_28", "SUPERVISOR_0553", "PACKET_29", "SUPERVISOR_0654", "SEMINAR_21", "PACKET_30", "SUPERVISOR_0754", "PACKET_31", "SUPERVISOR_0855", "SEMINAR_22", "PACKET_32", "SUPERVISOR_0955", "PACKET_33", "SUPERVISOR_1056", "SEMINAR_23", "PACKET_34", "SUPERVISOR_1157")
                                        or (R == "A02" and k in a02_mapping) for k in live - recorded)
    assert rec["summary"]["A02"]["current_roots"] == 0 and rec["summary"]["A02"]["rejected_legacy_roots"] == 6
