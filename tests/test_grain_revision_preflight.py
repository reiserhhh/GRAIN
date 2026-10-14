"""Preflight repair (packet 13 Part A1/A2): no child process may start on invalid declared inputs."""
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
    root = tmp_path / "rootA"
    root.mkdir()
    (root / "data.csv").write_text("x\n1\n")
    (root / "decision.md").write_text("# d\n")
    (root / "output_manifest.json").write_text(json.dumps({"files": [{"path": n, "sha256": _sha((root / n).read_bytes())} for n in ("data.csv", "decision.md")]}))
    code = tmp_path / "code.py"
    code.write_text("print(1)\n")
    auth = tmp_path / "auth.md"
    auth.write_text("authorized\n")
    entry = {"kind": "CURRENT_REVISION", "curation": "T", "evidence": {"decision": str(root / "decision.md")}, "relations": [], "amendments": [], "depends_on_revisions": []}
    ad = {"root": str(root), "mode": "execute", "dependency_mode": "scientific", "compute_cap_seconds": 10, "authorization_sha256": _sha(auth.read_bytes()),
          "inputs": [{"role": "data", "path": str(root / "data.csv"), "kind": "file", "sha256": _sha((root / "data.csv").read_bytes())}],
          "code": [{"path": str(code), "sha256": _sha(code.read_bytes())}], "requires_ready": [str(root)]}
    reg = {"schema": "grain-v3-revision-registry-v1", "revisions": {str(root): entry}, "cards": {}, "adapters": {"a": ad}}
    return {"reg": reg, "root": root, "auth": auth, "code": code, "child": child, "tmp": tmp_path}


def refused(w, **kw):
    with pytest.raises(rs.StageRefusal) as e:
        rs.run_adapter(w["reg"], "a", w["tmp"] / kw.get("out", "out"), kw.get("auth", w["auth"]), kw.get("cap", 10))
    assert w["child"].calls == 0
    return str(e.value)


def test_valid_scientific_adapter_launches_exactly_one_child(world):
    res = rs.run_adapter(world["reg"], "a", world["tmp"] / "out", world["auth"], 10)
    assert world["child"].calls == 1 and res["status"] == "PASS" and res["grants_readiness"] is False and not res["diagnostic_only"]


def test_supervisor_case_missing_unregistered_root_now_refused(world):
    rep = json.loads((ROOT / "results/grain_gptpro_foundation_v3/F00/revision_stage_r2/defect_reproduction.json").read_text())
    assert rep["defect_reproduced"] and rep["child_invocations"] == 1          # the pre-repair behaviour, recorded
    world["reg"]["adapters"]["a"]["root"] = str(world["tmp"] / "missing_unregistered_root")
    assert "not registered" in refused(world)


def test_missing_and_changed_inputs_refused(world):
    (world["root"] / "data.csv").write_text("x\n2\n")
    assert "CHANGED input data" in refused(world)
    (world["root"] / "data.csv").unlink()
    assert "MISSING input data" in refused(world)


def test_changed_code_pin_refused(world):
    world["code"].write_text("print(2)\n")
    assert "CHANGED code" in refused(world)


def test_unknown_mapping_input_refused(world):
    outside = world["tmp"] / "outside.csv"
    outside.write_text("z\n")
    world["reg"]["adapters"]["a"]["inputs"].append({"role": "x", "path": str(outside), "kind": "file", "sha256": _sha(outside.read_bytes())})
    assert "unknown mapping" in refused(world)


def test_ambiguous_supersession_refused(world):
    for n in ("s1", "s2"):
        world["reg"]["revisions"][n] = {"kind": "CURRENT_REVISION", "relations": [{"type": "SUPERSEDES_SCIENTIFIC_USE", "other": str(world["root"])}]}
    assert "ambiguous supersession" in refused(world)


def test_stale_authorization_and_unapproved_compute_refused(world):
    world["auth"].write_text("edited after registration\n")
    assert "authorization document hash" in refused(world)
    world["auth"].write_text("authorized\n")
    assert "unapproved compute" in refused(world, cap=5)
    assert "requires the registered authorization" in refused(world, auth=None)


def test_scientific_dependency_not_ready_refused(world):
    (world["root"] / "decision.md").unlink()
    world["reg"]["revisions"][str(world["root"])]["evidence"]["decision"] = str(world["root"] / "decision.md")
    world["reg"]["adapters"]["a"]["inputs"][0]["path"] = str(world["root"] / "data.csv")
    msg = refused(world)
    assert "scientific dependency" in msg


def test_existing_output_refused(world):
    (world["tmp"] / "out").mkdir()
    assert "fresh" in refused(world)


def test_diagnostic_history_requires_explicit_listing_and_grants_nothing(world):
    world["reg"]["revisions"][str(world["root"])]["kind"] = "REJECTED_LEGACY"
    world["reg"]["adapters"]["a"].update({"dependency_mode": "diagnostic", "requires_ready": []})
    assert "not listed in history_roots" in refused(world)
    world["reg"]["adapters"]["a"]["history_roots"] = [str(world["root"])]
    before = rs.root_readiness(world["reg"], str(world["root"]))["state"]
    res = rs.run_adapter(world["reg"], "a", world["tmp"] / "out", world["auth"], 10)
    assert world["child"].calls == 1 and res["status"] == "PASS" and res["diagnostic_only"] and res["grants_readiness"] is False
    assert rs.root_readiness(world["reg"], str(world["root"]))["state"] == before == "NOT_READY_REJECTED_LEGACY"


def test_real_registry_preflight_refuses_without_launch(monkeypatch, tmp_path):
    child = Child()
    monkeypatch.setattr(rs.subprocess, "run", child)
    reg = rs.load_registry()
    with pytest.raises(rs.StageRefusal):
        rs.run_adapter(reg, "r10_one_episode_replay", tmp_path / "o", None, 300)      # no authorization
    with pytest.raises(rs.StageRefusal):
        rs.run_adapter(reg, "r10_one_episode_replay", tmp_path / "o", tmp_path / "nope.md", 300)
    assert child.calls == 0
