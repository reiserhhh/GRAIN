"""Tests for the revision registry and fail-closed stage modes (packet 12 Part C)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from grain_research.gptpro_foundation_v3 import revision_stage as rs  # noqa: E402

PY = sys.executable
CLI = ROOT / "scripts/run_grain_v3_revision_stage.py"
STAGE = ROOT / "results/grain_gptpro_foundation_v3/F00/revision_stage_r1"
PLAN = Path("/Volumes/mobile3/projects/project persona/configs/grain_gptpro_atomic_v3.json")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def make_root(base: Path, name: str, decision: bool = True) -> Path:
    r = base / name
    r.mkdir(parents=True)
    (r / "data.csv").write_text("a,b\n1,2\n")
    if decision:
        (r / "decision.md").write_text("# decision\n")
    files = [{"path": f.name, "sha256": _sha(f.read_bytes())} for f in sorted(r.iterdir())]
    (r / "output_manifest.json").write_text(json.dumps({"files": files}))
    return r


def entry(root: Path, kind="CURRENT_REVISION", **kw) -> dict:
    e = {"kind": kind, "curation": "TEST", "evidence": {"decision": str(root / "decision.md") if (root / "decision.md").exists() else None},
         "relations": [], "depends_on_revisions": [], "amendments": [], "tests": []}
    e.update(kw)
    return e


def reg_of(revs: dict, cards: dict | None = None, adapters: dict | None = None) -> dict:
    return {"schema": "grain-v3-revision-registry-v1", "revisions": revs, "cards": cards or {}, "adapters": adapters or {}}


def test_integrity_ok_then_hash_mismatch(tmp_path):
    r = make_root(tmp_path, "A")
    reg = reg_of({str(r): entry(r)})
    assert rs.integrity(str(r), reg["revisions"][str(r)])["integrity"] == "OK"
    (r / "data.csv").write_text("a,b\n1,3\n")
    res = rs.integrity(str(r), reg["revisions"][str(r)])
    assert res["integrity"] == "FAIL" and any(p.startswith("CHANGED data.csv") for p in res["problems"])
    assert rs.root_readiness(reg, str(r))["state"] == "NOT_READY_EVIDENCE_OR_INTEGRITY"


def test_declared_append_only_amendment_only(tmp_path):
    r = make_root(tmp_path, "B")
    cr = r / "common_record_r1"
    cr.mkdir()
    original = (r / "decision.md").read_bytes()
    (cr / "input_manifest.json").write_text(json.dumps({"packet_files": [{"path": "decision.md", "sha256": _sha(original)}]}))
    (r / "decision.md").write_bytes(original + b"\n## addendum\n")
    (r / "output_manifest.json").write_text(json.dumps({"files": [{"path": "data.csv", "sha256": _sha((r / "data.csv").read_bytes())}]}))
    e = entry(r)
    assert rs.integrity(str(r), e)["integrity"] == "FAIL"                       # undeclared change
    amend = {"file": "decision.md", "recorded_sha256": _sha(original), "recorded_prefix_bytes": len(original),
             "current_sha256": _sha((r / "decision.md").read_bytes()), "authority": "TEST_AUTH"}
    assert rs.integrity(str(r), entry(r, amendments=[amend]))["integrity"] == "OK_DECLARED_AMENDMENTS"
    assert rs.integrity(str(r), entry(r, amendments=[dict(amend, authority=None)]))["integrity"] == "FAIL"   # no authority
    (r / "decision.md").write_bytes(b"# edited in place\n## addendum\n")                                   # not an append
    assert rs.integrity(str(r), entry(r, amendments=[amend]))["integrity"] == "FAIL"


def test_rejected_legacy_is_readable_history_but_never_ready(tmp_path):
    r = make_root(tmp_path, "legacy")
    reg = reg_of({str(r): entry(r, kind="REJECTED_LEGACY", authority="AUDIT")})
    res = rs.root_readiness(reg, str(r))
    assert res["state"] == "NOT_READY_REJECTED_LEGACY" and res["integrity"] == "OK" and res["history_readable"] is True


def test_missing_evidence_and_unknown_mapping(tmp_path):
    r = make_root(tmp_path, "C", decision=False)
    reg = reg_of({str(r): entry(r)})
    res = rs.root_readiness(reg, str(r))
    assert res["state"] == "NOT_READY_EVIDENCE_OR_INTEGRITY" and "missing decision/evidence document" in res["reasons"]
    assert rs.root_readiness(reg, str(tmp_path / "nowhere"))["state"] == "NOT_READY_UNKNOWN_MAPPING"
    assert rs.card_readiness(reg, {}, "F99-99")["state"] == "NOT_READY_UNKNOWN_MAPPING"


def test_transitive_dependencies(tmp_path):
    a, b, c = (make_root(tmp_path, n) for n in "abc")
    reg = reg_of({str(a): entry(a, depends_on_revisions=[str(b)]), str(b): entry(b, depends_on_revisions=[str(c)]), str(c): entry(c)})
    assert rs.root_readiness(reg, str(a))["state"] == rs.READY
    (c / "data.csv").write_text("changed\n")
    res = rs.root_readiness(reg, str(a))
    assert res["state"] == "NOT_READY_DEPENDENCY" and res["dependencies"][0]["state"] == "NOT_READY_DEPENDENCY"


def test_supersession_semantics(tmp_path):
    x, s1, s2 = (make_root(tmp_path, n) for n in ("x", "s1", "s2"))
    sup = {"type": "SUPERSEDES_SCIENTIFIC_USE", "other": str(x), "scope": "t"}
    one = reg_of({str(x): entry(x), str(s1): entry(s1, relations=[sup])})
    assert rs.root_readiness(one, str(x))["state"] == "NOT_READY_SUPERSEDED"
    both = reg_of({str(x): entry(x), str(s1): entry(s1, relations=[sup]), str(s2): entry(s2, relations=[sup])})
    assert rs.root_readiness(both, str(x))["state"] == "NOT_READY_AMBIGUOUS_SUPERSESSION"
    interp = reg_of({str(x): entry(x), str(s1): entry(s1, relations=[dict(sup, type="SUPERSEDES_INTERPRETATION")])})
    assert rs.root_readiness(interp, str(x))["state"] == rs.READY


def test_failed_child_tests_fail_closed(tmp_path):
    r = make_root(tmp_path, "t")
    bad = tmp_path / "test_bad.py"
    bad.write_text("def test_fails():\n    assert False\n")
    reg = reg_of({str(r): entry(r, tests=[str(bad)])})
    assert rs.root_readiness(reg, str(r), run_tests=True)["state"] == "NOT_READY_CHILD_TESTS_FAILED"
    assert rs.root_readiness(reg, str(r), run_tests=False)["state"] == rs.READY


def test_card_readiness_blocks_on_transitive_card_dependency(tmp_path):
    a, b = make_root(tmp_path, "ca"), make_root(tmp_path, "cb", decision=False)
    plan = {"X": {"depends_on": ["Y"]}, "Y": {"depends_on": []}}
    reg = reg_of({str(a): entry(a), str(b): entry(b)}, cards={"X": {"current_evidence": [str(a)]}, "Y": {"current_evidence": [str(b)]}})
    res = rs.card_readiness(reg, plan, "X")
    assert res["state"] == "NOT_READY_DEPENDENCY" and res["blocked_dependencies"] == ["Y"] and res["own"]["state"] == rs.READY


def test_execute_refuses_unapproved_compute(tmp_path):
    auth = tmp_path / "auth.md"
    auth.write_text("authorized\n")
    ad = {"root": str(make_root(tmp_path, "ad")), "mode": "execute", "compute_cap_seconds": 60, "authorization_sha256": _sha(auth.read_bytes())}
    reg = reg_of({}, adapters={"x": ad})
    with pytest.raises(rs.StageRefusal):
        rs.authorize(reg, "x", None, 100)
    wrong = tmp_path / "wrong.md"
    wrong.write_text("other\n")
    with pytest.raises(rs.StageRefusal):
        rs.authorize(reg, "x", wrong, 100)
    with pytest.raises(rs.StageRefusal):
        rs.authorize(reg, "x", auth, 30)
    assert rs.authorize(reg, "x", auth, 60)["mode"] == "execute"
    with pytest.raises(rs.StageRefusal):
        rs.authorize(reg, "unknown", auth, 60)
    existing = tmp_path / "exists"
    existing.mkdir()
    with pytest.raises(rs.StageRefusal):
        rs.run_adapter(reg, "x", existing, auth, 60)


def _cli(*args):
    done = subprocess.run([PY, str(CLI), *args], capture_output=True, text=True, cwd=ROOT)
    return done.returncode, done.stdout


def test_cli_exit_status_and_output_on_real_registry():
    rc, out = _cli("integrity", "--root", "F13/personal_law_r2")
    assert rc == 0 and json.loads(out)["integrity"] == "OK"
    rc, out = _cli("readiness", "--root", "F02/r1/F02-01")
    assert rc == 3 and json.loads(out)["state"] == "NOT_READY_REJECTED_LEGACY"
    rc, out = _cli("execute", "--adapter", "r10_one_episode_replay", "--output", "/tmp/should_not_exist_grain_stage")
    assert rc == 2 and json.loads(out)["decision"] == "REFUSED"
    rc, out = _cli("reproduce", "--adapter", "r10_one_episode_replay", "--output", "/tmp/should_not_exist_grain_stage")
    assert rc == 2 and "registered for mode execute" in json.loads(out)["reason"]
    assert not Path("/tmp/should_not_exist_grain_stage").exists()


def test_stored_bounded_replays_passed_with_numeric_checks():
    reps = {p.name: json.loads((p / "stage_result.json").read_text()) for p in (STAGE / "replays").iterdir() if p.is_dir() and not p.name.startswith("._")}
    assert set(reps) == {"r24_r1_analysis", "r24_r2_cold_verify", "r10_episode_analysis", "r24_r1_canonical_replay",
                         "r10_one_episode_replay", "f16_sealed_cold_subset", "r08_gp01_collision"}
    for name, r in reps.items():
        assert r["status"] == "PASS" and r["child_exit"] == 0 and r["source_artifacts_unchanged"], name
        assert r["child_result"]["network_guard"] is True
    assert all(v["max_abs_err"] == 0.0 for v in reps["r10_one_episode_replay"]["child_result"]["episodes"].values())
    assert all(v["parsed_equal"] and v["byte_identical"] for v in reps["f16_sealed_cold_subset"]["child_result"]["bundles"].values())
    assert all(reps["r24_r1_analysis"]["child_result"]["byte_identical"].values())
    assert reps["r24_r2_cold_verify"]["child_result"]["rows"] == 25200


def test_collision_provenance_values_and_live_recompute():
    import verify_grain_r08_collision_provenance as v
    stored = json.loads((ROOT / "results/grain_gptpro_foundation_v3/F02/collision_provenance_r1/provenance.json").read_text())
    assert stored["hashes"]["status"] == "MATCH" and stored["equations"]["status"] == "PASS" and stored["regeneration"]["status"] == "REPRODUCED"
    eq = stored["equations"]["c2"]
    assert eq["future_tv_00_vs_11_by_horizon"] == ["0", "1/2", "1/2", "1/2", "1/2", "1/2"]
    assert eq["lag3_equal"] == {"iid": "1/2", "xor": "5/8"} and eq["hankel_rank"]["xor_h1"] == 2 and eq["hankel_rank"]["xor_h2"] == 4
    live = json.loads(json.dumps(v.equations(), sort_keys=True))
    assert live == stored["equations"]


def test_registry_consistency():
    reg = rs.load_registry()
    plan = {t["id"]: t for t in json.loads(PLAN.read_text())["tasks"]}
    assert set(reg["cards"]) == set(plan)
    legacy = {t["output_root"].split("results/grain_gptpro_foundation_v3/")[1] for t in plan.values()}
    assert {k for k, e in reg["revisions"].items() if e["kind"] == "REJECTED_LEGACY"} == legacy
    assert not any(e["scientific_readiness_eligible"] for e in reg["revisions"].values() if e["kind"] != "CURRENT_REVISION")
    for k, e in reg["revisions"].items():
        for rel in e.get("relations", []):
            assert rel["other"] in reg["revisions"], (k, rel)
        for a in e.get("amendments", []):
            assert a["append_only_prefix_found"] and a["authority"], (k, a)
    for aid, ad in reg["adapters"].items():
        assert ad["root"] in reg["revisions"], aid


def test_coverage_table_consistent():
    import csv
    cov = STAGE / "coverage"
    summ = json.loads((cov / "coverage_summary.json").read_text())
    rows = list(csv.DictReader((cov / "root_coverage.csv").open()))
    cards = list(csv.DictReader((cov / "card_readiness.csv").open()))
    assert summ["roots"] == len(rows) and summ["cards"] == len(cards) == 100
    assert sum(summ["readiness"].values()) == len(rows)
    assert all(r["readiness"] == "NOT_READY_REJECTED_LEGACY" for r in rows if r["kind"] == "REJECTED_LEGACY")
    assert all(r["adapter_gap"] for r in rows if not r["adapters"])
