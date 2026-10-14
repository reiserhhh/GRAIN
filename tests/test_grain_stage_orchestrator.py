import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("orch", ROOT / "scripts/run_grain_gptpro_stage.py")
orch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orch)


def make_card(tmp: Path, cid: str, deps: list[str], content: str = "x") -> dict:
    root = tmp / cid
    root.mkdir()
    (root / "summary.json").write_text(content)
    (root / "._summary.json").write_text("sidecar")
    files = [{"path": "summary.json", "sha256": orch.sha(root / "summary.json")}, {"path": "._summary.json", "sha256": "0" * 64}]
    (root / "output_manifest.json").write_text(json.dumps({"task_id": cid, "status": "PASS", "files": files}))
    return {"id": cid, "depends_on": deps, "output_root": str(root), "future_run_command": f"run {cid}"}


def test_verified_chain_is_runnable_and_sidecars_ignored(tmp_path):
    plan = {"A": make_card(tmp_path, "A", []), "B": make_card(tmp_path, "B", ["A"])}
    c = orch.check_stage("B", plan)
    assert c["runnable"] and c["own"]["sidecars_ignored"] == 1 and c["own"]["integrity"] == "OK"


def test_tampered_dependency_refuses(tmp_path):
    plan = {"A": make_card(tmp_path, "A", []), "B": make_card(tmp_path, "B", ["A"])}
    (tmp_path / "A" / "summary.json").write_text("tampered")
    c = orch.check_stage("B", plan)
    assert not c["runnable"] and c["failing_dependencies"] == ["A"]


def test_transitive_dependency_checked(tmp_path):
    plan = {"A": make_card(tmp_path, "A", []), "B": make_card(tmp_path, "B", ["A"]), "C": make_card(tmp_path, "C", ["B"])}
    (tmp_path / "A" / "summary.json").unlink()
    c = orch.check_stage("C", plan)
    assert c["failing_dependencies"] == ["A"]
