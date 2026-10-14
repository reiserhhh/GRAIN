#!/usr/bin/env python
"""Stage orchestrator for the 100 atomic V3 cards (F00-04 execution contract; engineering r1).

Verification and refusal only. This never runs research, and it never modifies any output.

  --check-all            verify every card's r1 output_manifest hashes and its dependencies' integrity
  --stage CARD --replay  re-verify one card and its dependency chain (the "replay existing numbers" mode)
  --stage CARD --run     refuse unless the full dependency chain verifies; if it does, print the card's
                         planned command (dry run). Execution needs a separate packet authorization.
  --evidence-refs        resolve every results reference in the requirement-ledger evidence to an existing
                         directory with a decision or summary

macOS AppleDouble sidecars (`._*`) listed in legacy manifests are filesystem metadata. They are excluded
from integrity checks and counted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PLAN = Path("/Volumes/mobile3/projects/project persona/configs/grain_gptpro_atomic_v3.json")
LEDGER = Path("/Volumes/mobile3/projects/project persona/reports/claude_research_loop_20260926/ledger/build_requirement_ledger.py")
RES = REPO / "results/grain_gptpro_foundation_v3"
OUT = RES / "F00/stage_orchestrator_r1"
REF_RE = re.compile(r"\b((?:F\d\d|P05|R17)/[A-Za-z0-9_]+_r\d+)\b")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_plan(plan: Path = PLAN) -> dict[str, dict]:
    return {t["id"]: t for t in json.loads(plan.read_text())["tasks"]}


def verify_card(task: dict) -> dict:
    root = Path(task["output_root"])
    om = root / "output_manifest.json"
    if not om.exists():
        return {"card": task["id"], "integrity": "MISSING_MANIFEST", "checked": 0, "mismatch": [], "missing": [], "sidecars_ignored": 0}
    manifest = json.loads(om.read_text())
    checked, mismatch, missing, side = 0, [], [], 0
    for entry in manifest.get("files", []):
        name = entry["path"]
        if Path(name).name.startswith("._"):
            side += 1
            continue
        p = root / name
        if not p.exists():
            missing.append(name)
        elif sha(p) != entry["sha256"]:
            mismatch.append(name)
        checked += 1
    status = "OK" if not mismatch and not missing and checked > 0 else ("EMPTY" if checked == 0 else "FAIL")
    return {"card": task["id"], "integrity": status, "checked": checked, "mismatch": mismatch, "missing": missing,
            "sidecars_ignored": side, "manifest_status": manifest.get("status")}


def chain(card: str, plan: dict[str, dict], seen: set[str] | None = None) -> list[str]:
    seen = set() if seen is None else seen
    for dep in plan[card]["depends_on"]:
        if dep not in seen:
            seen.add(dep)
            chain(dep, plan, seen)
    return sorted(seen)


def check_stage(card: str, plan: dict[str, dict]) -> dict:
    own = verify_card(plan[card])
    deps = {d: verify_card(plan[d]) for d in chain(card, plan)}
    bad = [d for d, v in deps.items() if v["integrity"] != "OK"]
    return {"card": card, "own": own, "dependencies": len(deps), "failing_dependencies": bad,
            "runnable": own["integrity"] in ("OK", "MISSING_MANIFEST") and not bad}


def evidence_refs() -> list[dict]:
    text = LEDGER.read_text()
    rows = []
    for ref in sorted(set(REF_RE.findall(text))):
        d = RES / ref
        rows.append({"ref": ref, "exists": d.is_dir(),
                     "has_decision_or_summary": d.is_dir() and any((d / f).exists() for f in ("decision.md", "summary.json", "DECISION.md", "decision.json", "INVENTORY.md", "REPORT.md")),
                     "has_common_record": (d / "common_record_r1").is_dir()})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-all", action="store_true")
    ap.add_argument("--stage")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--evidence-refs", action="store_true")
    a = ap.parse_args()
    plan = load_plan()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.check_all:
        rows = []
        for card in plan:
            c = check_stage(card, plan)
            rows.append({"card": card, "integrity": c["own"]["integrity"], "files_checked": c["own"]["checked"],
                         "sidecars_ignored": c["own"]["sidecars_ignored"], "mismatch": ";".join(c["own"]["mismatch"]),
                         "missing": ";".join(c["own"]["missing"]), "dependencies": c["dependencies"],
                         "failing_dependencies": ";".join(c["failing_dependencies"]), "runnable_by_contract": c["runnable"]})
        with (OUT / "stage_integrity.csv").open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        summary = {"cards": len(rows), "integrity_ok": sum(r["integrity"] == "OK" for r in rows),
                   "integrity_fail": [r["card"] for r in rows if r["integrity"] != "OK"],
                   "runnable_by_contract": sum(r["runnable_by_contract"] for r in rows),
                   "sidecars_ignored": sum(r["sidecars_ignored"] for r in rows), "plan_sha256": sha(PLAN),
                   "orchestrator_sha256": sha(Path(__file__))}
        (OUT / "stage_integrity_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(json.dumps(summary, indent=1))
    if a.evidence_refs:
        refs = evidence_refs()
        with (OUT / "ledger_evidence_refs.csv").open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(refs[0]))
            w.writeheader()
            w.writerows(refs)
        dangling = [r["ref"] for r in refs if not r["exists"]]
        undocumented = [r["ref"] for r in refs if r["exists"] and not r["has_decision_or_summary"]]
        out = {"refs": len(refs), "dangling": dangling, "without_decision_or_summary": undocumented,
               "with_common_record": sum(r["has_common_record"] for r in refs)}
        (OUT / "ledger_evidence_refs_summary.json").write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out, indent=1))
    if a.stage:
        c = check_stage(a.stage, plan)
        if a.run:
            if not c["runnable"]:
                print(json.dumps({"card": a.stage, "decision": "REFUSED", "reason": c}, indent=1))
                return 2
            print(json.dumps({"card": a.stage, "decision": "DRY_RUN_ONLY", "planned_command": plan[a.stage]["future_run_command"],
                              "note": "execution requires separate packet authorization"}, indent=1))
            return 0
        print(json.dumps(c, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
