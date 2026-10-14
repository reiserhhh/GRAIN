#!/usr/bin/env python
"""R35 packet-17 Part D: bounded evidence/dependency reconciliation of the 12 rows citing the cross-cutting R35 gap.
Reads the live registry, integrity/readiness functions and the requirement ledger. No reruns, no registration."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3 import revision_stage as rs  # noqa: E402

LEDGER = Path("/Volumes/mobile3/projects/project persona/reports/claude_research_loop_20260926/ledger/REQUIREMENT_LEDGER_20260926.csv")
OUT = ROOT / "results/grain_gptpro_foundation_v3/F00/r35_twelve_row_reconciliation_r1"
ROWS = ["R05", "R06", "R09", "R10", "R12", "R18", "R24", "R26", "R27", "R31", "R34", "A02"]
CROSS = "cross-cutting: F00 run_stage/--stage orchestration and legacy r1 roots (R35)"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    reg = rs.load_registry()
    led = {r["requirement"]: r for r in csv.DictReader(LEDGER.open())}
    icache, out_rows, summary = {}, [], {}
    incoming = {}
    for k, v in reg["revisions"].items():
        for rel in v.get("relations", []):
            incoming.setdefault(rel.get("other"), []).append({"from": k, "type": rel["type"]})
    for R in ROWS:
        L = led[R]
        roots = [k for k, v in reg["revisions"].items() if R in v.get("requirements", [])]
        cur = [k for k in roots if reg["revisions"][k]["kind"] == "CURRENT_REVISION"]
        rej = [k for k in roots if reg["revisions"][k]["kind"] == "REJECTED_LEGACY"]
        recs = []
        for k in cur:
            v = reg["revisions"][k]
            rd = rs.root_readiness(reg, k, icache=icache)
            lv = rs.evidence_level(reg, k, icache=icache)
            tests_txt = rs.resolve_root(k) / "common_record_r1/tests.txt"
            legacy_tests_format = bool(tests_txt.exists() and not tests_txt.read_text().startswith(("exit_code=", "no packet-specific")) and " passed" in tests_txt.read_text())
            rec = {"requirement": R, "root": k, "kind": v["kind"], "cards": v.get("cards", []), "scientific_status": v.get("status"),
                   "readiness": rd["state"], "reasons": rd.get("reasons", []), "integrity_basis": rd.get("integrity_basis"),
                   "provenance_basis": rd.get("provenance_basis"), "evidence_level_closure": lv,
                   "depends_on": v.get("depends_on_revisions", []), "outgoing_relations": [r["type"] + "->" + r["other"] for r in v.get("relations", [])],
                   "incoming_relations": [i["type"] + "<-" + i["from"] for i in incoming.get(k, [])],
                   "legacy_tests_txt_format_passing_output": legacy_tests_format}
            recs.append(rec)
            out_rows.append(rec)
        gap = L["remaining_executable_gap"].split(" | F17-01 unmet")[0]
        missing = "; ".join(x.strip() for x in gap.replace(CROSS, "").split(";") if x.strip()) or "(no computation named beyond the cross-cutting clause)"
        ready = [r for r in recs if r["readiness"] == rs.READY]
        summary[R] = {"title": L["title"], "sources": L["sources"], "atomic_tasks": L["atomic_tasks"], "science_resolution": L["science_resolution"],
                      "acceptance_decision": L["acceptance_decision"], "protocol": L["protocol"],
                      "current_roots": len(cur), "rejected_legacy_roots": len(rej), "rejected_legacy_list": rej,
                      "ready_runtime_manifest": [r["root"] for r in ready if r["evidence_level_closure"] == "runtime_manifest"],
                      "ready_snapshot_limited": [r["root"] for r in ready if r["evidence_level_closure"] == "current_snapshot_limited"],
                      "not_ready": {r["root"]: r["reasons"][:2] for r in recs if r["readiness"] != rs.READY},
                      "legacy_tests_format_false_negative": [r["root"] for r in recs if r["legacy_tests_txt_format_passing_output"] and r["readiness"] != rs.READY],
                      "usable_now": ("no current revision; only rejected legacy history (readable, never ready)" if not cur else
                                     f"{len(ready)}/{len(cur)} current roots READY: {len([r for r in ready if r['evidence_level_closure']=='runtime_manifest'])} on run-time manifests, "
                                     f"{len([r for r in ready if r['evidence_level_closure']=='current_snapshot_limited'])} only on current snapshots (LIMITED; usable only by contracts that permit it)"),
                      "genuine_computation_missing": missing}
    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(out_rows[0].keys())
    with (OUT / "reconciliation_roots.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader()
        for r in out_rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    res = {"rows": ROWS, "summary": summary, "registry_sha256": sha(rs.REGISTRY), "ledger_sha256": sha(LEDGER),
           "stage_library_sha256": sha(ROOT / "grain_research/gptpro_foundation_v3/revision_stage.py"), "created": datetime.now(timezone.utc).isoformat(),
           "policy": "snapshots are typed LIMITED inputs (current bytes only); never historical run-time provenance or preregistration; REJECTED_LEGACY never ready; readiness grants no scientific/protocol acceptance"}
    (OUT / "reconciliation.json").write_text(json.dumps(res, indent=1) + "\n")
    for R, s in summary.items():
        print(R, "|", s["usable_now"], "| not ready:", list(s["not_ready"]), "| missing:", s["genuine_computation_missing"][:150])


if __name__ == "__main__":
    main()
