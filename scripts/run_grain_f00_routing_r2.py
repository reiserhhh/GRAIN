#!/usr/bin/env python
"""F00-05 current revision (packet 13 Part A4): repair routing regenerated from the revision registry and the
requirement ledger. It routes every real gap to named cards and shows each card's current evidence and readiness.
This is contract/plan-base acceptance only; no old long run is restarted.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from grain_research.gptpro_foundation_v3 import revision_stage as rs  # noqa: E402

PERSONA = Path("/Volumes/mobile3/projects/project persona")
OUT = rs.RES / "F00/repair_routing_r2"
PLAN = PERSONA / "configs/grain_gptpro_atomic_v3.json"
LEDGER = PERSONA / "reports/claude_research_loop_20260926/ledger/REQUIREMENT_LEDGER_20260926.csv"
CROSS = PERSONA / "reports/gptpro_seminar_20260915/atomic_v3/SOURCE_TASK_CROSSWALK.csv"
LEGACY_ITEMS = [  # the five literal rows of the rejected legacy F00-05, re-verified against current evidence
    ("GP03 B1 mismatch", "F08-01;F08-05", "WITHDRAWN on actual replay (F00/baseline_inventory_r2)"),
    ("GP03 B2/M and evidence cards", "F08-02;F08-03;F08-04;F08-05;F08-06;F08-07", None),
    ("GP04 held corners and M4 calibration", "F09-01;F09-02;F09-03;F09-04;F09-05;F09-06", None),
    ("GP06 wrong-world cold scoring", "F16-01;F16-02;F16-03;F16-04;F16-05;F16-06", None),
    ("old r1 mutation", "", "CLOSED_NO_CHANGE: 80/80 legacy GP/PA/PV files match the 2026-09-15 snapshot (F00/baseline_inventory_r2)"),
]


def expand(spec: str) -> list[str]:
    return [c for c in re.split(r"[;,]", spec) if c]


def main() -> int:
    if OUT.exists():
        raise FileExistsError("repair_routing_r2 exists; preserve earlier runs")
    OUT.mkdir(parents=True)
    reg = rs.load_registry()
    plan = {t["id"]: t for t in json.loads(PLAN.read_text())["tasks"]}
    cache, icache = {}, {}
    card_state = {c: rs.card_readiness(reg, plan, c, cache=cache, icache=icache) for c in plan}

    def card_view(cards):
        ev = sorted({r for c in cards for r in reg["cards"][c]["current_evidence"]})
        own = {c: card_state[c]["own"]["state"] for c in cards}
        return {"cards": ";".join(cards), "current_evidence_roots": ";".join(ev),
                "cards_own_evidence_ready": sum(v == rs.READY for v in own.values()), "cards_total": len(cards),
                "cards_without_current_evidence": ";".join(c for c, v in own.items() if v == "NOT_READY_NO_CURRENT_EVIDENCE")}
    rows = []
    for issue, cards, status in LEGACY_ITEMS:
        cs = expand(cards)
        rows.append({"gap_source": "legacy F00-05 routing (literal rows re-verified)", "gap": issue, "status": status or "OPEN",
                     **(card_view(cs) if cs else {"cards": "", "current_evidence_roots": "", "cards_own_evidence_ready": 0, "cards_total": 0, "cards_without_current_evidence": ""})})
    cross = {r["requirement"]: expand(r["atomic_tasks"]) for r in csv.DictReader(CROSS.open())}
    for r in csv.DictReader(LEDGER.open()):
        gap = r["remaining_executable_gap"].split(" | F17-01")[0]
        cs = [c for c in cross.get(r["requirement"], []) if c in plan]
        rows.append({"gap_source": f"requirement ledger {r['requirement']} ({r['acceptance_decision']})", "gap": gap or "(none recorded)",
                     "status": r["acceptance_decision"], **card_view(cs)})
    no_ev = [c for c, v in card_state.items() if v["own"]["state"] == "NOT_READY_NO_CURRENT_EVIDENCE"]
    for c in no_ev:
        rows.append({"gap_source": "card without current evidence", "gap": f"{c}: only the rejected legacy root exists", "status": "OPEN_NO_CURRENT_EVIDENCE",
                     **card_view([c])})
    with (OUT / "repair_routing.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    gaps_needing_cards = [r for r in rows if not r["status"].startswith(("CLOSED", "WITHDRAWN"))]
    baseline = json.loads((rs.RES / "F00/baseline_inventory_r2/baseline_inventory.json").read_text())
    acc = {"task_id": "F00-05", "revision": "F00/repair_routing_r2", "registry_sha256": rs.sha(rs.REGISTRY), "ledger_sha256": rs.sha(LEDGER),
           "rows": len(rows), "every_open_gap_has_responsible_card": all(r["cards"] for r in gaps_needing_cards),
           "gaps_without_card": [r["gap"] for r in gaps_needing_cards if not r["cards"]],
           "original_results_unchanged": {"legacy_gp_pa_pv_files_match_snapshot": baseline["legacy_roots"]["gp_pa_pv_files_vs_20260915_snapshot"],
                                          "rejected_legacy_card_roots": baseline["legacy_roots"]["rejected_legacy_card_roots"]},
           "cards_without_current_evidence": no_ev, "no_old_long_runs_restarted": True,
           "scope": "contract/plan-base acceptance only: routing and evidence pointers, no scientific acceptance of any card"}
    (OUT / "acceptance.json").write_text(json.dumps(acc, indent=1) + "\n")
    print(json.dumps({k: acc[k] for k in ("rows", "every_open_gap_has_responsible_card", "gaps_without_card", "cards_without_current_evidence")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
