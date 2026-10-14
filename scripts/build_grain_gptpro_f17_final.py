#!/usr/bin/env python3
"""F17-05 final acceptance: 44 computational questions + 6 human items.

Reads the requirement ledger (after F17-01 integration) and classifies each
row separately as RESOLVED_FINITE, INCONCLUSIVE, BLOCKED,
SOURCES_UNAVAILABLE, OPEN_PARTIAL (computational) or HUMAN_PREP_READY /
HUMAN_PREP_OPEN (human). Three numbers are reported separately:
full_scope_planned, computational_questions_resolved, scientifically_supported.
A plan or test PASS never counts as theory being correct.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

LEDGER = Path("/Volumes/mobile3/projects/project persona/reports/claude_research_loop_20260926/ledger/REQUIREMENT_LEDGER_20260926.csv")
BLOCKED_WORDS = ("blocked", "never saved", "not authorized")


def classify(r):
    """Separate status for one ledger row."""
    if r["scope"].startswith("FUTURE"):
        return "HUMAN_PREP_READY" if r["acceptance_decision"] == "HUMAN_STAGE_PREP_READY" else "HUMAN_PREP_OPEN"
    if r.get("science_resolution") == "RESOLVED_SOURCE_DEMANDS_MET":
        return "RESOLVED_FINITE"
    if r.get("f17_01_audit") == "MISSING":
        return "MISSING_SOURCE_DEMAND"
    if r["science"].startswith("INCONCLUSIVE"):
        return "INCONCLUSIVE"
    if r["science"] == "PARTIAL_SOURCES_UNAVAILABLE":
        return "SOURCES_UNAVAILABLE"
    if any(w in r["remaining_executable_gap"].lower() for w in BLOCKED_WORDS):
        return "BLOCKED"
    return "OPEN_PARTIAL"


def main():
    """Write final_acceptance.json and final_report.md."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-notes", type=Path, default=None, help="merged F17-01 findings (markdown) to cite")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = list(csv.DictReader(LEDGER.open()))
    items = [{"requirement": r["requirement"], "title": r["title"], "scope": r["scope"], "status": classify(r),
              "implementation": r["implementation"], "protocol": r["protocol"], "science": r["science"],
              "remaining": r["remaining_executable_gap"], "next": r["next_action"], "f17_01_audit": r.get("f17_01_audit"),
              "protocol_complete": r["protocol"] == "DONE"} for r in rows]
    comp = [i for i in items if not i["scope"].startswith("FUTURE")]
    human = [i for i in items if i["scope"].startswith("FUTURE")]
    counts = Counter(i["status"] for i in comp)
    supported = sum(i["science"] in ("SUPPORTED_FINITE", "FINITE_CERTIFICATE") for i in comp)
    final = {
        "full_scope_planned": {"computational_questions": len(comp), "human_items": len(human), "total": len(items)},
        "computational_questions_resolved": counts.get("RESOLVED_FINITE", 0),
        "resolved_and_protocol_complete": sum(i["status"] == "RESOLVED_FINITE" and i["protocol_complete"] for i in comp),
        "f17_01_source_audit": dict(Counter(i["f17_01_audit"] for i in items)),
        "scientifically_supported_method_level": supported,
        "psychological_claim_level": "below L0 for every human construct (not assessed on human data)",
        "computational_status_counts": dict(counts),
        "human_status_counts": dict(Counter(i["status"] for i in human)),
        "items": items,
        "plan_closed": counts.get("RESOLVED_FINITE", 0) == len(comp),
        "note": "only per-item RESOLVED_FINITE counts toward closure; inconclusive, blocked, unavailable and open items are listed separately; test/plan PASS is not theory correctness",
        "audit_notes": str(args.audit_notes) if args.audit_notes else None,
    }
    (args.output / "final_acceptance.json").write_text(json.dumps(final, indent=1, ensure_ascii=False) + "\n")
    lines = ["# GRAIN full-loop final acceptance (F17-05)", "",
             f"- Full scope planned: {len(comp)} computational questions + {len(human)} human-stage items = {len(items)}.",
             f"- Computational questions resolved with source demands met (both F17-01 audits ADDRESSED): {final['computational_questions_resolved']} of {len(comp)}; of these protocol-complete: {final['resolved_and_protocol_complete']}.",
             f"- F17-01 merged source audit over all 50 rows: {final['f17_01_source_audit']}.",
             f"- Science label supported at method level in declared worlds (finite certificates/simulations), regardless of source-demand coverage: {supported} of {len(comp)}.",
             "- Psychological claims: below L0 for every human construct (not assessed on human data); no human validity is claimed.",
             f"- Plan closed: {final['plan_closed']}.", "", "## Not resolved (listed separately)", ""]
    for i in comp:
        if i["status"] != "RESOLVED_FINITE":
            lines.append(f"- {i['requirement']} {i['title']}: {i['status']} — {i['remaining']} (next: {i['next']})")
    lines += ["", "## Human-stage items", ""] + [f"- {i['requirement']} {i['title']}: {i['status']}" for i in human]
    lines += ["", "## Production workflow (for future constructions)", "",
              "declare target and object class -> design opportunities/conditions -> measure the reader emission law and material mix on pilot materials -> fit sources with the emission law (pooled E-step for population models, author E-step for author structure) -> calibrate/transport (direct / global / conditional pilot / refuse) -> serialize a public calibrated bundle -> sealed cold scoring with component uncertainty, support and status -> attach claim level; any psychological reading requires the H-stage contracts.",
              "", "## Remaining triggers", "", "- Human data per H01-H06 contracts.", "- Primary sources unavailable (R03) if the user retrieves them.", "- Native vector signatures need saved source interventions/weights (not available).", "- A certifying end-to-end FP64 derivative check would be a new experiment."]
    (args.output / "final_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: final[k] for k in ("full_scope_planned", "computational_questions_resolved", "scientifically_supported_method_level", "computational_status_counts", "human_status_counts", "plan_closed")}, indent=1))


if __name__ == "__main__":
    main()
