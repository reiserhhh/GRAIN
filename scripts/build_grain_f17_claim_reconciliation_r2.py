#!/usr/bin/env python
"""Packet 30 A: union-of-current-gaps supplement to F17/claim_level_reconciliation_r1
(docs/GRAIN_F17_CLAIM_RECONCILIATION_PROTOCOL_V2_SUPPLEMENT.md). Metadata only; r1 is never modified."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3 import revision_stage as rs  # noqa: E402

PROTOCOL = ROOT / "docs/GRAIN_F17_CLAIM_RECONCILIATION_PROTOCOL_V2_SUPPLEMENT.md"
PROTOCOL_SHA = "2f83c29eb8a275d17ea8dc0be87f352bce93fbbbc82b41b8a0282804313c664c"
V3 = ROOT / "results/grain_gptpro_foundation_v3"
R1 = V3 / "F17/claim_level_reconciliation_r1"
OUT = V3 / "F17/claim_level_reconciliation_r2"
LEDGER = Path("/Volumes/mobile3/projects/project persona/reports/claude_research_loop_20260926/ledger/REQUIREMENT_LEDGER_20260926.csv")
RES, CAN, EXE, BLK, MAT, HUM, OPT, SUP, UNA = ("RESOLVED_FINITE_WITH_AUTHORITY", "CANDIDATE_FINITE_PENDING_REVIEW", "EXECUTABLE_CPU_CACHE", "BLOCKED_MODEL_BUDGET",
                                             "MISSING_MATERIAL_OR_METADATA", "HUMAN_VALIDATION", "OPTIONAL_EXTENSION", "SUPERSEDED_HISTORICAL", "UNADJUDICATED")


def G(part, status, roots="", authority="", action="", resource="", reason=""):
    return dict(part=part, status=status, roots=roots, authority=authority, action=action, resource=resource, reason=reason)


# mandatory splits of CURRENT ledger gaps (supervisor 0754); every other row keeps its gap verbatim as UNADJUDICATED
SPLITS = {
 "R12": [G("condition-diversity / informative-context (context-specific or nonlinear) world", EXE, action="declare a context-specific informative-context world and test whether diversity raises target reliability at fixed readings", resource="CPU/declared world", reason="executable theory; not run; packet 28 answered only the scalar linear world"),
         G("real-token reliability curves", UNA, action="inventory cached public-text readouts with repeated authors and conditions; decide whether a real-token curve is computable without new data", resource="CPU/cache or new data (unknown)", reason="resource state unknown; not assumed human"),
         G("real person-level beta/gamma retest; external/informant/EMA validity (H01)", HUM, action="H01 human stage", resource="human participants", reason="source clause requires real persons"),
         G("stale R35 decision-doc metadata", SUP, roots="F00/record_reconciliation_r1", action="none", resource="none", reason="decision documents already exist in record_reconciliation_r1")],
 "R26": [G("finite Gaussian scalar-target reference comparison", RES, roots="F06/recovery_committed_design_r1", authority="SUPERVISOR_CHECK_20260927_0452", action="none", resource="none", reason="accepted finite subclause"),
         G("direct P01 p.24 ALBA / scalar-IRT reference comparison", EXE, action="retrieve the published ALBA / scalar-IRT method and implement it as a reference in the F06 world (not authorized in this packet)", resource="literature retrieval + CPU", reason="source clause open; Gaussian reference is not the published method"),
         G("real recovery/burden parameters and external criterion (H04)", HUM, action="H04 human stage", resource="human participants", reason="source requires real persons")],
 "R13": [G("declared-world crossed estimation of freedom-dependent components", EXE, action="packet 30 B executes this (F07/freedom_components_r1)", resource="CPU/declared world", reason="executable theory gap, not human"),
         G("natural texts at five response-freedom levels", BLK, action="generation (API not authorized) or human collection (H04)", resource="generation budget or human data", reason="material bank lacks the levels")],
 "R17": [G("construct contract for construct-changing edits (P01 p.23)", EXE, action="write the construct contract (doc)", resource="doc", reason="doc work, not human"),
         G("construct-changing edits themselves; LLM and further supervised readers", BLK, action="need generation / forwards", resource="generation + forward budget", reason="not authorized"),
         G("human check of edit neutrality", HUM, action="human raters", resource="human participants", reason="requires human judgement"),
         G("synthetic 1,560-pair precursor (deferred, not approved)", EXE, action="awaits explicit approval; CPU synthetic", resource="CPU/declared world", reason="not approved is not human")],
 "R20": [G("robust REML revision", EXE, action="robust REML for the protocol-form loading world", resource="CPU", reason="executable"),
         G("LLM readers, out-of-reader transfer, version robustness, >2-reader T_C", BLK, action="need forwards", resource="forward budget", reason="not authorized"),
         G("human readers", HUM, action="human readers", resource="human participants", reason="requires humans"),
         G("cross-cutting R35", UNA, action="name the R35 item meant", resource="doc", reason="unspecified")],
 "H05": [G("re-audit of the r3 contract against the sources", EXE, action="doc re-audit of the H05 contract", resource="doc", reason="preparation is doc work, not human"),
         G("real human data", HUM, action="human stage", resource="human participants", reason="requires real persons")],
 "H06": [G("re-audit of the r3 contract against the sources", EXE, action="doc re-audit of the H06 contract", resource="doc", reason="preparation is doc work, not human"),
         G("real human data", HUM, action="human stage", resource="human participants", reason="requires real persons")],
}
# r2 overrides of r1 audit rows (r1 values retained in their own columns)
OVERRIDES = {
 "R21.u1": (UNA, "F12/intervention_transport_r1 already has source-correspondence and whole-author-signature shuffles (delivered); a new cross-construct null needs distinct matched construct inputs, which the E-only panel does not supply", "inventory distinct matched constructs before any new permutation run"),
 "R23.f3": (UNA, "as R21.u1: existing shuffles delivered; a random-map baseline needs distinct construct inputs", "inventory distinct matched constructs first"),
 "R36.u1d": (EXE, "F14/known_structure_r1 is a planted choice-composition positive control, already run; it is NOT an established-construct control and is not repeated", "an established-construct positive control needs distinct material; identify it before any run"),
 "R03.u2": (EXE, "unread open papers are retrieval work; browsing is permitted by the source handoff", "retrieve AERA Standards 2014 and Peters (PNAS Nexus) in a literature packet"),
 "R03.u3": (EXE, "retrieval work, not an access blocker", "retrieve Littman, ALBA and Geiger 2025 full texts in a literature packet"),
}
REPLACE_R1 = {"R17.g", "R20.g", "H05.g", "H06.g"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol changed")
    r1 = list(csv.DictReader((R1 / "clause_reconciliation.csv").open()))
    ledger = list(csv.DictReader(LEDGER.open()))
    reg = rs.load_registry()
    rows = []
    for r in r1:
        o = OVERRIDES.get(r["clause_id"])
        disp = "REPLACED_BY_SPLIT" if r["clause_id"] in REPLACE_R1 else ("OVERRIDDEN_IN_R2" if o else "KEPT")
        rows.append({**r, "source_revision": "r1", "r2_disposition": disp, "effective_status": o[0] if o else r["status"],
                     "r2_reason": o[1] if o else "", "effective_action": o[2] if o else r["remaining_action"]})
    for lr in ledger:
        req = lr["requirement"]; gap = lr["remaining_executable_gap"].split(" | F17-01 unmet:")[0]
        parts = SPLITS.get(req) or [G("whole current gap (verbatim)", UNA, action="map this current gap clause to evidence (not settled within cap)", resource="unknown", reason="union supplement: preserved verbatim")]
        if req not in SPLITS and any(x["requirement"] == req and x["clause_type"] == "current_ledger_gap" for x in r1):
            continue  # r1 already holds this row's current gap with a classified status
        for i, p in enumerate(parts, 1):
            for root in [x for x in p["roots"].split(";") if x]:
                e = reg["revisions"].get(root)
                if e is None or e["kind"] != "CURRENT_REVISION":
                    raise RuntimeError(f"bad root {root}")
            rows.append({"clause_id": f"{req}.cg{i}", "requirement": req, "cards": lr["atomic_tasks"], "sources": lr["sources"], "clause_type": "current_ledger_gap",
                         "origin": "CURRENT_GAP", "part": p["part"], "verbatim": gap, "audit_locator": f"ledger {req} remaining_executable_gap (current, all 50 rows)",
                         "status": p["status"], "evidence_roots": p["roots"], "evidence_pointer": "", "authority": p["authority"],
                         "integrity_basis": ";".join(f"{x}:{rs.evidence_level(reg, x)}" for x in p["roots"].split(";") if x), "pointer_ok": True if p["roots"] else None,
                         "scientific_answer": "", "remaining_action": p["action"], "resource": p["resource"], "reason": p["reason"],
                         "row_science_resolution": lr["science_resolution"], "row_acceptance_decision": lr["acceptance_decision"],
                         "source_revision": "r2", "r2_disposition": "NEW_IN_R2", "effective_status": p["status"], "r2_reason": p["reason"], "effective_action": p["action"]})
    OUT.mkdir(parents=False, exist_ok=False)
    with (OUT / "clause_reconciliation_union.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    eff = [r for r in rows if r["r2_disposition"] != "REPLACED_BY_SPLIT"]
    summ = {"protocol_sha256": PROTOCOL_SHA, "r1_rows": len(r1), "union_rows": len(rows), "effective_rows": len(eff),
            "new_current_gap_rows": sum(r["r2_disposition"] == "NEW_IN_R2" for r in rows), "overrides": list(OVERRIDES), "replaced_r1_rows": sorted(REPLACE_R1),
            "effective_status_counts": dict(Counter(r["effective_status"] for r in eff)), "rows_with_current_gap": len({r["requirement"] for r in eff if r["clause_type"] == "current_ledger_gap"}),
            "ledger_sha256": sha(LEDGER), "r1_csv_sha256": sha(R1 / "clause_reconciliation.csv"),
            "provenance_note": "r1 summary.json counted 218 unregistered depth-2 dirs at build time (before r1 itself was registered); the r1 report's 217 is the count after registration. Not a result change.",
            "withdrawn": ["r1 CANDIDATE_ADJUDICATION section B inference that R12 has a human-only remainder (R12 keeps an executable informative-context clause and an unadjudicated real-token clause)",
                          "r1 proposed package 1 wording 'known-construct positive control' (F14/known_structure_r1 is a planted choice control, already run)",
                          "r1 proposed package 2 as a new finding (existing shuffles delivered; needs distinct construct inputs)"],
            "created": datetime.now(timezone.utc).isoformat()}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
