#!/usr/bin/env python3
"""Build configs/grain_v3_card_registry.json: every one of the 100 atomic cards
-> canonical artifacts (result dirs, code, tests, docs) and dependencies.
Seeded from the census evidence columns plus explicit additions; paths that do
not exist are dropped and reported, never invented."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = "results/grain_gptpro_foundation_v3"
ATOMIC = Path("/Volumes/mobile3/projects/project persona/configs/grain_gptpro_atomic_v3.json")
CENSUS = Path("/Users/huokean/.claude/jobs/582bb33b/tmp/census_paths.json")

EXTRA = {
    **{f"F00-0{i}": [f"{R}/F00/r1/F00-0{i}"] for i in range(1, 6)},
    "F01-01": [f"{R}/F01/mechanism_graph_r1", f"{R}/F01/object_counterexamples_r1"],
    **{f"F01-0{i}": [f"{R}/F01/recovery_r1/F01-0{i}"] for i in (2, 3, 4)},
    "F01-05": [f"{R}/F01/theory_map_r1"],
    "F02-03": ["grain_research/gptpro_foundation_v3/f02_recovery.py", "grain_research/measurement_discovery/estimators.py"],
    "F02-04": ["tests/test_grain_gptpro_f02.py", "tests/test_grain_gptpro_f02_recovery.py", "tests/test_grain_gptpro_f02_adverse.py"],
    "F02-05": [f"{R}/F02/recovery_full_r2"],
    "F03-03": [f"{R}/F03/recovery_full_r1", f"{R}/F03/audit_full_r1"],
    "F03-04": ["tests/test_grain_gptpro_f03_recovery.py", "tests/test_grain_gptpro_f03_identification.py"],
    "F03-05": [f"{R}/F03/recovery_full_r1", f"{R}/F03/audit_full_r1"],
    "F04-03": ["tests/test_grain_gptpro_f04_recovery.py"],
    "F05-03": [f"{R}/F05/invariants_r1"],
    "F05-06": [f"{R}/F05/invariants_r1"],
    "F07-03": ["tests/test_grain_gptpro_f07_estimation.py"],
    "F08-06": [f"{R}/F08/downstream_full_r2", f"{R}/F16/serialize_full_r1"],
    "F09-04": ["tests/test_grain_gptpro_f09_rivals.py", "tests/test_grain_gptpro_f09_detection.py"],
    "F10-04": [f"{R}/F10/derivative_triage_r1", f"{R}/F10/derivative_triage_r2"],
    "F10-05": [f"{R}/F10/decision_r1", f"{R}/F10/review_r1/F10-05"],
    "F11-03": ["tests/test_grain_gptpro_f11_mtmm.py", "tests/test_grain_gptpro_f11_boundary.py"],
    "F11-04": [f"{R}/F11/source_uncertainty_r1"],
    "F12-03": ["tests/test_grain_gptpro_f12_batch.py", "tests/test_grain_gptpro_f12_dif.py"],
    "F12-04": [f"{R}/F12/dif_table_r1"],
    "F13-01": ["reports/grain_measurement_foundation/GPTPRO_V3_F13_UNCERTAINTY_BUILD_20260916.md"],
    "F13-03": ["tests/test_grain_gptpro_f13_contract.py", "tests/test_grain_gptpro_f13_history_contract.py", "tests/test_grain_gptpro_f13_uncertainty.py"],
    "F13-04": [f"{R}/F13/recovery2_r1", f"{R}/F13/recovery3_r1", f"{R}/F13/material_facet_r1"],
    "F14-02": [f"{R}/F14/metadata_full_r2"],
    "F14-03": [f"{R}/F14/metadata_full_r2", "tests/test_grain_gptpro_f14_role.py"],
    **{f"F15-0{i}": [f"{R}/F15/recovery_r2/F15-0{i}"] for i in range(1, 7)},
    "F16-01": [f"{R}/F16/construction_briefs_r2"],
    "F16-02": ["grain_research/gptpro_foundation_v3/f16_serialize.py", "docs/GRAIN_F16_PUBLIC_CALIBRATED_BUNDLE_HANDBOOK_V1.md", "docs/GRAIN_F16_BUNDLE_SCHEMA_V1.json", f"{R}/F16/bundle_schema_r1"],
    "F16-03": [f"{R}/F16/serialize_audit_r4", "tests/test_grain_gptpro_f16_c2_rejection.py"],
    "F17-01": [f"{R}/F17/source_audit_r1"],
    "F17-02": [f"{R}/F17/integrated_theory_r2"],
    "F17-03": [f"{R}/F17/human_contracts_r2"],
    "F17-04": [f"{R}/F17/numeric_audit_r1"],
    "F17-05": [f"{R}/F17/final_acceptance_r2"],
}


def main():
    """Assemble and write the registry."""
    atomic = json.loads(ATOMIC.read_text())
    census = json.loads(CENSUS.read_text())
    reviews = {p.name: str(p.relative_to(ROOT)) for p in sorted((ROOT / R).glob("F*/review_r*/F*"))}
    registry, dropped = {}, []
    for t in atomic["tasks"]:
        cid = t["id"]
        cand = [f"{R}/{p}" for p in census.get(cid, [])] + EXTRA.get(cid, [])
        if cid in reviews:
            cand.append(reviews[cid])
        keep = []
        for p in dict.fromkeys(cand):
            if (ROOT / p).exists():
                keep.append(p)
            else:
                dropped.append(f"{cid}:{p}")
        registry[cid] = {"package": t["package"], "kind": t["kind"], "requirements": t["requirements"], "depends_on": t["depends_on"],
                         "artifacts": keep, "declared_artifacts": t["artifacts"]}
    out = {"schema": "grain_v3_card_registry_v1", "cards": registry, "dropped_nonexistent": dropped,
           "note": "artifacts are canonical recovery/review locations; COMMON r1 roots hold rejected legacy outputs"}
    (ROOT / "configs/grain_v3_card_registry.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"cards": len(registry), "without_artifacts": [c for c, v in registry.items() if not v["artifacts"]], "dropped": dropped}, indent=1))


if __name__ == "__main__":
    main()
