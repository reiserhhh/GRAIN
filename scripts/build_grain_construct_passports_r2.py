#!/usr/bin/env python3
"""Construct passports r2: target-specific T1/T2/T3 and a re-instantiation on a real-text readout."""

from __future__ import annotations

import json
import time
from pathlib import Path

from scripts.build_grain_construct_passports import (
    INDEX,
    ROOT,
    SCHEMA,
    SCHEMA_SHA,
    V3,
    E,
    claim_level,
    contract,
    digest,
    matrices,
    modules,
)

OUT = ROOT / V3 / "F16/passports_r2"
LBR = "results/grain_vector_mechanisms/all_plan_closure_v1/local_bilinear_r2"
SPECIFIC = {
    "T1": {
        "competing_explanations": E("PARTIAL_SIM", "T1 rivals: person condition slope vs condition-specific convention/proficiency vs selection composition across cells; per-person rival model comparison .948 coded vs .743 uncoded register", f"{V3}/F17/reaudit_r2_followup2", f"{V3}/F03/four_way_r1", f"{V3}/F09/rival_person_simulation_r1"),
        "discriminating_intervention": E("PARTIAL_SIM", "T1: factorial cells with coded register; operator-aware design wins on unseen situations", f"{V3}/F17/reaudit_r2_followup2", f"{V3}/F06/head_to_head_r1"),
        "cross_context_invariance": E("PARTIAL_SIM", "T1: qualification intervention signature preserved under a fitted reader map, pole not; conditional DIF survives renorm", f"{V3}/F17/reaudit_r2_followup3", f"{V3}/F12/dif_table_r1"),
    },
    "T2": {
        "competing_explanations": E("PARTIAL_SIM", "T2 rivals: momentary state vs trait carry-over vs MNAR missingness vs reader drift; MNAR does not bias T2 state in F13", f"{V3}/F13/recovery3_r1", f"{V3}/F13/joint_decomposition_r1"),
        "discriminating_intervention": E("ABSENT_HUMAN_STAGE", "T2: randomized state manipulation defined in the H01 contract r4; probes only in simulation", f"{V3}/F17/human_contracts_r4", f"{V3}/F06/measurement_modes_r1"),
        "cross_context_invariance": E("PARTIAL_SIM", "T2: dynamic transport arms in declared worlds", f"{V3}/F16/dynamic_transport_full_r1"),
    },
    "T3": {
        "competing_explanations": E("PARTIAL_SIM", "T3 rivals: persistence vs mean reversion vs MNAR forecast bias (-.064); predictive-state equivalence judged by future prediction", f"{V3}/F13/recovery3_r1", f"{V3}/F15/bisimulation_metric_r1"),
        "discriminating_intervention": E("PARTIAL_SIM", "T3: held-out future steps; extrapolation refused under curvature (coverage .48)", f"{V3}/F04/heldout_rank_r1"),
        "cross_context_invariance": E("PARTIAL_SIM", "T3: dynamic transport arms in declared worlds", f"{V3}/F16/dynamic_transport_full_r1"),
    },
}


def readout_passport(schema):
    m = modules("T1")
    m.update({
        "measurement_object": E("DECLARED", "historical source-selected GRAIN ridge head, Neuroticism output, on PANDORA 36-comment and Essays authors (a supervised readout, not a construct)", LBR),
        "context_of_use": E("DECLARED", "research readout only", LBR),
        "reference_population": E("EVIDENCED_REAL", "PANDORA Big5-only authors and Essays authors; held-split predictions", LBR),
        "elicitation_model": E("DECLARED", "none: naturally occurring comments / one prompted essay", LBR),
        "selection_model": E("NEGATIVE_EVIDENCE", "venue choice carries topic; semantic-head topic share .20 E (GRAIN head none); exposure unobserved", f"{V3}/F10/shortcut_r1", f"{V3}/F17/reaudit_r2_followup1"),
        "representation": E("NEGATIVE_EVIDENCE", "GRAIN coordinates and head move strongly under a source-declared nuisance edit (first/third person; N -.34 SD per Essays chunk); between- and within-person structure differ", f"{V3}/R17/construct_edit_grain_r1", f"{V3}/F17/reaudit_r2_followup1"),
        "competing_explanations": E("ABSENT_EXECUTABLE", "no rival set declared for the readout", LBR),
        "discriminating_intervention": E("ABSENT_HUMAN_STAGE", "no intervention data", f"{V3}/F17/human_contracts_r4"),
        "criterion_evidence": E("NEGATIVE_EVIDENCE", "held-out Pearson with self-reported Neuroticism: PANDORA .048 [-.133,.225], Essays .031 [-.093,.153]", f"{LBR}/pandora/metrics_by_trait.csv", f"{LBR}/essays/metrics_by_trait.csv"),
        "model_invariance": E("NEGATIVE_EVIDENCE", "reader substitution not law-preserving; edit signatures not preserved across encoders", f"{V3}/F11/reader_concordance_r1", f"{V3}/F17/reaudit_r2_followup3"),
        "uncertainty": E("EVIDENCED_REAL", "held-out prediction SD and bootstrap CIs on held metrics", LBR),
        "drift_provenance": E("EVIDENCED_COMPUTATIONAL", "model hashes in local_bilinear_r2 manifests", LBR),
    })
    c = contract("T1", m)
    c["phenomenon"] = E("DECLARED", "none established: the readout is not an admitted phenomenon", f"{V3}/F17/admission_r1")
    c["construct_ontology"] = E("DECLARED", "undeclared (supervised index)", LBR)
    p02, p03 = matrices(m)
    p03["structural_reliability"] = E("NEGATIVE_EVIDENCE", "frozen lexical/SVD default misses the .80 conditional-basis rule (.755/.689)", f"{V3}/F13/structural_multiverse_r1")
    p03["criterion_validity"] = m["criterion_evidence"]
    return m, c, p02, p03


def main():
    if digest(SCHEMA) != SCHEMA_SHA:
        raise ValueError("Schema changed")
    schema = json.loads(SCHEMA.read_text())
    bundles = json.loads(INDEX.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    items = {}
    for t in ("T1", "T2", "T3"):
        m = modules(t)
        m.update(SPECIFIC[t])
        c = contract(t, m)
        p02, p03 = matrices(m)
        items[t] = (m, c, p02, p03, sorted(b["sha256"] for b in bundles if b["target"] == t))
    m, c, p02, p03 = readout_passport(schema)
    items["GRAIN_head_Neuroticism_readout"] = (m, c, p02, p03, [])
    for name, (m, c, p02, p03, bound) in items.items():
        paths = sorted({p for block in (m, c, p02, p03) for v in block.values() for p in v["evidence"]})
        missing = [p for p in paths if not (ROOT / p).exists()]
        if missing:
            raise FileNotFoundError(missing)
        level, blockers = claim_level(schema, m, p03)
        doc = {"schema_version": schema["schema_version"], "schema_sha256": SCHEMA_SHA, "target": name,
               "bound_bundles": {"count": len(bound), "bundle_sha256": bound},
               "passport_modules": m, "construct_contract": c, "evidence_matrix_P02": p02, "validation_matrix_P03": p03,
               "claim_level": level, "claim_blockers": blockers, "evidence_paths_verified": len(paths),
               "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
        (OUT / f"{name}.passport.json").write_text(json.dumps(doc, indent=1) + "\n")
        counts = {}
        for v in {**m, **p03}.values():
            counts[v["status"]] = counts.get(v["status"], 0) + 1
        summary[name] = {"claim_level": level, "status_counts": counts, "first_blocker": blockers}
    t1 = json.loads((OUT / "T1.passport.json").read_text())
    t2 = json.loads((OUT / "T2.passport.json").read_text())
    same = sum(t1["passport_modules"][k] == t2["passport_modules"][k] for k in t1["passport_modules"])
    (OUT / "summary.json").write_text(json.dumps({"summary": summary, "T1_T2_identical_modules": same,
                                                  "code_sha256": digest(Path(__file__))}, indent=1) + "\n")
    print(json.dumps({"summary": summary, "T1_T2_identical_modules": same}, indent=1))


if __name__ == "__main__":
    main()
