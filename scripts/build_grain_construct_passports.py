#!/usr/bin/env python3
"""Instantiate construct passports for the F16 targets, bound to the 162 bundles."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "docs/GRAIN_CONSTRUCT_PASSPORT_SCHEMA_V1.json"
SCHEMA_SHA = "81dde4872bb486283741f6368a5242c4a10490d120c9719ec375dce474be3fba"
V3 = "results/grain_gptpro_foundation_v3"
INDEX = ROOT / V3 / "F16/serialize_full_r1/bundle_index.json"
OUT = ROOT / V3 / "F16/passports_r1"
HANDBOOK = "docs/GRAIN_F16_PUBLIC_CALIBRATED_BUNDLE_HANDBOOK_V1.md"

TARGETS = {
    "T1": {"object": "person condition-response law at fixed clock t=31 (four-cell target function); trait-like conditional parameters of a declared computational world",
           "elicitation": "four declared conditions per author (numeric) / source-role materials read by a changed reader (text)",
           "nu": "the four declared condition cells with equal weight"},
    "T2": {"object": "latent dynamic state at step 47 of a declared state-space world",
           "elicitation": "one record per step 0..47 with z in {-1,1}",
           "nu": "the declared z schedule over 48 steps"},
    "T3": {"object": "one-step forecast of the next observation at step 47 of the same world",
           "elicitation": "one record per step 0..47 with z in {-1,1}",
           "nu": "the declared z schedule over 48 steps"},
}


def E(status, text, *paths):
    return {"status": status, "text": text, "evidence": list(paths)}


def modules(t):
    spec = TARGETS[t]
    return {
        "measurement_object": E("DECLARED", spec["object"] + "; computational target, not a validated psychological construct", HANDBOOK),
        "context_of_use": E("DECLARED", "research instrument calibration only; no decision about any person", HANDBOOK),
        "reference_population": E("EVIDENCED_COMPUTATIONAL", "reference_policy block per route (kind, mu, sigma, scope); never application-derived; text routes FRESH_SOURCE_SCORED", f"{V3}/F16/serialize_full_r1/bundle_index.json", f"{V3}/F16/serialize_reference_shift_r1"),
        "elicitation_model": E("DECLARED", spec["elicitation"], HANDBOOK),
        "selection_model": E("PARTIAL_SIM", "opportunity/selection/response/condition identified only with logged opportunities and coded conditions; MNAR emission enters the likelihood; no real exposure denominators", f"{V3}/F03/four_way_r1", f"{V3}/F08/mnar_source_r1", f"{V3}/F13/recovery3_r1"),
        "representation": E("PARTIAL_SIM", "numeric routes are identity sensors; text routes use NLI/lexical readers; GRAIN and E5/BGE natural-text geometric sensitivity measured (nuisance edits small per unit, systematic at readout)", f"{V3}/F16/text_transport_full_r1", f"{V3}/R17/natural_sensitivity_r1"),
        "competing_explanations": E("PARTIAL_SIM", "rival families and person-level mechanism parameters incl. convention/proficiency rival; names need a breaking observation", f"{V3}/F09/recovery_tables_r3", f"{V3}/F09/rival_person_simulation_r1", f"{V3}/P05/contrast_ledger_r1"),
        "discriminating_intervention": E("PARTIAL_SIM", "operator-aware design beats scalar CAT on unseen situations; factorial with recorded belief identifies M1-M4", f"{V3}/F06/head_to_head_r1", f"{V3}/F09/rival_person_simulation_r1"),
        "convergent_evidence": E("ABSENT_HUMAN_STAGE", "needs independent measures on the same persons (H02)", f"{V3}/F17/human_contracts_r3"),
        "discriminant_evidence": E("ABSENT_HUMAN_STAGE", "non-redundancy with existing traits shown only in simulation; real near-construct data needed (H02)", f"{V3}/F09/rival_person_simulation_r1", f"{V3}/F17/human_contracts_r3"),
        "criterion_evidence": E("ABSENT_HUMAN_STAGE", "no independent behavior or outcome data (H02)", f"{V3}/F17/human_contracts_r3"),
        "response_process": E("PARTIAL_SIM", "reader response process layered (render / application reader / source fit); psychological response process not established", f"{V3}/F08/end_to_end_r1", f"{V3}/F08/author_estep_r1"),
        "cross_context_invariance": E("PARTIAL_SIM", "transport is direct / global pilot / conditional pilot / refuse; conditional DIF survives global renorm; bisimulation metric available for finite laws", f"{V3}/F12/transport_full_r1", f"{V3}/F12/dif_table_r1", f"{V3}/F15/bisimulation_metric_r1"),
        "cross_language_invariance": E("ABSENT_HUMAN_STAGE", "same-person parallel materials needed (H03)", f"{V3}/F12/multilingual_link_r1", f"{V3}/F17/human_contracts_r3"),
        "model_invariance": E("NEGATIVE_EVIDENCE", "reader substitution is not law-preserving: single-law test rejects (qualification), GRAIN pole channel absent; encoders functionally close but not gauge-identical", f"{V3}/F11/reader_concordance_r1", f"{V3}/F12/heldout_map_r1", f"{V3}/F12/encoder_gauge_r1"),
        "uncertainty": E("EVIDENCED_COMPUTATIONAL", "calibration draws and posterior intervals per bundle; source-fit uncertainty small; population-model calibration not certified", f"{V3}/F16/serialize_full_r1", f"{V3}/F11/source_uncertainty_r1"),
        "drift_provenance": E("EVIDENCED_COMPUTATIONAL", "code manifests, bundle hashes, sealed cold replay, reader-hash reconciliation", f"{V3}/F16/serialize_audit_r4", f"{V3}/F16/serialize_os_seal_r1"),
    }


def contract(t, m):
    spec = TARGETS[t]
    return {
        "phenomenon": E("DECLARED", spec["object"], HANDBOOK),
        "construct_ontology": E("DECLARED", {"T1": "disposition-like conditional law", "T2": "dynamic state", "T3": "predictive forecast"}[t], HANDBOOK),
        "reference_universe": m["reference_population"],
        "opportunity": E("PARTIAL_SIM", "declared condition plan; opportunity logged only in simulation", f"{V3}/F03/four_way_r1"),
        "selection_mechanism": m["selection_model"],
        "response_mechanism": m["response_process"],
        "rival_explanations": m["competing_explanations"],
        "discriminating_interventions": m["discriminating_intervention"],
        "reader_class": E("PARTIAL_SIM", "numeric identity sensor; NLI and lexical readers for text routes; a substitute reader must pass the single-law test with no absent channel", f"{V3}/F11/reader_concordance_r1"),
        "transport_domain": m["cross_context_invariance"],
        "external_consequences": m["criterion_evidence"],
        "invalidating_result": E("DECLARED", "a replay mismatch, a failed single-law test for a substituted reader, a coverage failure in a declared world, or any human-stage criterion failure voids the interpretation", HANDBOOK, f"{V3}/F11/reader_concordance_r1"),
    }


def matrices(m):
    rel = E("EVIDENCED_SIM", "slope reliability vs tokens and context diversity; dual report of mean vs individual reliability", f"{V3}/F04/excitation_reliability_r2", f"{V3}/F04/dual_report_r1")
    p02 = {
        "behavior_repeatability": rel,
        "representation_sufficiency": E("PARTIAL_SIM", "held-out maps and contrast tests show what representations keep and drop", f"{V3}/F12/heldout_map_r1", f"{V3}/P05/contrast_ledger_r1"),
        "mechanism_identification": m["competing_explanations"],
        "nuisance_discriminance": m["representation"],
        "construct_sensitivity": m["discriminating_intervention"],
        "external_grounding": m["criterion_evidence"],
        "individual_reliability": E("EVIDENCED_SIM", "rank stability .938; same-person re-identification .142 vs chance .0017", f"{V3}/F04/heldout_rank_r1"),
        "state_sensitivity": E("EVIDENCED_SIM", "state vs trait components recovered jointly", f"{V3}/F13/joint_decomposition_r1", f"{V3}/F01/combined_world_r1"),
        "transport": m["cross_context_invariance"],
        "observer_independence": m["model_invariance"],
        "falsifiability": E("DECLARED", "invalidating results declared; preregistered decisive human tests pending", f"{V3}/F17/human_contracts_r3"),
    }
    p03 = {
        "structural_reliability": E("NEGATIVE_EVIDENCE", "real text: the frozen historical lexical/SVD default misses the declared .80 conditional-basis split-half subspace rule on PANDORA/Essays (.755/.689); per-target structural reliability not run (r2 audit relabel)", f"{V3}/F13/structural_multiverse_r1"),
        "person_reliability": p02["individual_reliability"],
        "dynamic_reliability": E("EVIDENCED_SIM", "state recovery and forecasts in declared worlds", f"{V3}/F13/recovery3_r1"),
        "reader_reliability": m["model_invariance"],
        "context_generalization": E("PARTIAL_SIM", "held-out context prediction under curvature fails (coverage .48): refusal justified", f"{V3}/F04/heldout_rank_r1"),
        "semantic_identifiability": m["competing_explanations"],
        "convergent_validity": m["convergent_evidence"],
        "discriminant_validity": m["discriminant_evidence"],
        "criterion_validity": m["criterion_evidence"],
        "incremental_validity": E("ABSENT_HUMAN_STAGE", "no traditional-scale comparison on real persons", f"{V3}/F17/human_contracts_r3"),
        "intervention_validity": m["discriminating_intervention"],
        "selection_validity": m["selection_model"],
        "trait_state_separation": p02["state_sensitivity"],
        "cross_language_identity": m["cross_language_invariance"],
        "model_version_transport": m["model_invariance"],
        "fairness": E("ABSENT_HUMAN_STAGE", "group data needed (H05)", f"{V3}/F17/human_contracts_r3"),
        "calibration": m["uncertainty"],
        "falsifiability": p02["falsifiability"],
    }
    return p02, p03


def claim_level(schema, m, p03):
    status = {**{k: v["status"] for k, v in m.items()}, **{k: v["status"] for k, v in p03.items()}}
    earned = "BELOW_L0"
    blockers = {}
    for lvl, spec in schema["claim_ladder"]["levels"].items():
        missing = [r for r in spec["requires"] if status.get(r) != "EVIDENCED_REAL"]
        if missing:
            blockers[lvl] = {r: status.get(r) for r in missing}
            break
        earned = lvl
    return earned, blockers


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if digest(SCHEMA) != SCHEMA_SHA:
        raise ValueError("Schema changed")
    schema = json.loads(SCHEMA.read_text())
    bundles = json.loads(INDEX.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for t in TARGETS:
        m = modules(t)
        c = contract(t, m)
        p02, p03 = matrices(m)
        paths = sorted({p for block in (m, c, p02, p03) for v in block.values() for p in v["evidence"]})
        missing = [p for p in paths if not (ROOT / p).exists()]
        if missing:
            raise FileNotFoundError(missing)
        level, blockers = claim_level(schema, m, p03)
        bound = [b for b in bundles if b["target"] == t]
        passport = {
            "schema_version": schema["schema_version"], "schema_sha256": SCHEMA_SHA, "target": t,
            "bound_bundles": {"count": len(bound), "index": str(INDEX.relative_to(ROOT)), "index_sha256": digest(INDEX),
                              "bundle_sha256": sorted(b["sha256"] for b in bound)},
            "passport_modules": m, "construct_contract": c,
            "measurement_tuple": {
                "A_admissible_context_space": TARGETS[t]["elicitation"],
                "nu_target_situation_universe": TARGETS[t]["nu"],
                "mu_observation_policy": "exogenous declared schedule; adaptive policies only in F06 simulations",
                "phi_evidence_extraction": "identity sensor (numeric) or NLI/lexical reader (text), per bundle engine",
                "Psi_scoring_functional": "bundle engine + estimator (target function); calibrated per bundle",
                "V_validity_contract": sorted(k for k, v in {**m, **p03}.items() if v["status"] in ("ABSENT_HUMAN_STAGE", "ABSENT_EXECUTABLE", "NEGATIVE_EVIDENCE")),
            },
            "compiler": {
                "input": {"target_phenomenon": TARGETS[t]["object"], "environment": "declared computational world",
                          "population": "simulated authors under declared priors", "language": "English synthetic materials (text routes)",
                          "intended_interpretation_use": "research calibration only"},
                "output": {"probe_universe": TARGETS[t]["elicitation"], "response_model": "bundle engine",
                           "score_operator": "bundle estimator", "uncertainty": "bundle calibration draws",
                           "validity_envelope": "declared worlds and conditions only; refuse outside",
                           "failure_conditions": c["invalidating_result"]["text"]},
            },
            "evidence_matrix_P02": p02, "validation_matrix_P03": p03,
            "claim_level": level, "claim_blockers": blockers,
            "evidence_paths_verified": len(paths),
            "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        }
        (OUT / f"{t}.passport.json").write_text(json.dumps(passport, indent=1) + "\n")
        counts = {}
        for v in {**m, **p03}.values():
            counts[v["status"]] = counts.get(v["status"], 0) + 1
        summary[t] = {"claim_level": level, "bound_bundles": len(bound), "status_counts": counts,
                      "first_blocker": blockers}
    (OUT / "summary.json").write_text(json.dumps({"schema_sha256": SCHEMA_SHA, "code_sha256": digest(Path(__file__)),
                                                   "targets": summary}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
