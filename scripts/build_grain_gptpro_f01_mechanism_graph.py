#!/usr/bin/env python3
"""F01-01 mechanism graph (three graphs) and seven-class object dictionary.

Every node/edge/class carries source addresses into the five frozen sources
(P01-P04 PDF pages, P05 sections) and executed evidence paths from F02-F16.
The builder checks: source pages exist; evidence paths exist; edge labels are
in the declared set; each object class has target, counterexample (with an
existing evidence path) and allowed interpretation; no personality word is used
as a latent-truth node. Passing the checks is bookkeeping, not validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COORD = Path("/Volumes/mobile3/projects/project persona")
SRC = COORD / "reports/gptpro_seminar_20260915/sources"
P05 = COORD / "gptproplans/思路汇总.md"
R = "results/grain_gptpro_foundation_v3"
LABELS = {"observed", "assigned", "latent", "assumed", "computed", "declared", "model_internal"}
FORBIDDEN_TRUTH = ("extraversion", "neuroticism", "conscientiousness", "agreeableness", "openness", "personality", "big5", "big_five")

NODES = [
    # graph, id, meaning, unit, eligibility, sources, evidence
    ("G1_expression", "A_input", "measurement input / situation / probe do(A_t)", "declared condition code", "assigned in designed elicitation; observed-only in natural data", "P01:3;P02:1;P03:12", f"{R}/F06/gaussian_full_summary_audit_r1/target_specific_comparisons.csv"),
    ("G1_expression", "O_opportunity", "expression opportunity offered by the situation", "binary per event", "observed only if the design records it; otherwise unknown (never imputed)", "P03:12;P01:4", f"{R}/F14/metadata_full_r2/natural_support_matrix.csv"),
    ("G1_expression", "theta_u", "stable parameters of a DECLARED response model (no cross-model default truth)", "model-specific parameter units", "latent", "P01:3;P01:15;P03:14", f"{R}/F05/invariants_r1/similarity_invariants.json"),
    ("G1_expression", "s_ut", "current dynamic state", "model state units", "latent", "P01:3;P01:17;P03:14", f"{R}/F05/application_full_r1/paired_comparisons.csv"),
    ("G1_expression", "q_u_selection_policy", "what the person chooses to enter / talk about", "selection probability", "latent (identified only with opportunity denominators)", "P01:4;P01:15", f"{R}/F03/audit_full_r1/same_observation_counterexample.json"),
    ("G1_expression", "pi_u_response_policy", "conditional language response given the chosen situation", "response law", "latent", "P01:4;P01:15;P03:14", f"{R}/F04/application_full_r1/paired_comparisons.csv"),
    ("G1_expression", "G_goal_Aud_L", "communicative goal, audience model, language/culture", "declared or assumed covariates", "assumed unless designed (F09 belief/audience)", "P04:1;P04:2", f"{R}/F09/recovery_tables_r3/extra_b_positive_control.csv"),
    ("G1_expression", "Z_selected_evidence", "evidence actually expressed (selected events)", "event with qualification (actual/negated/quoted/hypothetical)", "observed only when selected", "P03:12", f"{R}/F08/factorial_full_r1/paired_decomposition.csv"),
    ("G1_expression", "X_text", "text event with address", "text + event address", "observed", "P01:3;P03:12;P03:15", f"{R}/F16/event_addresses_r1"),
    ("G1_expression", "B_external", "private belief / subsequent action / external criterion", "task-specific", "observed only if designed; external criteria are human-stage (H02)", "P04:19;P02:1", f"{R}/F09/recovery_tables_r3/extra_b_positive_control.csv"),
    ("G2_reader", "h_sensor", "sensor coordinate phi_m(X) (embedding / GRAIN features); phi is not psychology", "vector coordinates", "computed", "P01:16;P03:12;P03:15", f"{R}/F10/decision_r1/paired_decisions.csv"),
    ("G2_reader", "qual_reading", "reader's ownership/actuality qualification of an event", "probability / hard category", "computed; error law measured on pilot materials", "P01:15;P03:12", f"{R}/F08/mnar_source_r1/emission_laws.json"),
    ("G2_reader", "y_reader", "reader judgment rho_{q,m}(h, C) for query q", "reader score", "computed", "P01:16;P03:12", f"{R}/F11/reader_interface_r1/agreement.csv"),
    ("G2_reader", "llm_state", "LLM internal state under query/intervention (vector path)", "activation coordinates", "model_internal (not human psychology)", "P01:17;P02:1", f"{R}/F10/derivative_triage_r2/diagnostic_posthoc.json"),
    ("G3_construction", "T_target", "declared target (object class, unit, clock, condition plan)", "target units", "declared before scoring", "P01:17;P04:17", f"{R}/F16/construction_briefs_r2/construction_briefs.json"),
    ("G3_construction", "E_environment", "E=(A, nu, mu): allowed inputs, target universe, selection of next input", "design specification", "declared", "P01:15;P01:16", f"{R}/F06/policy_sensitivity_r1/conditional_optima.csv"),
    ("G3_construction", "Psi_functional", "scoring functional Theta^nu = int Psi[P(X|do(A=a))] d nu(a)", "target units", "declared", "P01:17", f"{R}/F16/serialize_audit_r4/summary.json"),
    ("G3_construction", "calibration", "pilot calibration / reader emission law / reference population", "calibration draws", "computed from pilot/source roles only", "P04:17;P01:16", f"{R}/F12/dif_table_r1/summary.json"),
    ("G3_construction", "S_score", "published score with components, support and status", "target units + variance components", "computed (cold, sealed)", "P04:17;P04:18", f"{R}/F13/recovery3_r1/summary.csv"),
    ("G3_construction", "claim_level", "evidence ladder L0-L6 attached to each claim", "ordinal claim level", "declared; L2+ needs external evidence (H02)", "P04:18", f"{R}/F16/review_r1/F16-06/acceptance.json"),
]

EDGES = [
    ("G1_expression", "A_input", "s_ut", "assumed", "measurement can change the measured state (reactivity)", "P01:18", f"{R}/F06/gaussian_full_summary_audit_r1/target_specific_comparisons.csv"),
    ("G1_expression", "A_input", "O_opportunity", "assigned", "the input defines which expressions are possible", "P03:12", f"{R}/F03/recovery_full_r1/exact_population_cells.json"),
    ("G1_expression", "O_opportunity", "Z_selected_evidence", "latent", "selection among opportunities (q_u); unidentified without denominators", "P01:4;P03:12", f"{R}/F03/audit_full_r1/same_observation_counterexample.json"),
    ("G1_expression", "theta_u", "q_u_selection_policy", "latent", "stable parameters enter selection", "P01:4", f"{R}/F03/recovery_full_r1/paired_comparisons.csv"),
    ("G1_expression", "theta_u", "pi_u_response_policy", "latent", "stable parameters enter response", "P01:4;P03:14", f"{R}/F04/application_full_r1/paired_comparisons.csv"),
    ("G1_expression", "s_ut", "pi_u_response_policy", "latent", "state modulates response", "P03:14", f"{R}/F05/application_full_r1/paired_comparisons.csv"),
    ("G1_expression", "G_goal_Aud_L", "X_text", "assumed", "goal/audience/language shape wording (rival explanations)", "P04:1;P04:2;P04:19", f"{R}/F09/recovery_tables_r3/recovery_table.csv"),
    ("G1_expression", "pi_u_response_policy", "X_text", "observed", "response policy generates the text given selected evidence", "P01:3;P03:12", f"{R}/F04/information_full_r1/S_mean_vs_profile_by_seed.csv"),
    ("G1_expression", "X_text", "s_ut", "assumed", "state transition after expression s_{t+1}=F(s_t, A_t, X_t)", "P01:3", f"{R}/F05/joint_full_r1/joint_summary.csv"),
    ("G1_expression", "theta_u", "B_external", "assumed", "external behaviour / criterion (human stage)", "P04:19", f"{R}/F14/review_r1/F14-04/derived_metrics.csv"),
    ("G2_reader", "X_text", "h_sensor", "computed", "sensor coordinates; not psychology", "P01:16", f"{R}/F10/decision_r1/paired_decisions.csv"),
    ("G2_reader", "X_text", "qual_reading", "computed", "qualification reading with measured MNAR error law", "P01:15", f"{R}/F08/end_to_end_r1/decision.md"),
    ("G2_reader", "h_sensor", "y_reader", "computed", "reader functional; agreement is not validity", "P01:16;P01:17", f"{R}/F11/review_r1/F11-05/acceptance.json"),
    ("G2_reader", "llm_state", "y_reader", "model_internal", "attention/activation paths are model mechanisms, never human causal edges", "P01:17;P02:1", f"{R}/F10/review_r1/F10-05/decision.md"),
    ("G3_construction", "T_target", "Psi_functional", "declared", "target function fixed before application scores", "P01:17", f"{R}/F16/construction_briefs_r2/construction_briefs.json"),
    ("G3_construction", "calibration", "S_score", "computed", "calibrated reader/transport enters the score with its uncertainty", "P04:17", f"{R}/F16/bundle_schema_r1/conformance.json"),
    ("G3_construction", "y_reader", "S_score", "computed", "reader outputs scored by the frozen functional", "P01:16", f"{R}/F16/serialize_audit_r4/summary.json"),
    ("G3_construction", "S_score", "claim_level", "declared", "claims carry an evidence level; L4-L5 need interventions/transport, L2+ external evidence", "P04:18", f"{R}/F16/review_r1/F16-06/acceptance.json"),
]

OBJECTS = [
    # class, input, unit, allowed_ops, identifiable_function, target_example, counterexample, counterexample_evidence, allowed_interpretation, interpretation_evidence_needed, standard_precedent, sources
    ("reflective_latent_variable", "multiple indicators of one attribute", "latent scale (model-defined)", "compare persons within the declared model; no cross-model identity", "covariance structure up to gauge", "F11 MTMM traits", "author-constant method variance is observationally identical to extra trait covariance (sum only); a recursive network reproduces the one-factor covariance exactly",
     f"{R}/F11/mtmm_audit_r1/summary.json|{R}/F01/object_counterexamples_r1/object_counterexamples.json", "a common cause of indicators under the stated independence/anchor assumptions", "indicator interventions do not propagate; anchors; L3 discriminant evidence", "factor analysis; MTMM (Campbell-Fiske); measurement invariance", "P03:15;P03:16"),
    ("formative_index", "declared composite of components", "composite units", "sum/weight by declared rule; no latent causal claim", "the composite itself (definitional)", "F16 T1 contrast as a declared functional of cell means", "non-vanishing tetrads reject a single reflective latent while the formative composite stays defined",
     f"{R}/F01/object_counterexamples_r1/object_counterexamples.json", "a use-defined summary for a declared purpose", "explicit composition rule and context of use", "formative measurement (Bollen-Lennox); index construction", "P03:16;P04:17"),
    ("conditional_response_function", "person x condition responses", "response units per condition", "contrasts across declared conditions; interpolation only inside support", "m_u(z) on sampled conditions; slopes need >=2 conditions", "F16 T1 m(+1)-m(-1); F04 slope", "same mean, different slope: mean AUC .50 vs slope AUC 1.0; single condition is rank 1",
     f"{R}/F04/information_full_r1/S_mean_vs_profile_by_seed.csv", "a disposition conditional on a declared situation universe", "condition-crossed designs (F03/F04); transport pilots (F12)", "IRT person x item; CAPS if-then signatures; reaction norms", "P01:2;P02:1;P03:14"),
    ("dynamic_state", "time-ordered observations with elapsed time", "state units (gauge-dependent)", "filtering, forecasting; no naming of arbitrary axes", "observation law up to similarity transforms", "F16 T2 current state; F05", "rotations/scalings of the latent state leave the observation law unchanged; good forecasts coexist with worse inertia recovery (near-root rho +.0036)",
     f"{R}/F05/invariants_r1/similarity_invariants.json|{R}/F05/application_full_r1/paired_comparisons.csv", "a time-varying state within the declared dynamic model", "longitudinal dynamics, elapsed-time designs, shared inputs", "state-space models; Kalman filtering; DSEM", "P01:17;P03:14;P03:15"),
    ("network_interaction", "multiple interacting processes / indicators", "edge weights (model-defined)", "interventional contrasts on nodes; no latent-cause claim", "requires interventions or time order", "none executed as a target in F02-F16", "a recursive network reproduces a one-factor covariance exactly (2.2e-16); only an indicator intervention separates them (.56/.48/.40 vs 0)",
     f"{R}/F01/object_counterexamples_r1/object_counterexamples.json", "organisation among processes, not a hidden common cause", "node interventions, dynamics", "psychological network models; SEM with feedback", "P03:15;P03:16"),
    ("predictive_state", "histories and future tests", "sufficient statistic of futures", "prediction of declared future tests only", "PSR rank / closure on declared tests", "F16 T3 planned four-step mean; F02 C2 order-2 state", "histories with equal one-step law differ in future law (TV 0 then .5); short-history PSR fails closure (.600)",
     f"{R}/F02/recovery_full_r2/future_equivalence.csv|{R}/F15/recovery_r2/F15-01/predictive_state_certificates.json", "the minimal information about declared futures", "future-test designs; no hidden-state ontology claim", "predictive state representations (Littman, Singh); causal states", "P02:2;P03:12;P03:16"),
    ("reader_judgment", "text + query to a reader", "reader score units", "calibrate, transport, compare readers on common materials", "reader emission law on pilot materials", "F08 NLI/lexical readers; F11; F16 text transport", "a strong stance reader fails qualification (+.023 Brier) and, used on the source cohort, collapses the source table (+.12); multi-reader agreement is not validity",
     f"{R}/F08/factorial_full_r1/paired_decomposition.csv|{R}/F08/end_to_end_r1/decision.md", "what this reader outputs for this query on these materials", "measured error law, material mix, model version; external validity is human-stage", "rater models; LLM-as-judge calibration", "P01:16;P01:17;P03:15"),
]


def page_ok(address):
    """Check P01-P04 pages exist in extracts; P05 addresses are section labels S1-S8."""
    pid, page = address.split(":")
    if pid == "P05":
        return re.fullmatch(r"S[1-8]", page) is not None
    text = (SRC / f"{pid}.md").read_text(encoding="utf-8")
    return f"## PDF page {int(page)}\n" in text


def main():
    """Write the graph, dictionary and checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checks, failures = [], []

    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            failures.append(name)

    node_ids = {n[1] for n in NODES}
    for g, nid, meaning, unit, elig, src, ev in NODES:
        check(f"node:{nid}:sources", all(page_ok(a) for a in src.split(";")), src)
        check(f"node:{nid}:evidence_exists", (ROOT / ev).exists(), ev)
        check(f"node:{nid}:no_personality_truth", not any(w in nid.lower() for w in FORBIDDEN_TRUTH))
        check(f"node:{nid}:unit_and_eligibility", bool(unit) and bool(elig))
    for g, a, b, label, meaning, src, ev in EDGES:
        check(f"edge:{a}->{b}:label", label in LABELS, label)
        check(f"edge:{a}->{b}:nodes", a in node_ids and b in node_ids)
        check(f"edge:{a}->{b}:sources", all(page_ok(x) for x in src.split(";")), src)
        check(f"edge:{a}->{b}:evidence_exists", (ROOT / ev).exists(), ev)
    check("model_internal_edges_not_human", all(e[3] == "model_internal" for e in EDGES if "llm" in e[1]))
    for row in OBJECTS:
        cls = row[0]
        check(f"object:{cls}:fields", all(row), cls)
        check(f"object:{cls}:counterexample_evidence", all((ROOT / p).exists() for p in row[7].split("|")), row[7])
        check(f"object:{cls}:sources", all(page_ok(a) for a in row[11].split(";")), row[11])
    check("seven_object_classes", len({o[0] for o in OBJECTS}) == 7)
    graph = {
        "schema": "F01_mechanism_graph_v1",
        "graphs": {
            "G1_expression": "human expression: input/opportunity -> selection -> response -> text; state transition (P01 p3-4, p15; P03 p12-15; P04 p1-2)",
            "G2_reader": "reader inference: text -> sensor -> reader judgment; LLM internal edges are model mechanisms (P01 p16-17; P02 p1; P03 p12, p15)",
            "G3_construction": "measurement construction: target -> environment -> functional -> calibration -> score -> claim level (P01 p15-17; P04 p17-18; P05 core task C)",
        },
        "tasks_P05": {"A": "computational mechanism", "B": "psychological interpretation", "C": "reusable measurement method"},
        "edge_labels": sorted(LABELS),
        "nodes": [dict(zip(("graph", "id", "meaning", "unit", "eligibility", "sources", "evidence"), n, strict=True)) for n in NODES],
        "edges": [dict(zip(("graph", "from", "to", "label", "meaning", "sources", "evidence"), e, strict=True)) for e in EDGES],
        "theta_rule": "theta is defined only inside a declared model; it is never a cross-model default personality truth",
    }
    (args.output / "mechanism_graph.json").write_text(json.dumps(graph, indent=1, ensure_ascii=False) + "\n")
    fields = ("object_class", "input", "unit", "allowed_operations", "identifiable_function", "target_example", "counterexample", "counterexample_evidence", "allowed_interpretation", "interpretation_evidence_needed", "standard_precedent", "sources")
    with (args.output / "object_dictionary.csv").open("x", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        w.writerows(OBJECTS)
    (args.output / "checks.json").write_text(json.dumps({"checks": len(checks), "failed": failures, "rows": checks}, indent=1, ensure_ascii=False) + "\n")
    (args.output / "input_manifest.json").write_text(json.dumps({
        **{p: hashlib.sha256((SRC / p).read_bytes()).hexdigest() for p in ("P01.md", "P02.md", "P03.md", "P04.md")},
        "P05": hashlib.sha256(P05.read_bytes()).hexdigest(), "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=1) + "\n")
    print(json.dumps({"checks": len(checks), "failed": failures}, ensure_ascii=False))


if __name__ == "__main__":
    main()
