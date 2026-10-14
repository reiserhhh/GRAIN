#!/usr/bin/env python3
"""F17-02 integrated theory (layered) + contribution ledger.

Each claim: component, statement, formula/assumption, standard precedent,
what this work adds (combination / exact certificate / counterexample /
production rule / none), evidence paths (existence-checked), evidence type,
P04 L0-L6 level for psychological claims (P04 p.18 definitions), promotion
evidence needed. Method properties in declared worlds are kept separate from
psychological meaning; no standard result is renamed as original.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = "results/grain_gptpro_foundation_v3"
LEVELS = {"L0": "Description", "L1": "Reliability", "L2": "External Association", "L3": "Discriminant Construct",
          "L4": "Intervention-Calibrated Construct", "L5": "Transportable Measurement", "L6": "Decision-valid Instrument", "BELOW_L0": "not assessed on human data (no observed stable language pattern for any human construct)", "NA_METHOD": "method property in declared worlds; no psychological construct claim"}
CLAIMS = [
    ("condition", "A contrast target needs at least two conditions per person; one condition is rank 1 at any budget.", "OLS information X'X", "optimal design / IRT person x item", "production rule for open-text elicitation", f"{R}/F04/information_full_r1/information_vs_mc_by_seed.csv", "exact certificate", "NA_METHOD"),
    ("condition", "A condition-blind mean discards conditional information that the same events identify (C1: mean worse than source table; S world: mean AUC .50 vs slope AUC 1.0).", "m_u(z) vs mean_u", "CAPS if-then signatures; IRT", "finite counterexamples in the project's worlds", f"{R}/F02/review_r1/F02-06/construction_rules.json|{R}/F04/information_full_r1/S_mean_vs_profile_by_seed.csv", "finite simulation", "NA_METHOD"),
    ("selection", "Published text without opportunity denominators identifies only the selected-text mean; different response means give identical observed laws.", "P(S0),P(S1,Y0),P(S1,Y1) = [.8,.1,.1] for r in [.1,.9]", "partial identification (Manski)", "exact finite certificate; production rule for text corpora", f"{R}/F03/audit_full_r1/same_observation_counterexample.json", "exact certificate", "NA_METHOD"),
    ("selection", "Under response-dependent selection, more text adds precision but not identification; loss improvement is not selection removal (IPW bias ~.17 at 32 and 512 events).", "IPW with observed propensities", "missing-data theory (MAR/MNAR)", "finite counterexample + rule", f"{R}/F03/review_r1/F03-06/opportunity_rules.json", "finite simulation", "NA_METHOD"),
    ("selection", "Designing opportunities (fixed/common) beats statistical adjustment in all 12 world x prefix cells.", "design vs adjustment", "experimental design", "finite evidence", f"{R}/F03/recovery_full_r1/paired_comparisons.csv", "finite simulation", "NA_METHOD"),
    ("response", "When the transition law is shared, a population second-order table is equivalent to the oracle scorer (+-.005 Brier); person-level adaptation adds noise.", "Markov order-2 table", "Markov chains", "equivalence test in declared world", f"{R}/F02/review_r1/F02-06/derived_metrics.csv", "finite simulation", "NA_METHOD"),
    ("response", "Hierarchical shrinkage (REML/BLUP) is the standard source of gains; the legacy GRAIN estimator's increment appears only under nonlinearity.", "BLUP", "mixed models", "accounting of standard vs GRAIN increments", f"{R}/F04/review_r1/F04-06/derived_metrics.csv", "finite simulation", "NA_METHOD"),
    ("state", "Equal one-step laws can have different future laws; the scoring state must carry what predicts declared futures (order-2 history here; PSR closure fails for short histories, .600).", "PSR / Hankel rank", "predictive state representations", "finite certificates", f"{R}/F02/recovery_full_r2/future_equivalence.csv|{R}/F15/recovery_r2/F15-01/predictive_state_certificates.json", "exact certificate", "NA_METHOD"),
    ("state", "Latent state axes are defined up to similarity transforms; good forecasts do not certify inertia recovery.", "Q -> S Q S', C -> C S^-1", "state-space identifiability", "counterexample in project world", f"{R}/F05/invariants_r1/similarity_invariants.json|{R}/F05/application_full_r1/paired_comparisons.csv", "exact certificate + finite simulation", "NA_METHOD"),
    ("reader", "Reader cost at application is mostly qualification (ownership/actuality), not stance.", "factorial substitution on identical histories", "measurement error / misclassification", "localisation of the failing interface", f"{R}/F08/factorial_full_r1/paired_decomposition.csv", "finite simulation with measured readers", "NA_METHOD"),
    ("reader", "The reader also enters the source fit: dropping unqualified reads under an MNAR reader collapses the source table; the measured emission law with a pooled E-step (population models) and an author-level E-step (author structure) recovers it.", "EM with P(r|y)", "EM for misclassified/MNAR data", "production rule for text-route source fitting", f"{R}/F08/end_to_end_r1/decision.md|{R}/F08/mnar_source_r1/decision.md|{R}/F08/author_estep_r1/decision.md", "finite simulation with measured readers", "NA_METHOD"),
    ("reader", "Reader agreement is not validity; author-constant method variance is identified only as a sum with trait variance unless an independent anchor exists.", "Sigma_f + tau 11'", "MTMM (Campbell-Fiske)", "exact certificate + anchor contrast", f"{R}/F11/review_r1/F11-05/reader_error_rules.json", "exact certificate + finite simulation", "NA_METHOD"),
    ("reader", "Text channels are mixtures over material subtypes; the material mix is a published field (misread 0 vs .625 by subtype).", "channel = sum_status w_status P(cat|b,q,status)", "rater/material facets (G-theory)", "diagnostic on measured reader", f"{R}/F13/material_facet_r1/material_facet.json", "measured reader diagnostic", "NA_METHOD"),
    ("target", "An estimand is fixed; the reader changes the scoring function and its source score-reference, never silently the estimand; public bundles bind target, clock, plan, reader, calibration draws and reference policy.", "bundle contract", "context of use (FDA biomarkers), ECD", "serialized instruments with sealed cold replay", f"{R}/F16/serialize_audit_r4/summary.json|{R}/F16/serialize_reference_shift_r1", "exact replay", "NA_METHOD"),
    ("target", "Transport: direct / global pilot / conditional pilot / refuse; conditional DIF survives the best global renorm.", "affine and conditional inverses", "measurement invariance / DIF", "rules with DIF table and API", f"{R}/F12/review_r1/F12-05/transport_rules.json|{R}/F12/dif_table_r1/summary.json", "finite simulation", "NA_METHOD"),
    ("interpretation", "Process names do not identify processes: corner reports carry 0 nats about process; observing the belief B identifies it (.109 nats).", "mutual information of designs", "rival-model discrimination (Platt strong inference)", "exact computation", f"{R}/F15/recovery_r2/F15-02/metrics.csv", "exact certificate", "NA_METHOD"),
    ("interpretation", "Out-of-family checks must be targeted at declared rivals: the omnibus check has no power (.027) while a targeted exact-calibrated LR detects M4 (.36).", "LR with exact composite calibration", "Neyman-Pearson; model checking", "exact bound + targeted test", f"{R}/F09/targeted_m4_r1/targeted_m4.json|{R}/F09/m4_power_bound_r1/summary.json", "exact computation", "NA_METHOD"),
    ("interpretation", "Reflective latent, network and formative objects can share an observed covariance; indicator interventions or tetrads separate them.", "Sigma = ll' + Psi vs (I-B)^-1 Omega (I-B)^-T", "network psychometrics; formative measurement", "exact counterexamples for the object dictionary", f"{R}/F01/object_counterexamples_r1/object_counterexamples.json", "exact certificate", "NA_METHOD"),
    ("interpretation", "The vector-path action metric never beats the zero reference; it is a better relative geometry than Euclidean only.", "paired MSE vs zero", "model interpretability baselines", "negative result retained", f"{R}/F10/decision_r1/paired_decisions.csv", "finite (model-internal)", "NA_METHOD"),
    ("uncertainty", "Interval components are target-specific (state/personal/source); MNAR enters through personal parameters; source-refit uncertainty is small at 300 source authors.", "law of total variance", "G-theory; bootstrap", "component decomposition on project targets", f"{R}/F13/recovery3_r1/decision.md|{R}/F11/source_uncertainty_r1/decision.md", "finite simulation", "NA_METHOD"),
    ("design", "No single observation policy dominates; policies are target-specific, a state-targeted heuristic is adverse, and prior-only outputs must be flagged.", "expected information gain with costs", "ADO / CAT", "per-target rules and counterexample", f"{R}/F06/review_r1/F06-06/policy_rules.json", "finite simulation", "NA_METHOD"),
    ("psychological", "Construct = identifiable family of context-conditioned response laws (source proposal, P03 p.13); no human construct in this program has reached even L0 (no observed stable pattern on human data was assessed).", "K_u(e|c) = P(E=e|do(C=c),u)", "CAPS; Whole Trait Theory; IRT", "none (source proposal; retained as hypothesis)", f"{R}/F17/human_contracts_r2/human_stage_contracts.json", "none (human pending)", "BELOW_L0"),
]


def main():
    """Write integrated_theory.md, contribution_ledger.csv and checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, missing = [], []
    for i, (comp, stmt, formula, precedent, adds, evidence, etype, level) in enumerate(CLAIMS, 1):
        for p in evidence.split("|"):
            if not (ROOT / p).exists():
                missing.append(p)
        rows.append({"claim_id": f"C{i:02d}", "component": comp, "statement": stmt, "formula_or_assumption": formula, "standard_precedent": precedent,
                     "this_work_adds": adds, "evidence": evidence, "evidence_type": etype, "claim_level": level, "claim_level_definition": LEVELS[level],
                     "promotion_evidence_needed": "human data per H01-H06 contracts" if level != "NA_METHOD" else "none for the method claim; any psychological reading needs H-stage evidence"})
    with (args.output / "contribution_ledger.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["# Integrated theory (F17-02), layered", "",
             "Layer 1 (method properties in declared worlds) is supported by exact certificates and finite simulations. Layer 2 (psychological meaning) is a hypothesis: no human construct here has evidence beyond L0 (P04 p.18 ladder). Standard results are cited by name; what this work adds is stated per claim.", "",
             "Generative chain (P01 p.3-4, P03 p.12): input/opportunity -> selection -> response -> text -> reader -> score, with state transition; see F01/mechanism_graph_r1.", ""]
    for comp in dict.fromkeys(r["component"] for r in rows):
        lines += [f"## {comp}", ""]
        for r in rows:
            if r["component"] == comp:
                lines.append(f"- **{r['claim_id']}** {r['statement']} Formula/assumption: {r['formula_or_assumption']}. Precedent: {r['standard_precedent']}. Adds: {r['this_work_adds']}. Evidence ({r['evidence_type']}): {r['evidence']}. Level: {r['claim_level']} ({r['claim_level_definition']}).")
        lines.append("")
    lines += ["## Not done (F17-01 source audit)", "",
              "- P05 core task A (discover a computationally defined new mechanism and compare it with baselines on the success/failure contrasts) was not executed; every claim above is a property of standard methods in declared worlds, plus production rules.",
              "- Construct Passport (P04 p.17): about 6 of 17 modules exist in bundles; the L0-L6 ladder is attached here only; no score or bundle carries a level.",
              "- R17 real-text nuisance vs construct edit battery across readers (P01 p.23) and A08 Representational Similarity Analysis (P04) are missing; see F17/source_audit_r1.", "",
              "## Not claimed", "", "- No human personality measurement validity (no human construct reached L0).", "- No originality for renamed standard theory; the source's construct proposal (P03 p.13) is retained as a hypothesis.", "- Blocked items: native vector signatures (weights never saved); certifying FP64 derivative check (new experiment)."]
    (args.output / "integrated_theory.md").write_text("\n".join(lines) + "\n")
    (args.output / "checks.json").write_text(json.dumps({"claims": len(rows), "missing_evidence": missing, "levels": sorted({r['claim_level'] for r in rows})}, indent=1) + "\n")
    print(json.dumps({"claims": len(rows), "missing_evidence": missing}))


if __name__ == "__main__":
    main()
