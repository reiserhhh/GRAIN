#!/usr/bin/env python3
"""F17-03 human-stage preparation contracts H01-H06 (no recruitment).

Each contract: required_columns, consent points, analysis estimand, parametric
sample-planning inputs (formulas + inputs, some taken from executed results),
agent-completed preparation (paths), and what remains unvalidated. Sample
numbers are planning inputs, not power claims about unknown human effects.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
R = "results/grain_gptpro_foundation_v3"


def n_proportion(p, half_width, conf=0.95):
    """Persons needed to estimate a proportion within +/- half_width (normal approx)."""
    z = norm.ppf(0.5 + conf / 2)
    return math.ceil(z * z * p * (1 - p) / half_width**2)


def n_correlation(r, alpha=0.05, power=0.8):
    """Fisher-z sample size to detect correlation r (two-sided)."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return math.ceil((z / math.atanh(r)) ** 2 + 3)


def n_two_group(delta, sd, alpha=0.05, power=0.8):
    """Per-group n for a mean difference delta with SD sd."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return math.ceil(2 * (z * sd / delta) ** 2)


def n_paired(d, alpha=0.05, power=0.8):
    """Pairs to detect a standardized within-person change d."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return math.ceil((z / d) ** 2)


def main():
    """Write human_stage_contracts.md and .json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    recovery = json.loads((ROOT / f"{R}/F17/h_parameter_recovery_r1/recovery_curves.json").read_text())
    mdc = next(r for r in csv.DictReader((ROOT / f"{R}/F13/recovery2_r1/minimum_change.csv").open()) if r["seed"] == "20260915" and r["method"] == "noise_predictive")
    reading_sd = 0.5  # F12/F11 declared residual SD per reading (variance .25)
    # slope estimate noise for 32 readings at z in {+-1, +-1/3}: var(b_hat) = sigma^2 / sum (z - zbar)^2
    slope_noise_var = reading_sd**2 / (32 * (1 + 1 / 9) / 2)
    person_slope_var = 0.7  # F12 simulated new-domain slope variance (replace with pilot estimate)
    within_person_slope_diff_sd = math.sqrt(2 * slope_noise_var)
    between_person_slope_sd = math.sqrt(person_slope_var + slope_noise_var)
    contracts = {
        "H01": {
            "title": "same-person intensive longitudinal (30-60 days), conditions A free / B forced topic / C identical material",
            "required_columns": ["person_id (opaque)", "day_index", "occasion_time (ISO8601 with tz)", "condition (A free|B forced topic|C identical material)", "condition_id / z", "material_id / prompt_id", "text", "text_sha256", "event_address", "opportunity_offered flag", "EMA state items", "trait baseline questionnaire", "informant ratings", "behavioural outcomes", "state manipulation arm (if any)", "reader_id", "reader_version", "consent_scope"],
            "consent_points": ["repeated participation across occasions", "storage of raw text and reuse for measurement research", "withdrawal and deletion", "no clinical or selection use of scores"],
            "estimand": "reliability of person condition slopes beta_u versus single-text score reliability (P01 p.21); beta stability across sessions; state-shock vs stable-reactivity separation under state manipulation; cross-occasion coverage of published intervals; fixed/forced/free opportunity effects (F03 rule)",
            "planning_inputs": {"method": "parameter-recovery simulation (P03 p.32), not a conventional power rule",
                                "recovery_curves": f"{R}/F17/h_parameter_recovery_r1/recovery_curves.json#H01_slope_vs_single_reliability",
                                "slope_reliability_by_days_4_readings": {str(x["days"]): round(x["slope_split_half_r"], 3) for x in recovery["H01_slope_vs_single_reliability"]},
                                "days_planned": "30-60 (P03); slope split-half reliability exceeds single-text reliability from ~4 days under declared inputs",
                                "coverage_precision_secondary": {"persons_for_+-.03": n_proportion(0.95, 0.03)},
                                "MDC95_raw_from_F13": float(mdc["mean_mdc95"])},
            "agent_preparation": [f"{R}/F16/serialize_full_r1 (public bundles, cold scorer)", f"{R}/F13/review_r1/F13-05/uncertainty_rules.json", f"{R}/F14/review_r1/F14-04/derived_metrics.csv (register)"],
            "unvalidated": "interval coverage and retest on real people",
        },
        "H02": {
            "title": "behavioural criteria and blind coding",
            "required_columns": ["person_id", "private_belief B (recorded before speaking)", "public language X", "subsequent action A", "manipulated conditions C1-C4 (certainty, evaluation, consequence, time pressure)", "criterion_id", "criterion_value", "criterion_time", "criterion_source (behaviour|other-report|record)", "coder_id", "coder_blind flag", "coding_manual_version", "linked text_ids"],
            "consent_points": ["collection of behavioural/other-report criteria", "linkage of criteria to texts", "coder access limited to blinded materials"],
            "estimand": "P(B, X, A | H_k, C_j): which rival mechanism explains the joint distribution (P04 p.19); claim-ladder L2-L3 evidence; inter-coder agreement",
            "planning_inputs": {"method": "parameter recovery on the F09 process world, not a conventional power rule",
                                "process_identification_with_B_observed": {"1_rep": 0.642, "2_reps": 0.690, "3_reps": 0.727, "corner_reports_only": 0.500},
                                "source": f"{R}/F09/recovery_tables_r3/extra_b_positive_control.csv", "rivals_min": 4, "coders_per_item_min": 2,
                                "conventional_reference_only": {"criterion_r_.2_persons": n_correlation(0.2)}},
            "agent_preparation": [f"{R}/F09/review_r1/F09-06/rival_rules.json", f"{R}/F16/review_r1/F16-06/construction_acceptance.json", f"{R}/F01/mechanism_graph_r1/object_dictionary.csv"],
            "unvalidated": "any external validity",
        },
        "H03": {
            "title": "same-person cross-language with functionally equivalent (non-translated) environments",
            "required_columns": ["person_id", "language", "environment_id (functionally equivalent, not a translation)", "language_order (randomized)", "condition_id", "text", "reader_id (several readers)", "pilot_flag", "environment_equivalence_rationale"],
            "consent_points": ["participation in multiple languages", "use of parallel materials", "cross-language linkage"],
            "estimand": "functional distance between language environments (sup over contexts of the response-law difference; bisimulation-type, P02 p.25) as primary; conditional DIF after pilot calibration only as benchmark (F12 rule)",
            "planning_inputs": {"method": "parameter-recovery simulation of the functional distance", "recovery_curves": f"{R}/F17/h_parameter_recovery_r1/recovery_curves.json#H03_functional_distance_recovery",
                                "null_q95_vs_shift_.3_mean_by_persons": {str(x["persons_per_language"]): round(x["q95_sup_distance"], 3) for x in recovery["H03_functional_distance_recovery"] if x["true_shift"] == 0.0},
                                "persons_per_language_planned": "50-100 under declared inputs", "pilot_pairs_per_condition_min": 15,
                                "benchmark_DIF_paired_persons": n_paired(0.2 / within_person_slope_diff_sd),
                                "input_provenance": "declared simulation inputs; replace with pilot estimates"},
            "agent_preparation": [f"{R}/F12/review_r1/F12-05/transport_rules.json", f"{R}/F12/dif_table_r1/dif_table.csv", "tests/test_grain_gptpro_f12_batch.py"],
            "unvalidated": "cross-language construct identity on real people",
        },
        "H04": {
            "title": "real questioning reactivity and openness cost",
            "required_columns": ["person_id", "question_id", "question_time", "openness_level (5 declared levels, matched theme/exposure)", "response text", "repeat_index", "self-report measure", "behavioural measure", "informant rating", "burden rating", "refusal flag", "pre/post state measure", "probing intensity arm (low|fixed|intensive, randomized)", "policy_id and action log (recorded policy likelihood)"],
            "consent_points": ["being asked adaptive questions", "possible state change from questioning", "right to refuse any question"],
            "estimand": "state displacement caused by asking (A->s edge), burden B(a), information/perturbation ratio per target (F06), and the openness optimum of I(theta; X | C)",
            "planning_inputs": {"method": "G-study parameter recovery", "recovery_curves": f"{R}/F17/h_parameter_recovery_r1/recovery_curves.json#H04_G_coefficient",
                                "G_by_repeats_per_level": {str(x["repeats_per_level"]): round(x["G_person_over_levels"], 3) for x in recovery["H04_G_coefficient"]},
                                "openness_levels": 5, "state_displacement_reference_pairs": n_paired(0.3)},
            "agent_preparation": [f"{R}/F06/review_r1/F06-06/policy_rules.json", f"{R}/F06/policy_sensitivity_r1/conditional_optima.csv", f"{R}/F07/recovery_measured_r1"],
            "unvalidated": "real reactivity, burden and openness costs",
        },
        "H05": {
            "title": "use consequences and fairness",
            "required_columns": ["person_id", "group_id(s) (self-reported, consented)", "use_context", "decision", "score", "support flag", "interval", "outcome"],
            "consent_points": ["collection of group membership", "study of decision consequences", "no automated consequential decisions from research scores"],
            "estimand": "conditional DIF by group surviving global renorm; decision error rates and interval coverage by group",
            "planning_inputs": {"design": "different persons per group (between-group slope contrast)", "slope_DIF_to_detect": 0.2,
                                "between_person_slope_sd": between_person_slope_sd, "persons_per_group": n_two_group(0.2, between_person_slope_sd),
                                "decision_error_half_width": 0.05, "persons_per_group_for_error_rate": n_proportion(0.2, 0.05),
                                "input_provenance": "person slope variance .7 from the F12 simulated new domain; replace with pilot estimates"},
            "agent_preparation": [f"{R}/F12/dif_table_r1/summary.json", f"{R}/F16/serialize_full_r1 (support flags)", f"{R}/F13/review_r1/F13-05/uncertainty_rules.json"],
            "unvalidated": "fairness and consequences on real decisions",
        },
        "H06": {
            "title": "external team rebuild",
            "required_columns": ["team_id", "materials_received (brief, handbook, schema, source data hashes)", "questions_log", "rebuilt_bundle_sha256", "cold_score_max_abs_diff"],
            "consent_points": ["independent team agreement; no access to T1/T2 implementation"],
            "estimand": "cold-score replication (<=1e-10) from brief + handbook + schema + source data only; number and type of clarification questions",
            "planning_inputs": {"independent_teams_min": 2, "targets": 3, "replication_tolerance": 1e-10},
            "agent_preparation": [f"{R}/F16/construction_briefs_r2/construction_briefs.json", "docs/GRAIN_F16_PUBLIC_CALIBRATED_BUNDLE_HANDBOOK_V1.md", "docs/GRAIN_F16_BUNDLE_SCHEMA_V1.json", f"{R}/F16/independent_t3_v1 (agent rebuild, not human)"],
            "unvalidated": "human external rebuild",
        },
    }
    missing = [(h, p.split(" ")[0]) for h, c in contracts.items() for p in c["agent_preparation"] if not (ROOT / p.split(" ")[0]).exists()]
    (args.output / "human_stage_contracts.json").write_text(json.dumps({"contracts": contracts, "missing_preparation_paths": missing, "recruitment": "none", "status": "HUMAN_PENDING"}, indent=1) + "\n")
    lines = ["# Human-stage preparation contracts H01-H06 (F17-03)", "", "No recruitment; no validation is claimed. Sample numbers are planning inputs from declared formulas.", ""]
    for h, c in contracts.items():
        lines += [f"## {h}: {c['title']}", "", f"- Estimand: {c['estimand']}", f"- Required columns: {', '.join(c['required_columns'])}",
                  f"- Consent points: {'; '.join(c['consent_points'])}", f"- Planning inputs: {json.dumps(c['planning_inputs'])}",
                  f"- Agent preparation: {'; '.join(c['agent_preparation'])}", f"- Unvalidated: {c['unvalidated']}", ""]
    lines += ["Computational items are not moved into H rows: every agent-executable gap keeps its F card (requirement ledger)."]
    (args.output / "human_stage_contracts.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({h: c["planning_inputs"] for h, c in contracts.items()}, indent=1), "\nmissing:", missing)


if __name__ == "__main__":
    main()
