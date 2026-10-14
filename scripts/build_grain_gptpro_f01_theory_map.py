#!/usr/bin/env python3
"""F01-05 theory_to_experiment_map.csv + acceptance.json.

Maps every F01-02 primary equation card (PC01-PC17) to concrete F02-F16
results/tests (existence-checked), with a usage boundary and which of the three
targets (reader mechanism / psychological interpretation / production) it
supports; checks every one of the 50 requirement-ledger rows has a destination.
Bibliographic coverage is never used to accept theory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = "results/grain_gptpro_foundation_v3"
LEDGER = Path("/Volumes/mobile3/projects/project persona/reports/claude_research_loop_20260926/ledger/REQUIREMENT_LEDGER_20260926.csv")
CARDS = ROOT / f"{R}/F01/recovery_r1/F01-02/primary_equation_cards.json"

MAP = {
    "PC01": ("predictive state representation", "production", f"{R}/F15/recovery_r2/F15-01/predictive_state_certificates.json|{R}/F02/recovery_full_r2/future_equivalence.csv", "tests/test_grain_gptpro_f15_theory.py", "state = sufficient statistic for declared future tests; no hidden-state ontology"),
    "PC02": ("PSR original (Littman et al.)", "production", f"{R}/F15/recovery_r2/F15-01/predictive_state_certificates.json", "tests/test_grain_gptpro_f15_theory.py", "primary text not retrieved this pass (R03); used via PC01"),
    "PC03": ("bisimulation", "production", f"{R}/F15/recovery_r2/F15-05/method_adoption_final.csv", "tests/test_grain_gptpro_f15_theory.py", "exact equivalence only; metric not computed"),
    "PC04": ("structural vs practical identifiability", "psychological", f"{R}/F05/invariants_r1/similarity_invariants.json|{R}/F01/recovery_r1/F01-04/identification_contract.md", "tests/test_grain_gptpro_f05_invariants.py", "local rank is not global uniqueness; practical identification is finite-sample"),
    "PC05": ("nonlinear ICA variability condition", "psychological", f"{R}/F15/recovery_r2/F15-03/auxiliary_identification.json", "tests/test_grain_gptpro_f15_theory.py", "rank condition checked on declared Gaussian models; learned iVAE not executed"),
    "PC06": ("iVAE identifiability", "psychological", f"{R}/F15/recovery_r2/F15-03/auxiliary_identification.json", "tests/test_grain_gptpro_f15_theory.py", "condition (iv) checked; four conditions give rank 4/3/0 as declared"),
    "PC07": ("unsupervised disentanglement impossibility", "reader", f"{R}/F15/recovery_r2/F15-04/metrics.csv|{R}/F05/invariants_r1/similarity_invariants.json", "tests/test_grain_gptpro_f15_theory.py", "stable latent coordinates carry no unique psychological meaning without auxiliary structure"),
    "PC08": ("information bottleneck", "production", f"{R}/F15/recovery_r2/F15-02/metrics.csv", "tests/test_grain_gptpro_f15_theory.py", "exact MI of given compressions; IB curve not optimised"),
    "PC09": ("Blackwell sufficiency / garbling", "production", f"{R}/F15/recovery_r2/F15-02/compression_order.json", "tests/test_grain_gptpro_f15_theory.py", "explicit kernels and Farkas certificates on finite laws"),
    "PC10": ("Mori-Zwanzig memory", "production", f"{R}/F15/recovery_r2/F15-01/metrics.csv", "tests/test_grain_gptpro_f15_theory.py", "finite discrete-time analog; memory variance .09"),
    "PC11": ("Bayesian experimental design", "production", f"{R}/F06/family_full_r2/paired_comparisons.csv|{R}/F06/irt_full_r1/paired_comparisons.csv", "tests/test_grain_gptpro_f06_policy_state.py", "per-target utility; no single policy dominates; reactivity cost is H04"),
    "PC12": ("ALBA language-based adaptive assessment", "production", f"{R}/F06/review_r1/F06-06/policy_rules.json", "tests/test_grain_gptpro_f06_policy_state.py", "not executed: needs a language-derived item bank and humans (H04)"),
    "PC13": ("rational speech acts / pragmatics", "psychological", f"{R}/F08/factorial_full_r1/paired_decomposition.csv", "tests/test_grain_gptpro_f08_materials.py", "RSA listener not executed; qualification law measured directly instead"),
    "PC14": ("counterfactual explanations", "reader", f"{R}/F16/serialize_audit_r4/summary.json", "tests/test_grain_gptpro_f16_serialize.py", "exact withdrawal effects of observed evidence, not searched counterfactual texts"),
    "PC15": ("partial identification (Manski)", "psychological", f"{R}/F03/audit_full_r1/same_observation_counterexample.json|{R}/F03/recovery_full_r1/exact_population_cells.json", "tests/test_grain_gptpro_f03_identification.py", "primary quotation unavailable; exact finite certificate used"),
    "PC16": ("MTMM / G-theory / measurement invariance", "psychological", f"{R}/F11/mtmm_audit_r1/summary.json|{R}/F12/dif_table_r1/summary.json|{R}/F13/recovery2_r1/variance_components.csv", "tests/test_grain_gptpro_f11_anchor.py", "primary definitions unavailable this pass; finite certificates carry the load"),
    "PC17": ("construct validity / validation argument", "psychological", f"{R}/F16/review_r1/F16-06/construction_acceptance.json|{R}/F01/mechanism_graph_r1/object_dictionary.csv", "tests/test_grain_gptpro_f16_serialize.py", "claims carry L0-L6 levels; external validity is human-stage (H02)"),
}


def main():
    """Write map, requirement destination check and acceptance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cards = json.loads(CARDS.read_text())
    rows, failures = [], []
    for c in cards:
        cid = c["id"]
        if cid not in MAP:
            failures.append(f"unmapped {cid}")
            continue
        family, target, evidence, test, boundary = MAP[cid]
        ok_ev = all((ROOT / p).exists() for p in evidence.split("|"))
        ok_test = (ROOT / test).exists()
        if not (ok_ev and ok_test):
            failures.append(f"missing evidence/test {cid}")
        rows.append({"card": cid, "family": family, "locator": c.get("locator", ""), "cited_by": c.get("cited_by", ""),
                     "target_supported": target, "experiment_evidence": evidence, "test": test, "usage_boundary": boundary,
                     "evidence_exists": ok_ev and ok_test})
    with (args.output / "theory_to_experiment_map.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ledger = list(csv.DictReader(LEDGER.open()))
    no_dest = [r["requirement"] for r in ledger if not (r["next_action"].strip() or r["acceptance_decision"] == "PROPOSED_FINITE_CLOSED")]
    targets = {t: sum(r["target_supported"] == t for r in rows) for t in ("reader", "psychological", "production")}
    acceptance = {
        "card": "F01-05",
        "primary_cards_mapped": len(rows),
        "failures": failures,
        "requirements": len(ledger),
        "requirements_without_destination": no_dest,
        "targets_supported": targets,
        "statuses": {"implementation": "DONE", "protocol": "DONE" if not failures and not no_dest else "PARTIAL", "science": "PARTIAL_SOURCES_UNAVAILABLE"},
        "note": "bibliographic coverage does not accept theory; PC02/PC15/PC16/PC17 primaries partly unavailable (R03)",
        "parent_card_promoted": False,
    }
    (args.output / "acceptance.json").write_text(json.dumps(acceptance, indent=1) + "\n")
    print(json.dumps(acceptance, indent=1))


if __name__ == "__main__":
    main()
