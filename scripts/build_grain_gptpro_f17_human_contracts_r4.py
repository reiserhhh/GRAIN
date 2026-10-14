#!/usr/bin/env python3
"""Human-stage contracts r4: r3 plus the columns and sizing named by re-audit r2 (no recruitment)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "results/grain_gptpro_foundation_v3"
R3 = V3 / "F17/human_contracts_r3/human_stage_contracts.json"
OUT = V3 / "F17/human_contracts_r4"


def n_two_arm(d, alpha, power=0.8):
    return int(round(2 * (norm.ppf(1 - alpha / 2) + norm.ppf(power)) ** 2 / d ** 2 + 0.4999))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = json.loads(R3.read_text())
    contracts = c.get("contracts", c)
    rec = json.loads((V3 / "F17/h01_person_n_recovery_r1/person_n_recovery.json").read_text())
    add = {
        "H01": {"columns": ["prompt_family_id", "paraphrase_id (declared paraphrase facet)", "life_event_log (dated, self-reported)",
                            "state_manipulation_arm (randomized: none | mood induction | social-evaluation probe; declared before recruitment)",
                            "coded_register (formal|informal)", "proficiency_measure"],
                "planning": {"persons": 300, "design": "30 days x 4 contexts per day", "source": "F17/h01_person_n_recovery_r1 (power .94 at 300 for slope-criterion r .2; 250-300 recommended)",
                             "recovery_protocol": "docs/GRAIN_H01_PERSON_N_RECOVERY_PROTOCOL_V1.md"}},
        "H02": {"columns": ["self_report Y_q (questionnaire for the rival constructs)", "behaviour Y_b", "text Y_t (link to text_ids)",
                            "coded_register (formal|informal)", "verbal_proficiency_measure", "held_out_intervention_arm (a C-cell withheld from fitting)",
                            "coding_manual_id and version (to be written before data; definitions for each rival M1-M5)"],
                "planning": {"persons": 300, "rule": "triangulation needs all three indicators (F17/admission_r1: dyads are not identified); person N taken from the H01 recovery until an H02-specific recovery is run",
                             "reason_for_register": "F09/rival_person_simulation_r1: uncoded register tied to stakes creates a false stable mechanism parameter"}},
        "H03": {"columns": ["behaviour B (same behavioural outcome in each language environment)", "bilingual_proficiency per language",
                            "shared_fact_material_id (same facts across environments)", "manipulation_factor names (declared, identical across languages)"],
                "planning": {"persons_per_language": "50-100 (h_parameter_recovery_r1)", "DIF_benchmark": "benchmark only; d_C over probe actions is primary",
                             "provenance_note": "h_parameter_recovery_r1 lacked a protocol and decision; decision added post hoc (2026-09-26)"}},
        "H04": {"columns": ["reader_id", "reader_version", "parallel_situation_id (matched theme and exposure across openness levels)"],
                "planning": {"reactivity_arms": "3 randomized arms (low | fixed | intensive probing)",
                             "persons_per_arm_d_.3_two_contrasts_bonferroni": n_two_arm(0.3, 0.025),
                             "persons_total": 3 * n_two_arm(0.3, 0.025),
                             "note": "analytic two-arm contrasts vs low probing with Bonferroni alpha .025; replaces the paired d=.3 figure of 88, which did not match a randomized 3-arm design"}},
    }
    r4 = {}
    for h, v in contracts.items():
        v = dict(v)
        if h in add:
            v["required_columns"] = list(v.get("required_columns", [])) + add[h]["columns"]
            v["planning_inputs_r4"] = add[h]["planning"]
            v["r4_source"] = "re-audit r2 (F17/source_audit_r2)"
        r4[h] = v
    out = {"contracts": r4, "base": str(R3.relative_to(ROOT)), "base_sha256": hashlib.sha256(R3.read_bytes()).hexdigest(),
           "h01_recovery_recommended": rec["recommended_persons"], "recruitment": False,
           "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (OUT / "human_stage_contracts.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    lines = ["# Human-stage preparation contracts r4 (F17-03)", "", "r3 plus re-audit r2 items. No recruitment, no validation claimed.", ""]
    for h, v in r4.items():
        lines += [f"## {h}", f"- Required columns ({len(v.get('required_columns', []))}): {', '.join(v.get('required_columns', []))}"]
        if "planning_inputs_r4" in v:
            lines.append(f"- r4 planning: {json.dumps(v['planning_inputs_r4'], ensure_ascii=False)}")
        lines.append("")
    (OUT / "human_stage_contracts.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({h: len(v.get("required_columns", [])) for h, v in r4.items()}), out["contracts"]["H04"].get("planning_inputs_r4"))


if __name__ == "__main__":
    main()
