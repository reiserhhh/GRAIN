#!/usr/bin/env python3
"""R33/R14 content coverage of elicitation banks and reader edit-battery coverage (no model calls)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from grain_research.gptpro_foundation_v3.f08_materials import material_bank

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "results/grain_gptpro_foundation_v3"
OUT = V3 / "F14/content_coverage_r1"
TYPES = ("events", "propositions", "subject", "stance", "intensity", "situation",
         "behavioral_tendency", "state_expression", "choice_pattern", "conditional_response")
EDITS = ("delete_evidence", "counterfactual_insert", "pragmatics_only", "ownership", "query", "model", "language")
RELEVANT = {
    "computational_plan_support_targets_T1_T3": ("propositions", "subject", "stance", "intensity", "situation"),
    "psychological_trait_target_example_extraversion": ("behavioral_tendency", "state_expression", "situation",
                                                        "choice_pattern", "conditional_response"),
}


def levels(values):
    return int(pd.Series(values).nunique())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f08 = pd.DataFrame(material_bank())
    f16 = pd.read_csv(V3 / "F16/text_transport_full_r1/materials.csv")
    f16["template"] = f16.material_id.str.rsplit("_", n=2).str[0]
    gen = pd.read_json(ROOT / "results/grain_vector_mechanisms/persona_generation_trace_v1/expanded/generations.jsonl", lines=True)
    r17 = pd.read_csv(V3 / "R17/natural_sensitivity_r1/manifest.csv")
    banks = {
        "F08_bank_96": {"items": len(f08), "coding": {
            "events": ("PRESENT_FIXED", 1), "propositions": ("VARIED", levels(f08.subject_status)),
            "subject": ("VARIED", levels(f08.subject_status)), "stance": ("VARIED", levels(f08.pole)),
            "intensity": ("VARIED", levels(f08.qualification)), "situation": ("VARIED_WRAPPER", levels(f08.family))}},
        "F16_transport_380": {"items": len(f16), "coding": {
            "events": ("PRESENT_FIXED_PER_TEMPLATE", levels(f16.template)), "propositions": ("VARIED", levels(f16.subject_status)),
            "subject": ("VARIED", levels(f16.subject_status)), "stance": ("VARIED", levels(f16.pole)),
            "intensity": ("VARIED", levels(f16.qualification)), "situation": ("VARIED", levels(f16.template))}},
        "persona_generations_270": {"items": len(gen), "coding": {
            "events": ("PRESENT_UNCODED", None), "propositions": ("PRESENT_UNCODED", None),
            "situation": ("VARIED", levels(gen.situation)),
            "behavioral_tendency": ("VARIED_BY_GENERATOR_PROMPT", levels(gen.condition)),
            "state_expression": ("VARIED_BY_GENERATOR_PROMPT", levels(gen.condition)),
            "conditional_response": ("VARIED_GENERATOR_NOT_PERSON", levels(gen.situation) * levels(gen.condition))}},
        "R17_natural_units_100": {"items": len(r17), "coding": {t: ("PRESENT_UNCODED", None) for t in TYPES}},
    }
    rows = []
    for bank, spec in banks.items():
        for t in TYPES:
            status, n = spec["coding"].get(t, ("ABSENT", 0))
            rows.append({"bank": bank, "items": spec["items"], "evidence_type": t, "status": status, "levels": n})
    cov = pd.DataFrame(rows)
    cov.to_csv(OUT / "coverage_matrix.csv", index=False)
    summary = {}
    for bank in banks:
        sub = cov[cov.bank == bank].set_index("evidence_type")
        varied = [t for t in TYPES if sub.loc[t, "status"].startswith("VARIED")]
        for target, rel in RELEVANT.items():
            covered = [t for t in rel if sub.loc[t, "status"].startswith("VARIED")]
            summary[f"{bank}|{target}"] = {
                "comprehensiveness": round(len(covered) / len(rel), 3),
                "relevance": round(len([t for t in varied if t in rel]) / len(varied), 3) if varied else None,
                "varied_types": varied, "relevant_missing": [t for t in rel if t not in covered]}
    battery = [
        {"edit": "delete_evidence", "status": "DONE", "where": "F08 deletion/withdrawal and ownership edits; F16 explanation cards", "paths": "F08/factorial_full_r1; F16/evidence_action_explanation_r1"},
        {"edit": "counterfactual_insert", "status": "ABSENT_EXECUTABLE", "where": "no inserted counterfactual evidence edits on any bank", "paths": ""},
        {"edit": "pragmatics_only", "status": "PARTIAL", "where": "R17 mechanical presentation/register edits on natural text (not pragmatics proper)", "paths": "R17/natural_sensitivity_r1"},
        {"edit": "ownership", "status": "DONE", "where": "F08 and F16 subject status (actual vs quoted vs hypothetical vs negated)", "paths": "F08/factorial_full_r1; F16/text_transport_full_r1"},
        {"edit": "query", "status": "DONE", "where": "F16 pole vs qualification queries; conditional operator E vs N queries", "paths": "F11/reader_concordance_r1; P05/contrast_ledger_r1"},
        {"edit": "model", "status": "DONE", "where": "NLI vs GRAIN (R16, R21) and E5 vs BGE (R17, R21) on the same items", "paths": "F11/reader_concordance_r1; F12/encoder_gauge_r1"},
        {"edit": "language", "status": "ABSENT_HUMAN_STAGE", "where": "same-person parallel materials needed (H03)", "paths": "F17/human_contracts_r3"},
    ]
    for b in battery:
        for pth in [x.strip() for x in b["paths"].split(";") if x.strip()]:
            if not (V3 / pth).exists():
                raise FileNotFoundError(pth)
    pd.DataFrame(battery).to_csv(OUT / "edit_battery_coverage.csv", index=False)
    out = {"code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "banks": {k: v["items"] for k, v in banks.items()},
           "coverage": summary, "edit_battery": {b["edit"]: b["status"] for b in battery},
           "neural_calls": 0, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
