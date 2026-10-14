#!/usr/bin/env python
"""Hash-only input freeze for the R34 cached error-budget reconciliation (packet20 A).
Reads no scores; writes docs/GRAIN_R34_ERROR_BUDGET_INPUTS_V1.json (x-mode, never overwrites)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
F16 = "results/grain_gptpro_foundation_v3/F16"
SEEDS = ("20260915", "20260916", "20260917")
ARMS6 = ("hybrid_joint", "nli_joint", "hybrid_factorized", "oracle", "naive_gaussian", "eligible_continuous")
MODULES = ("f16_t1_text.py", "f16_dynamic_text.py", "f16_t1_source.py", "f16_independent_t3.py", "f16_serialize.py")


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def files():
    out = {"canonical_historical": [], "canonical_serialized": [], "alias_check": [], "replay": [], "modules": []}
    for s in SEEDS:
        out["canonical_historical"] += [f"{F16}/t1_source_full_r1/{s}/cold_scores.jsonl", f"{F16}/t2_cold_full_r1/{s}/cold_scores.jsonl",
                                        f"{F16}/t3_application_full_r1/{s}/cold_scores.jsonl", f"{F16}/t3_application_full_r1/{s}/bundle.json",
                                        f"{F16}/t1_cold_full_r1/{s}/scores.jsonl", f"{F16}/t1_text_full_r1/{s}/arm_scores.json"]
        out["canonical_historical"] += [f"{F16}/dynamic_text_full_r1/{s}/{a}_scores.json" for a in ARMS6[:4]]
        out["canonical_historical"] += [f"{F16}/dynamic_text_gaussian_r1/{s}/{a}_scores.json" for a in ARMS6[4:]]
        out["alias_check"] += [f"{F16}/t1_text_full_r1/{s}/application_cold_scores.jsonl", f"{F16}/dynamic_text_full_r1/{s}/application_cold_scores.jsonl"]
        out["alias_check"] += [f"{F16}/dynamic_text_full_r1/{s}/{a}_scores.json" for a in ARMS6[4:]]
        out["alias_check"] += [f"{F16}/text_transport_full_r1/scores/{s}/{k}_changed_pilot_{e}.json" for k in ("T1", "dynamic") for e in ("mixture", "plugin")]
        out["alias_check"] += [f"{F16}/t1_transport_full_r1/{s}/application_scores_before_truth.json"]
        out["alias_check"] += [f"{F16}/public_contract_full_r1/numeric_T{k}_{s}/cold.jsonl" for k in (1, 2, 3)]
    idx = f"{F16}/serialize_full_r1/bundle_index.json"
    out["canonical_serialized"].append(idx)
    for b in json.loads((ROOT / idx).read_text()):
        out["canonical_serialized"].append(f"{F16}/serialize_full_r1/cold/{b['name']}.scores.jsonl")
    out["replay"] += [f"{F16}/t1_text_full_r1/20260915/bundle.json", f"{F16}/t1_text_full_r1/20260915/application_inputs.jsonl",
                      f"{F16}/dynamic_text_full_r1/20260915/bundle.json", f"{F16}/dynamic_text_full_r1/20260915/application_inputs.jsonl"]
    out["modules"] += [f"grain_research/gptpro_foundation_v3/{m}" for m in MODULES]
    return out


def main():
    groups = files()
    manifest = {"purpose": "packet20 A input freeze (hash only)", "groups": {g: {p: sha(p) for p in ps} for g, ps in groups.items()}}
    manifest["counts"] = {g: len(v) for g, v in manifest["groups"].items()}
    dest = ROOT / "docs/GRAIN_R34_ERROR_BUDGET_INPUTS_V1.json"
    with dest.open("x") as fh:
        fh.write(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(manifest["counts"], hashlib.sha256(dest.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
