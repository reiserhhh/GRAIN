#!/usr/bin/env python3
"""F16-01 consolidated construction_briefs.json from frozen sources (no new choices).

T1/T2/T3 target blocks are taken verbatim from serialized bundles (the frozen
target contract each bundle binds); T2/T3 construction fields from the frozen
dynamic brief config; T1 world from the T1 cold brief. Hashes of every source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
F16 = ROOT / "results/grain_gptpro_foundation_v3/F16"
SOURCES = {
    "t1_brief": F16 / "t1_cold_full_r1/brief.json",
    "dynamic_briefs": ROOT / "configs/grain_f16_dynamic_briefs_v1.json",
    "t1_bundle": F16 / "serialize_full_r1/bundles/numeric_t1_20260915_identity_global_mixture_T1.json",
}


def main():
    """Write construction_briefs.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dyn = json.loads(SOURCES["dynamic_briefs"].read_text())
    t1b = json.loads(SOURCES["t1_brief"].read_text())
    bundles = sorted(p for p in (F16 / "serialize_full_r1/bundles").glob("*.json") if not p.name.startswith("._"))
    targets = {}
    for p in bundles:
        b = json.loads(p.read_text())
        tid = b["target"]["target_id"]
        if tid not in targets:
            targets[tid] = {"bundle_example": p.name, "target_contract": b["target"]}
            SOURCES[f"bundle_{tid}"] = p
    briefs = {
        "schema": "construction_briefs_v1",
        "status": "CONSOLIDATED_FROM_FROZEN_SOURCES",
        "T1": {"target_id": t1b["target_id"], "world": t1b["world"], "target": t1b["target"], **targets["T1"]},
        "T2": {**dyn["T2"], **targets.get("T2", {})},
        "T3": {**dyn["T3"], **targets.get("T3", {})},
        "shared": {k: dyn[k] for k in ("world", "seeds", "source_authors", "application_authors", "reference_prefix", "application_prefix", "source_bootstraps", "personal_draws")},
        "limits": dyn.get("limits"),
        "rejected": {"GP06": "old wrong-world brief is not a construction brief; T1 is defined on world L, dynamic targets on core (F16-01)"},
        "sources": {k: {"path": str(v.relative_to(ROOT)), "sha256": hashlib.sha256(v.read_bytes()).hexdigest()} for k, v in SOURCES.items()},
    }
    (args.output / "construction_briefs.json").write_text(json.dumps(briefs, indent=1) + "\n")
    print(json.dumps({k: (v.get("target_id") if isinstance(v, dict) else v) for k, v in briefs.items() if k in ("T1", "T2", "T3")}))


if __name__ == "__main__":
    main()
