#!/usr/bin/env python3
"""F16-02 bundle_schema.json: derive the declared block/key structure of
F16_public_calibrated_bundle_v1 per route from the serializer and verify every
serialized bundle against it (no undeclared or missing keys)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3 import f16_serialize as pub

BUNDLES = ROOT / "results/grain_gptpro_foundation_v3/F16/serialize_full_r1/bundles"


def keys(obj):
    """Second-level key sets of a bundle's dict blocks."""
    return {k: sorted(v) if isinstance(v, dict) else type(v).__name__ for k, v in obj.items()}


def main():
    """Write the schema and a conformance report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(p for p in BUNDLES.glob("*.json") if not p.name.startswith("._"))
    per_route = defaultdict(list)
    for p in files:
        b = json.loads(p.read_text())
        per_route[b["route"]].append(keys(b))
    schema = {
        "schema_version": pub.VERSION,
        "top_level_required": sorted(pub.TOP_LEVEL),
        "routes": {},
        "notes": "Derived from serializer TOP_LEVEL and all serialized bundles; each route lists the required keys of every dict block (intersection) and keys allowed in some bundles (union minus intersection, e.g. estimator-specific draws).",
    }
    for route, rows in sorted(per_route.items()):
        blocks = {}
        for block in sorted(set().union(*[set(r) for r in rows])):
            vals = [r.get(block) for r in rows]
            if all(isinstance(v, list) for v in vals):
                inter = sorted(set.intersection(*[set(v) for v in vals]))
                union = sorted(set.union(*[set(v) for v in vals]))
                blocks[block] = {"required": inter, "optional": sorted(set(union) - set(inter))}
            else:
                blocks[block] = {"type": sorted({str(v) for v in vals})}
        schema["routes"][route] = {"bundles": len(rows), "blocks": blocks}
    violations = []
    for p in files:
        b = json.loads(p.read_text())
        if set(b) != pub.TOP_LEVEL:
            violations.append({"bundle": p.name, "problem": "top-level keys"})
        spec = schema["routes"][b["route"]]["blocks"]
        for block, value in b.items():
            if isinstance(value, dict):
                allowed = set(spec[block]["required"]) | set(spec[block]["optional"])
                if not set(spec[block]["required"]) <= set(value) or not set(value) <= allowed:
                    violations.append({"bundle": p.name, "problem": f"block {block}"})
    args.schema.write_text(json.dumps(schema, indent=1) + "\n")
    report = {"bundles": len(files), "routes": {r: len(v) for r, v in per_route.items()}, "violations": violations,
              "schema_sha256": hashlib.sha256(args.schema.read_bytes()).hexdigest()}
    args.report.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report | {"violations": len(violations)}, indent=1))


if __name__ == "__main__":
    main()
