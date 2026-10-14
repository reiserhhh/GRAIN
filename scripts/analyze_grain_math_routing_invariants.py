#!/usr/bin/env python3
"""HM46A: distinguish attenuation from reorganization without a precision sweep."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction.math_routing_invariants import restrictions
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_construct_routing import OUT, write

CODE = (
    "grain_research/construction/math_routing_invariants.py",
    "scripts/analyze_grain_math_routing_invariants.py",
    "tests/test_grain_math_routing_invariants.py",
    "docs/GRAIN_CONDITIONAL_ROUTING_INVARIANTS_V1.md",
)


def prepare(out):
    """Seal the additional analysis before reading any HM46 response outputs."""
    if out.exists():
        raise FileExistsError("Preserve existing companion")
    out.mkdir(parents=True)
    files = {}
    for name in CODE:
        p = out / "code" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT / name).read_bytes())
        files[name] = digest(p)
    write(
        out / "manifest.json",
        {
            "packet": "HM46A",
            "code": files,
            "new_model_calls": 0,
            "scope": "analysis declared during HM46 collection, before output inspection; not an independent preregistration",
        },
    )


def analyze(parent, out):
    """Compute all three conditions against self/actual with every form retained."""
    manifest = read_json(out / "manifest.json")
    for name, sha in manifest["code"].items():
        if digest(ROOT / name) != sha or digest(out / "code" / name) != sha:
            raise ValueError("Changed companion code")
    verified = read_json(parent / "verification.json")
    if verified["status"] != "VERIFIED_FINITE_HM46_NOT_GLOBAL":
        raise ValueError("Require actual HM46 evidence before analyzing it")
    config = read_json(parent / "00/manifest.json")["spec"]
    targets = config["targets"]
    with (parent / "03/response_matrices.csv").open() as stream:
        records = list(csv.DictReader(stream))
    matrices = {}
    for r in records:
        key = (r["role"], int(r["wording"]), int(r["other"]), int(r["imagined"]))
        matrices.setdefault(key, np.full((6, 6), np.nan))[
            targets.index(r["content"]), targets.index(r["query"])
        ] = float(r["polarity_delta"])
    results = []
    for role, wording in itertools.product(("source", "application"), (0, 1, "mean")):

        def matrix(o, a, role=role, wording=wording):
            return (
                np.mean([matrices[role, w, o, a] for w in (0, 1)], axis=0)
                if wording == "mean"
                else matrices[role, wording, o, a]
            )

        for other, imagined in ((1, 0), (0, 1), (1, 1)):
            results.append(
                {
                    "role": role,
                    "wording": wording,
                    "other": other,
                    "imagined": imagined,
                    **restrictions(matrix(0, 0), matrix(other, imagined)),
                }
            )
    write(
        out / "results.json",
        {
            "status": "FINITE_CONDITIONAL_ROUTING_INVARIANTS",
            "conditions": results,
            "parent_matrix_sha256": digest(parent / "03/response_matrices.csv"),
            "no_statistical_p_values": True,
            "no_psychological_factor_claim": True,
        },
    )
    print({"conditions": len(results), "new_model_calls": 0})


def main():
    """Explicit prepare/analyze stages, with no tuning or recursive worker launch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "analyze"))
    parser.add_argument("--parent", type=Path, default=OUT)
    parser.add_argument(
        "--out", type=Path, default=OUT.parent / "construct_routing_invariants_r1"
    )
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.out)
    else:
        analyze(args.parent, args.out)


if __name__ == "__main__":
    main()
