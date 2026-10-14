#!/usr/bin/env python3
"""Recheck the two explicitly authorized GP05 caches; no private text is read."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grain_research.gptpro_foundation_v3.f14_cache_interface import check_interface


def main():
    """Export aggregate checks only, preserving failures and exact input hashes."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    base = repo / "results/grain_vector_mechanisms/all_plan_closure_v1"
    results = []
    columns = ["author_index", "unit", "start", "end"]
    for domain in ("pandora", "essays"):
        local = base / "local_bilinear_r2" / domain
        reference = base / "conditional_reference_r2" / domain
        paths = [
            local / "material.json",
            local / "private_units.npz",
            reference / "private_units.npz",
            local / "private_addresses.csv",
            reference / "private_addresses.csv",
        ]
        hashes = {
            str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        }
        with np.load(paths[1], allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in ("g", "h", "semantic", "authors")}
        with np.load(paths[2], allow_pickle=False) as z:
            arrays.update(
                scores=z["scores"].copy(), reference_authors=z["authors"].copy()
            )
            source, held = z["source"].copy(), z["held"].copy()
        addresses = pd.read_csv(paths[3], usecols=columns)[columns].to_numpy()
        ref_addresses = pd.read_csv(paths[4], usecols=columns)[columns].to_numpy()
        material = json.loads(paths[0].read_text())
        try:
            checks = check_interface(
                arrays, addresses, ref_addresses, source, held, int(material["authors"])
            )
            status = "CHECKS_PASSED"
        except ValueError as exc:
            checks = {"failure": str(exc)}
            status = "CHECKS_FAILED"
        results.append(
            {
                "dataset": domain,
                "status": status,
                "checks": checks,
                "inputs": hashes,
                "scope": "existing GP05 cached views only; not original corpus metadata",
                "calendar_time": "absent from these address columns; raw source NOT_INSPECTED",
                "opportunity_denominator": "not present in inspected view",
                "text_or_questionnaire_values_read": False,
                "source_normalization_reconstruction": "NOT_PERFORMED_IN_THIS_PACKET",
            }
        )
    result = {
        "datasets": results,
        "F14_parent_complete": False,
        "remaining": [
            "original metadata locators and availability",
            "X and MEPS authorized schema inventory",
            "source-fixed normalization and complete adapter tests",
            "target support and card integration",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            [{k: r[k] for k in ("dataset", "status", "checks")} for r in results],
            indent=2,
        )
    )
    if any(r["status"] == "CHECKS_FAILED" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
