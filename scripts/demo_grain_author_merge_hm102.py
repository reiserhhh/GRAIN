#!/usr/bin/env python3
"""Demonstrate reusable author composition on every fixed HM102 synthetic profile."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_author_evidence_merge import author_laws
from scripts.compose_grain_author_evidence import score_profiles
from scripts.run_grain_author_evidence_merge_hm102 import (
    check,
    load,
    seals,
    sha,
    verify_hashes,
    write,
)


def main():
    """Exercise the public interface without fitting or reading text/personality labels."""
    out = ROOT / load(ROOT / "configs/grain_author_merge_hm102.json")["output"]
    check(out)
    verify_hashes(out, load(out / "output_seal.json"))
    if load(out / "verification.json")["status"] != "VERIFIED_FINITE_HM102_NOT_GLOBAL":
        raise ValueError("Verified primary mechanism required")
    reader = load(out / "00/reader.json")
    with np.load(out / "00/carriers.npz", allow_pickle=False) as z:
        indices, counts = z["indices"], np.diff(z["offsets"])
    profiles = []
    for bank, bridge, family, arm in itertools.product(
        range(2), range(2), range(6), range(4)
    ):
        ids = np.flatnonzero(
            np.all(indices[:, [0, 1, 2, 5]] == [bank, bridge, family, arm], axis=1)
        )
        ids = ids[np.lexsort((indices[ids, 4], indices[ids, 3]))]
        for label, weights in zip(("plus", "minus"), author_laws(), strict=True):
            profiles.append(
                {
                    "name": f"synthetic_{bank}_{bridge}_{family}_{arm}_{label}",
                    "record_indices": ids.tolist(),
                    "weights": weights.tolist(),
                }
            )
    declaration = {
        "targets": reader["targets"],
        "profiles": profiles,
        "atom_counts": counts.tolist(),
        "atom_convention": "unit_mean",
        "partition_scale_contract": "Frozen HM101 queries, registered per-record positions and stored K/V; scale-aware alternative estimand, not gauge-invariant author measurement",
    }
    dest = out / "03_interface"
    dest.mkdir()
    write(dest / "declaration.json", declaration)
    with np.load(out / "01_cold.npz", allow_pickle=False) as observations:
        expectation = score_profiles(observations, declaration, "declared_expectation")
        partition = score_profiles(observations, declaration, "calibrated_partition")
    write(dest / "declared_scores.json", expectation)
    write(dest / "partition_scores.json", partition)
    # Reconcile the public profile interface against the frozen primary table.
    import csv

    expected_lookup = {
        r["profile"]: r["score_by_target"] for r in expectation["scores"]
    }
    merge_lookup = {r["profile"]: r["score_by_target"] for r in partition["scores"]}
    errors = []
    with (out / "02/author_scores.csv").open() as stream:
        for r in csv.DictReader(stream):
            if r["atom"] != "unit_mean" or r["gauge"] != "base":
                continue
            base = f"synthetic_{r['bank']}_{r['bridge']}_{r['family']}_{r['arm']}_"
            for lookup, metric in (
                (expected_lookup, "expected_contrast"),
                (merge_lookup, "partition_merge_contrast"),
            ):
                errors.append(
                    abs(
                        lookup[base + "plus"][r["target"]]
                        - lookup[base + "minus"][r["target"]]
                        - float(r[metric])
                    )
                )
    if max(errors) > 1e-8:
        raise ValueError("Public author profile interface differs from primary law")
    report = {
        "status": "VERIFIED_FINITE_INTERFACE_HM102",
        "synthetic_profiles": len(profiles),
        "contrast_checks": len(errors),
        "max_error": max(errors),
        "measurement_modes": ["declared_expectation", "calibrated_partition"],
        "original_scope_complete": False,
    }
    write(dest / "verification.json", report)
    write(
        out / "interface_seal.json",
        {
            "code": {
                n: sha(ROOT / n)
                for n in (
                    "scripts/compose_grain_author_evidence.py",
                    "scripts/demo_grain_author_merge_hm102.py",
                    "tests/test_grain_author_evidence_interface.py",
                )
            },
            "outputs": seals(out, dest),
        },
    )
    print("INTERFACE VERIFIED", report)


if __name__ == "__main__":
    main()
