#!/usr/bin/env python3
"""Repair only the frozen readout binding, without changing collected science."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_evidence_allocation as A
from scripts import run_grain_evidence_allocation_logits_hm272 as W
from scripts import run_grain_observed_family_hm170 as R


def prepare(out):
    """Freeze the explicit original splitter before the companion numerical readout."""
    manifest = R.check(out, ("collected_seal.json",))
    if (out / "analyzed_seal.json").exists():
        raise ValueError("No recomputation of analyzed science")
    R.write(
        out / "readout_recovery_manifest.json",
        {
            "post_collection_readout_not_new_experiment": True,
            "code": {str(Path(__file__).relative_to(ROOT)): R.digest(__file__)},
            "collection_sha256": R.digest(out / "collected_seal.json"),
            "collection_manifest_sha256": R.digest(out / "manifest.json"),
            "binding": "W.T.A=math_evidence_allocation only for analyze/verify; no observe/collect",
            "registered_splitter_sha256": manifest["code"][
                "grain_research/construction/math_evidence_allocation.py"
            ],
            "original_analyze_exit": 1,
            "original_error": "AttributeError: recovered observation module lacks split (pre-freeze Ruff removed unused re-export)",
            "original_verify_exit": 1,
            "original_verify_error": "missing analyzed seal after failed analyze",
            "equation_unchanged": "native = declared + allocation_increment; same 14400 signed contacts/all data/no fitting",
        },
    )
    print("Readout recovery frozen; no numerical results recalculated", flush=True)


def check(out):
    """Verify immutable evidence and the explicit analysis-provider declaration."""
    manifest = R.read(out / "readout_recovery_manifest.json")
    for p, h in manifest["code"].items():
        if R.digest(ROOT / p) != h:
            raise ValueError("Readout companion changed")
    if (
        R.digest(out / "collected_seal.json") != manifest["collection_sha256"]
        or R.digest(out / "manifest.json") != manifest["collection_manifest_sha256"]
    ):
        raise ValueError("Original collection changed during readout recovery")
    if (
        R.digest(ROOT / "grain_research/construction/math_evidence_allocation.py")
        != manifest["registered_splitter_sha256"]
    ):
        raise ValueError("Original declared splitter changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "analyze", "verify", "check"))
    args = parser.parse_args()
    out = W.T.DEFAULT
    if args.stage == "prepare":
        prepare(out)
    else:
        check(out)
        W.T.A = A
        if args.stage == "check":
            R.check(
                out,
                ("collected_seal.json", "analyzed_seal.json", "verification_seal.json"),
            )
            print("All readout/collection seals PASS; no model experiment")
        else:
            getattr(W.T, args.stage)(out)
