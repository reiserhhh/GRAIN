#!/usr/bin/env python3
"""Separate post-failure finite construction, preserving the failed IPF packet."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_ternary_trade as T
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_ternary_meaning_hm180 as U

DEFAULT = ROOT / "results/grain_math_response_v1/ternary_target_trade_r1"
SCIENCE = U.SCIENCE + (
    "configs/grain_ternary_trade_hm180.json",
    "docs/GRAIN_TERNARY_TRADE_PLAN_HM180.md",
    "grain_research/construction/math_ternary_trade.py",
    "scripts/run_grain_ternary_trade_hm180.py",
    "scripts/verify_grain_ternary_trade_hm180.py",
    "tests/test_grain_ternary_trade.py",
)


def prepare(out):
    """Bind failed science unchanged, then use the same frozen event/fold adapter."""
    U.CONFIG = "configs/grain_ternary_trade_hm180.json"
    U.SCIENCE = SCIENCE
    U.T = T
    spec = R.read(ROOT / U.CONFIG)
    failed = ROOT / spec["failed_parent"]
    R.check(failed)
    for name, expected in spec["failed_artifact_hashes"].items():
        if R.digest(failed / name) != expected:
            raise ValueError("Original failed construction provenance changed")
    if "budget exhausted" not in R.read(failed / "compile_failure.json")["error"]:
        raise ValueError("Expected original finite IPF failure was not retained")
    U.prepare(out)


def compile_laws(out):
    """Construct one explicit finite action per law; no iterative entropy solver."""
    R.check(out)
    start, z = time.perf_counter(), R.prepared(out)
    receipts, components = [], None
    for held in (-1, 0, 1):
        used = np.arange(386) if held == -1 else np.flatnonzero(z["folds"] != held)
        b, changes = T.build_with_components(
            z["source_mass"][used], z["source_targets"][used]
        )
        T.validate_bundle(b)
        name = "source_bundle.json" if held == -1 else f"crossfit_bundle_{held}.json"
        R.write(out / name, b)
        receipts.append(b["solver_receipts"])
        if held == -1:
            components = changes
    R.archive(out / "source_trade_components.npz", changes=np.array(components))
    receipt = {
        "method": "finite_support_preserving_cube_action",
        "constructions": 3,
        "view_laws": 6,
        "ipf_solvers": 0,
        "maximum_entropy_claim": False,
        "receipts": receipts,
        "seconds": time.perf_counter() - start,
    }
    R.write(out / "construction_receipt.json", receipt)
    R.seal(
        out,
        "compiled_seal.json",
        (
            "source_bundle.json",
            "crossfit_bundle_0.json",
            "crossfit_bundle_1.json",
            "source_trade_components.npz",
            "construction_receipt.json",
        ),
    )
    print({"view_laws": 6, "ipf_solvers": 0, "seconds": receipt["seconds"]})


def main():
    """Reuse only addressed scoring/analysis, not the failed constructor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "compile", "score", "analyze"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    U.CONFIG, U.T = "configs/grain_ternary_trade_hm180.json", T
    try:
        {
            "prepare": prepare,
            "compile": compile_laws,
            "score": U.score,
            "analyze": U.analyze,
        }[args.stage](args.out)
    except Exception as error:
        if args.out.exists() and not (args.out / f"{args.stage}_failure.json").exists():
            R.write(
                args.out / f"{args.stage}_failure.json",
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "retry": False,
                },
            )
        raise


if __name__ == "__main__":
    main()
