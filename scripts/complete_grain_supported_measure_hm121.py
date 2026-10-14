#!/usr/bin/env python3
"""Explicit safe-attention adapter; retains the failed full attempt and budget."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_supported_attention_safe as S
from grain_research.vector_mechanisms.navigation import write_json
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts import run_grain_supported_measure_hm121 as P


def prepare(parent):
    """Seal a mathematical-equivalence repair with the actual remaining budget."""
    m = P.check(parent)
    state = H.load(parent / "01/status.json")
    if state["completed_pairs"] != 144 or state["pending"] is not None:
        raise ValueError("The original full failure must be preserved before adapter")
    out = parent / "safe_attention_r1"
    out.mkdir(exist_ok=False)
    for folder in ("00", "01/units", "02"):
        (out / folder).mkdir(parents=True)
    Pairs = H.load(parent / "00/pairs.json")
    H.write(out / "00/pairs.json", Pairs)
    spec = dict(m["spec"])
    spec["max_numeric_seconds"] -= state["seconds"]
    m["spec"] = spec
    m["code"] |= {
        name: H.sha(ROOT / name)
        for name in (
            "grain_research/construction/math_supported_attention_safe.py",
            "tests/test_grain_supported_attention_safe.py",
            "scripts/complete_grain_supported_measure_hm121.py",
        )
    }
    m["inputs"] = H.seals(out, out / "00")
    m["source"] |= {
        str((parent / name).relative_to(ROOT)): H.sha(parent / name)
        for name in (
            "00/manifest.json",
            "01/status.json",
            "01/seal.json",
            "analysis_failure.json",
        )
    }
    m["repair"] = {
        "reason": "Original zero-mass exponent overflow poisoned all readings and actions",
        "same_measure_path_and_four_nodes": True,
        "original_seconds": state["seconds"],
        "remaining_seconds": spec["max_numeric_seconds"],
        "original_budget_reset": False,
    }
    H.write(out / "00/manifest.json", m)
    write_json(
        out / "01/status.json",
        {
            "state": "PREPARED",
            "completed_pairs": 0,
            "pending": None,
            "seconds": 0.0,
            "attempts": 0,
        },
    )
    print("Safe adapter prepared;remaining", spec["max_numeric_seconds"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "analyze"))
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    parent = args.parent.resolve()
    # Dependency injection is explicit and sealed in this independent process.
    P.S = S
    if args.stage == "prepare":
        prepare(parent)
    else:
        out = parent / "safe_attention_r1"
        {"run": P.run, "analyze": P.analyze}[args.stage](out)
        if args.stage == "run":
            H.write(
                out / "numerical_extension_receipt.json",
                {
                    "overflow_witnesses": S.OVERFLOW_WITNESSES,
                    "original_failure_retained": True,
                },
            )
