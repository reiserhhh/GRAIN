#!/usr/bin/env python3
"""Restore one frozen cache layout; do not rerun/edit the failed primary."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_anchor_law_layout as M
from scripts import run_grain_anchor_law_disintegration_hm217 as H

PARENT = ROOT / "results/grain_math_response_v1/anchor_law_disintegration_r1"
DEFAULT = ROOT / "results/grain_math_response_v1/anchor_law_layout_recovery_r1"
FILES = (
    "docs/GRAIN_ANCHOR_LAW_LAYOUT_RECOVERY_HM217.md",
    "grain_research/construction/math_anchor_law_layout.py",
    "tests/test_grain_anchor_law_layout.py",
    "scripts/run_grain_anchor_law_layout_hm217.py",
)


def prepare(out):
    """New provenance-bound cache installation, unchanged scientific operation."""
    manifest = H.R.check(PARENT)
    z = H.R.prepared(PARENT)
    z["native_response"] = M.restore(z["native_response"])
    native = H.select(
        ROOT / manifest["config"]["native_parent"] / "observations.npz", ("response",)
    )["response"]
    expected = native[:, :, [0, 6]][:, :, :, 2, :, 2]
    if not np.array_equal(z["native_response"], expected):
        raise ValueError("Layout recovery changed cached model responses")
    inputs = manifest["inputs"] | {
        str((PARENT / f).relative_to(ROOT)): H.R.digest(PARENT / f)
        for f in ("observations.npz", "manifest.json", "prepared_seal.json")
    }
    code = manifest["code"] | {f: H.R.digest(ROOT / f) for f in FILES}
    out.mkdir(parents=True, exist_ok=False)
    H.R.write(
        out / "manifest.json",
        manifest
        | {
            "inputs": inputs,
            "code": code,
            "recovery": "known_response_axis_transpose_only_after_pre_statistic_primary_failure",
        },
    )
    H.R.archive(out / "observations.npz", **z)
    H.R.write(out / "cold_example.json", H.R.read(PARENT / "cold_example.json"))
    H.R.write(
        out / "input_quality.json",
        H.R.read(PARENT / "input_quality.json")
        | {
            "response_shape": list(z["native_response"].shape),
            "literal_parent_response_error": 0,
            "primary_compute_failed_before_statistic": True,
        },
    )
    H.R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "cold_example.json", "input_quality.json"),
    )
    print("PREPARED HM217 layout recovery; parent response error0", flush=True)


def main():
    """Run each compatibility science stage once; retain all primary seals."""
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "compute", "verify", "cold", "check"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    args = p.parse_args()
    if args.stage == "prepare":
        prepare(args.out)
    elif args.stage == "check":
        H.R.check(PARENT)
        H.R.check(
            args.out, ("computed_seal.json", "verification_seal.json", "cold_seal.json")
        )
        print("PRIMARY PREPARED AND RECOVERY COMPLETED SEALS PASS")
    else:
        getattr(H, args.stage)(args.out)


if __name__ == "__main__":
    main()
