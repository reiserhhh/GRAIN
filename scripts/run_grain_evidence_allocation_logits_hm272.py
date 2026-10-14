#!/usr/bin/env python3
"""Transparent representation recovery of HM272; preserve failed r1 science."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_evidence_allocation_logits as C
from scripts import run_grain_evidence_allocation_hm272 as T
from scripts import run_grain_observed_family_hm170 as R

T.A = C
T.CONFIG = "configs/grain_evidence_allocation_logits_hm272.json"
T.DEFAULT = ROOT / "results/grain_math_response_v1/evidence_allocation_logits_r1"
T.SCIENCE = (
    T.CONFIG,
    "docs/GRAIN_EVIDENCE_ALLOCATION_PLAN_HM272.md",
    "docs/GRAIN_EVIDENCE_ALLOCATION_LOGITS_RECOVERY_HM272.md",
    "grain_research/construction/math_evidence_allocation.py",
    "grain_research/construction/math_evidence_allocation_logits.py",
    "scripts/run_grain_evidence_allocation_hm272.py",
    "scripts/run_grain_evidence_allocation_logits_hm272.py",
    "scripts/diagnose_grain_evidence_allocation_hm272.py",
    "tests/test_grain_evidence_allocation.py",
    "tests/test_grain_evidence_allocation_logits.py",
)


def prepare(out):
    """Include the immutable failed primary and actual support diagnosis in provenance."""
    primary = ROOT / "results/grain_math_response_v1/evidence_allocation_r1"
    if R.read(primary / "runtime.json")["state"] != "FAILED":
        raise ValueError("No recovery over an active/unattempted primary")
    diagnostic = primary / "support_diagnosis"
    ds = R.read(diagnostic / "diagnosed_seal.json")
    for p, h in ds.items():
        if R.digest(diagnostic / p) != h:
            raise ValueError("Support diagnosis changed")
    T.prepare(out)
    manifest = R.read(out / "manifest.json")
    for base, names in (
        (primary, ("manifest.json", "runtime.json")),
        (
            diagnostic,
            (
                "manifest.json",
                "diagnosed_seal.json",
                "bank_masses.npz",
                "diagnosis.json",
            ),
        ),
    ):
        for name in names:
            path = base / name
            manifest["inputs"][str(path.relative_to(ROOT))] = R.digest(path)
    manifest["explicit_harness_binding"] = (
        "T.A=math_evidence_allocation_logits; newCONFIG/SCIENCE/output; immutable harness reused"
    )
    R.write(out / "manifest.json", manifest)
    R.check(out)
    print(
        "HM272-r2 provenance includes failed primary and one native support diagnosis",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "collect", "analyze", "verify", "check")
    )
    parser.add_argument("--out", type=Path, default=T.DEFAULT)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.out)
    elif args.stage == "check":
        R.check(
            args.out,
            ("collected_seal.json", "analyzed_seal.json", "verification_seal.json"),
        )
        print("HM272-r2 seals PASS; primary failure retained")
    else:
        getattr(T, args.stage)(args.out)
