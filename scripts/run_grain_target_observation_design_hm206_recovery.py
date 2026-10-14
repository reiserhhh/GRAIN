#!/usr/bin/env python3
"""One boundary-only entropy companion; sealed primary implementation is retained."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_target_observation_design as M
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_target_observation_design_hm206 as S

CONFIG = "configs/grain_target_observation_design_hm206_recovery.json"
DEFAULT = (
    ROOT
    / "results/grain_math_response_v1/target_observation_design_entropy_recovery_r1"
)
EXTRA = (
    CONFIG,
    "docs/GRAIN_HM206_ENTROPY_COMPATIBILITY_RECOVERY_PLAN.md",
    "scripts/run_grain_target_observation_design_hm206_recovery.py",
    "tests/test_grain_target_observation_design_recovery.py",
)


@contextmanager
def entropy_compatibility():
    """Only an ulp-level negative probability complement may represent exact zero."""
    original = M.entropy
    tolerance = 8 * np.finfo(float).eps
    receipt = {
        "repaired_entries": 0,
        "minimum_entry": 0.0,
        "tolerance": float(tolerance),
        "scores_or_gains_clipped": False,
    }

    def entropy(values):
        p = np.asarray(values, float)
        if not np.isfinite(p).all() or np.any(p < -tolerance):
            raise ValueError(
                "Entropy requires finite nonnegative probabilities beyond ulp tolerance"
            )
        negative = p < 0
        receipt["repaired_entries"] += int(negative.sum())
        receipt["minimum_entry"] = min(
            receipt["minimum_entry"], float(p.min(initial=0))
        )
        safe = np.where(negative, 0.0, p)
        return original(safe)

    M.entropy = entropy
    try:
        yield receipt
    finally:
        M.entropy = original


def main():
    """Reuse the primary scientific formulas under a recorded entropy boundary guard."""
    S.CONFIG, S.DEFAULT = CONFIG, DEFAULT
    S.SCIENCE = (*S.SCIENCE, *EXTRA)
    stage = sys.argv[1] if len(sys.argv) > 1 else None
    out = (
        Path(sys.argv[sys.argv.index("--output") + 1])
        if "--output" in sys.argv
        else DEFAULT
    )
    if stage == "prepare":
        R.check(ROOT / R.read(ROOT / CONFIG)["failed_primary"])
    with entropy_compatibility() as receipt:
        S.main()
    if stage == "build":
        # All scores/policies are sealed by S.build before this ancillary receipt.
        t = receipt["tolerance"]
        receipt["single_term_absolute_roundoff_bound"] = float(-t * np.log(t))
        R.write(out / "entropy_compatibility.json", receipt)
        R.seal(out, "entropy_compatibility_seal.json", ("entropy_compatibility.json",))


if __name__ == "__main__":
    main()
