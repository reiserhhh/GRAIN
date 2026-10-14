#!/usr/bin/env python3
"""Companion verification using direct tall SVD with the same1e-8 tolerance."""

from __future__ import annotations

import math
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_target_active_support_stable as STABLE
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts import run_grain_math_target_active_support_r2 as COMPANION
from scripts import verify_grain_math_target_active_support as ORIGINAL
from scripts.run_grain_math_response import write_json

DESTINATION = (
    ROOT / "results/grain_math_response_v1/target_active_support_verification_r2"
)


def verify():
    """Verify without new fits/calls, with an alarm for the remaining combined budget."""
    for name, expected in COMPANION.EXPECTED.items():
        if digest(ROOT / name) != expected:
            raise ValueError("frozen primary changed")
    status = read_json(COMPANION.OUT / "run_status.json")
    remaining = 900 - COMPANION.RESERVED_PRIOR_SECONDS - status["analysis_seconds"]
    if remaining <= 0:
        raise TimeoutError("no combined analysis budget remaining")

    def stop(_signal, _frame):
        raise TimeoutError("combined primary/companion verification budget reached")

    signal.signal(signal.SIGALRM, stop)
    signal.alarm(math.floor(remaining))
    try:
        ORIGINAL.qr_svd_basis = STABLE.alternate_basis
        ORIGINAL.verify(COMPANION.OUT, DESTINATION)
    finally:
        signal.alarm(0)
    summary = read_json(DESTINATION / "summary.json")
    summary.update(
        numerical_repair="same Gram-selected ranks, QR/SVD construction, direct tall SVD verification",
        primary_failure=str(
            (
                ROOT
                / "results/grain_math_response_v1/target_active_support_verification_r1/failure.json"
            ).relative_to(ROOT)
        ),
        conservatively_reserved_prior_seconds=COMPANION.RESERVED_PRIOR_SECONDS,
        combined_charged_budget_seconds=summary["combined_analysis_seconds"]
        + COMPANION.RESERVED_PRIOR_SECONDS,
    )
    if summary["combined_charged_budget_seconds"] > 900:
        raise TimeoutError("combined charged budget exceeded")
    write_json(DESTINATION / "summary.json", summary)


if __name__ == "__main__":
    verify()
