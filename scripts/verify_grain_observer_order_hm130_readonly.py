#!/usr/bin/env python3
"""Replay frozen HM130 verification without rewriting its saved verification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_observer_order_hm130 as R


def existing_report_sink(out):
    """Replace only the final report sink with comparison to the existing record.

    The sealed verifier remains unchanged. All reads, hash checks, arithmetic,
    count checks and errors still execute; no new save or overwrite is allowed.
    """
    expected_path = out.resolve() / "verification.json"
    expected = R.H.load(expected_path)
    calls = []

    def compare(path, value):
        """Reject any other write, or a recomputation different from saved proof."""
        if Path(path).resolve() != expected_path or calls:
            raise ValueError("Only one final existing verification comparison allowed")
        if value != expected:
            raise ValueError("Recomputed verification differs from frozen record")
        calls.append(True)

    return compare, calls


def main():
    """Run the original sealed arithmetic with a read-only report adapter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/observer_order_triage_r1",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    sink, calls = existing_report_sink(out)
    with patch.object(R.H, "write", sink):
        R.verify(out)
    if calls != [True]:
        raise ValueError(
            "The original complete verifier did not reach its final report"
        )
    print("HM130 READ-ONLY REPLAY: existing verification identical;no writes")


if __name__ == "__main__":
    main()
