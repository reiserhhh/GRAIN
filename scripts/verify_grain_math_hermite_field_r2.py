#!/usr/bin/env python3
"""Alternate HM26 arithmetic with explicit conditional-score cache compatibility."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import verify_grain_math_hermite_field as base
from scripts.run_grain_math_hermite_field_r2 import read_json
from scripts.run_grain_math_response import write_json


def main():
    """Verify the r2 recovery without touching the r1 frozen code or cache."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--packet",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/hermite_field_r2",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/hermite_field_verification_r2",
    )
    a = p.parse_args()
    base.read_json = read_json
    try:
        base.verify(a.packet, a.out)
    except BaseException as exc:
        if a.out.exists():
            write_json(a.out / "failure.json", {"error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
