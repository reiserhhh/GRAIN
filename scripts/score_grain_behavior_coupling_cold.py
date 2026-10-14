#!/usr/bin/env python3
"""Numeric-only cold paired-behavior measurement, independent of the simulator."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_behavior_coupling import cold_profile


def main():
    """Read a qualified source bridge and new vector-only paired profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    result = cold_profile(
        json.loads(args.bundle.read_text()), json.loads(args.record.read_text())
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
