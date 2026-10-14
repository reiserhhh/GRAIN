#!/usr/bin/env python3
"""Numeric person-bound SOURCE/reader contacts,no model or outcome fitting."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_person_bound_response import cold


def main():
    """Read a typed source/actor record and preserve all character contacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(cold(json.loads(args.input.read_text())), allow_nan=False))


if __name__ == "__main__":
    main()
