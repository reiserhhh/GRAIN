#!/usr/bin/env python3
"""Score new addressed vector evidence without text, labels or a neural runtime."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_event_anchor_field import cold_record


def main():
    """Read an outcome-free JSON record and the fixed source instrument."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.bundle, allow_pickle=False) as archive:
        bank = {
            k: archive[k]
            for k in ("source_comments", "codes", "anchors", "variance", "directions")
        }
    print(
        json.dumps(
            cold_record(bank, json.loads(args.input.read_text())), allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
