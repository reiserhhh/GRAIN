#!/usr/bin/env python3
"""Standalone numerical behavioral reading; no model or credentials needed."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_behavior_predictive import cold_score


def main():
    """Write once; importing and applying the scorer never invokes the decoder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = json.loads(args.input.read_text())
    result = cold_score(packet)
    with args.output.open("x") as stream:
        json.dump(result, stream, allow_nan=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
