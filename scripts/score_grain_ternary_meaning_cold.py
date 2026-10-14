#!/usr/bin/env python3
"""Standalone SOURCE-law scorer: numerical evidence in, coherent targets out."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_ternary_meaning as T


def main():
    """Load one frozen bundle, rejecting outcomes and repeated evidence events."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    b = T.validate_bundle(json.loads(args.bundle.read_text()))
    with args.input.open() as source, args.output.open("x") as sink:
        for line in source:
            sink.write(
                json.dumps(T.cold_record(b, json.loads(line)), allow_nan=False) + "\n"
            )


if __name__ == "__main__":
    main()
