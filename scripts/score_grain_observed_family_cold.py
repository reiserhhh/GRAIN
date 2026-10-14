#!/usr/bin/env python3
"""Score a frozen author observation family from current numeric events only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_observed_family as F


def main():
    """Authenticate a source law and emit once-only cold author scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.bundle.read_bytes()
    if hashlib.sha256(data).hexdigest() != args.sha256:
        raise ValueError("Changed source bundle")
    law = json.loads(data)
    records = [json.loads(line) for line in args.input.read_text().splitlines()]
    with args.output.open("x") as output:
        for rec in records:
            output.write(json.dumps(F.cold_author(law, rec), allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "cold_windows": len(records),
                "neural_forwards": 0,
                "application_fit": False,
            }
        )
    )


if __name__ == "__main__":
    main()
