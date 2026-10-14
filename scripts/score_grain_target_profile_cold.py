#!/usr/bin/env python3
"""Cold five-target-profile evidence composition without native reader outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_target_profile as P


def main():
    """Score once to a new output; never read raw text or application scales."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text())
    P.validate_bundle(bundle)
    with args.input.open() as source, args.output.open("x") as sink:
        for line in source:
            sink.write(
                json.dumps(P.cold_record(bundle, json.loads(line)), allow_nan=False)
                + "\n"
            )


if __name__ == "__main__":
    main()
