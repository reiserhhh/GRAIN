#!/usr/bin/env python3
"""Score a numerical event JSONL using a frozen empirical SOURCE law only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_joint_evidence as J


def main():
    """No model, application outcome, teacher answer or source refitting."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    bundle = json.loads(args.bundle.read_text())
    J.validate_bundle(bundle)
    with args.input.open() as source, args.output.open("x") as out:
        for line in source:
            result = J.cold_record(bundle, json.loads(line))
            out.write(json.dumps(result, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
