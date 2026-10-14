#!/usr/bin/env python3
"""Score numeric evidence edits without loading native outputs or model weights."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction import math_diagnostic_action_bridge as M


def main():
    """Accept just one frozen source-bank file and an addressed vector input."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--source-bank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.bank, allow_pickle=False) as z:
        prefix = f"b{args.source_bank}_"
        bank = {key: z[prefix + key] for key in ("prototypes", "codes", "variance")}
    record = json.loads(args.input.read_text())
    print(json.dumps(M.cold(bank, record), allow_nan=False))


if __name__ == "__main__":
    main()
