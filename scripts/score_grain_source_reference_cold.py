#!/usr/bin/env python3
"""Read a source-reference certificate without native outputs or current labels."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction import math_source_reference_action as M


def main():
    """Use the explicitly addressed source/destination instruments and event edit."""
    p = argparse.ArgumentParser()
    p.add_argument("--banks", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--source", type=int, choices=(0, 1), required=True)
    args = p.parse_args()
    with np.load(args.banks, allow_pickle=False) as z:
        banks = [
            {k[3:]: z[k] for k in z.files if k.startswith(f"b{b}_")} for b in (0, 1)
        ]
    result = M.cold(
        banks[args.source], banks[1 - args.source], json.loads(args.input.read_text())
    )
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
