#!/usr/bin/env python3
"""Cold SOURCE assignment accounts; never load cached model answers."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_anchor_law_disintegration as M


def main():
    """Load explicitly allowed SOURCE keys and an addressed vector-only record."""
    p = argparse.ArgumentParser()
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--record", type=Path, required=True)
    args = p.parse_args()
    with np.load(args.bank, allow_pickle=False) as z:
        bank = {k: z[k] for k in ("source_means", "codes", "variance")}
    record = json.loads(args.record.read_text())
    print(json.dumps(M.cold(bank, record), allow_nan=False))


if __name__ == "__main__":
    main()
