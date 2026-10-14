#!/usr/bin/env python3
"""Cold SOURCE-qualified pair scoring, no neural decoder or evaluated labels."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction.math_criterion_reader import cold_pair


def main():
    """Read a frozen numerical SOURCE bundle and one addressed vector pair."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--space", choices=("E5", "GRAIN15"), required=True)
    args = p.parse_args()
    with np.load(args.bundle, allow_pickle=False) as z:
        bundle = {k: z[k] for k in z.files}
    with args.record.open() as f:
        record = json.load(f)
    print(json.dumps(cold_pair(bundle, record, args.space), allow_nan=False))


if __name__ == "__main__":
    main()
