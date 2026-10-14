#!/usr/bin/env python3
"""Cold dyadic source-linkage contrast: no model call or application targets."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction.math_linked_join import cold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--record", required=True)
    args = parser.parse_args()
    with np.load(args.bundle, allow_pickle=False) as z:
        bank = {
            k: z[k]
            for k in ("source_comments", "codes", "anchors", "variance", "directions")
        }
    record = json.loads(Path(args.record).read_text())
    print(json.dumps(cold(bank, record), allow_nan=False))
