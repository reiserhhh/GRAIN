#!/usr/bin/env python3
"""Cold source-anchored numeric scoring without labels or new model calls."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_trait_anchor as A
from scripts.run_grain_observed_family_hm170 import digest


def main():
    """Authenticate one frozen bundle and score each supplied current window."""
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("bundle", "sha256", "input", "output"):
        p.add_argument("--" + field, required=True)
    args = p.parse_args()
    if digest(args.bundle) != args.sha256:
        raise ValueError("Changed frozen target bundle")
    bundle = json.loads(Path(args.bundle).read_text())
    count = 0
    with Path(args.input).open() as src, Path(args.output).open("x") as dst:
        for line in src:
            dst.write(
                json.dumps(A.cold_author(bundle, json.loads(line)), allow_nan=False)
                + "\n"
            )
            count += 1
    print(
        json.dumps(
            {"cold_windows": count, "new_neural_forwards": 0, "labels_received": False}
        )
    )


if __name__ == "__main__":
    main()
