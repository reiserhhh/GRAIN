#!/usr/bin/env python3
"""Standalone bank/partition scorer that never opens a criterion file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_block_observation as M
from scripts import run_grain_observed_family_hm170 as R


def main():
    """Read only frozen SOURCE programs and the explicitly label-free archive."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "input", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    bundle = R.read(args.bundle)
    with np.load(args.input, allow_pickle=False) as d:
        if set(d.files) != {"mu", "gamma", "views", "folds"}:
            raise ValueError("Cold input must not include labels or private material")
        readings = M.cold_readings(
            d["mu"], d["gamma"], d["views"], d["folds"], bundle["programs"]
        )
    R.archive(args.out, **readings)
    R.seal(args.out.parent, "cold_seal.json", (args.out.name,))
    print(
        json.dumps({"shape": list(readings["view_scores"].shape), "labels_read": False})
    )


if __name__ == "__main__":
    main()
