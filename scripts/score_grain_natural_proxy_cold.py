#!/usr/bin/env python3
"""Cold scorer: frozen numerical coefficients and label-free views only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_observed_family_hm170 as R
from scripts.run_grain_natural_proxy_hm253 import predictions


def main():
    """Score each whole author with the nonoverlapping calibration fold."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    bundle = json.loads(args.bundle.read_text())
    with np.load(args.input, allow_pickle=False) as d:
        if set(d.files) != {"features", "folds"}:
            raise ValueError("Cold input must contain no criteria or raw material")
        scores = predictions(d["features"], d["folds"], bundle["programs"])
    R.archive(args.out, scores=scores)
    R.seal(args.out.parent, "cold_seal.json", (args.out.name,))
    print(
        json.dumps(
            {
                "cold_shape": list(scores.shape),
                "criteria_read": False,
                "finite": bool(np.isfinite(scores).all()),
            }
        )
    )


if __name__ == "__main__":
    main()
