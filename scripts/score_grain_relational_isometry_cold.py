#!/usr/bin/env python3
"""Cold supported-transport scorer with numeric vectors and calibration only."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_relational_isometry import score_vectors


def main():
    """Write coordinates, raw source readout, reference movement and remainder."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--arm",
        default="matched_anchors",
        choices=("no_rotation", "matched_anchors", "wrong_anchors"),
    )
    args = parser.parse_args()
    with np.load(args.bundle, allow_pickle=False) as z:
        bundle = dict(z)
    with np.load(args.input, allow_pickle=False) as z:
        values = z["vectors"]
    scored = score_vectors(values, bundle, args.arm)
    with args.out.open("xb") as stream:
        np.savez_compressed(stream, **scored)
    if any(
        n in sys.modules for n in ("torch", "transformers", "sentence_transformers")
    ):
        raise RuntimeError("Cold arithmetic unexpectedly loaded a neural runtime")
    print(f"Cold transport scorer: {len(values)} rows, no neural runtime", flush=True)


if __name__ == "__main__":
    main()
