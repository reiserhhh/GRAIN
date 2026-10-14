#!/usr/bin/env python3
"""Cold, offline score with a frozen bundle; accepts vectors, not labels/IDs."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction.math_response import score_bundle


def main():
    """Validate dimensions and save two material-unit score coordinates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.bundle, allow_pickle=False) as bundle:
        model = {key: bundle[key] for key in bundle.files}
    x = np.load(args.vectors, allow_pickle=False)
    input_dim = len(model.get("factor_mu", model["mu"]))
    if x.ndim != 2 or x.shape[1] != input_dim or not np.all(np.isfinite(x)):
        raise ValueError("invalid vectors for this bundle")
    np.save(args.output, score_bundle(model, x))


if __name__ == "__main__":
    main()
