#!/usr/bin/env python3
"""Cold event-weighted author observations, without text, model or truth inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction.math_qualified_composition import cold_scores


def main():
    """Accept only explicit numeric evidence and event counts; write exclusively."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as z:
        scores = cold_scores(z["observations"], z["counts"], z["owners"])
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, scores=scores)
    print("HM110 cold numeric scoring completed", flush=True)


if __name__ == "__main__":
    main()
