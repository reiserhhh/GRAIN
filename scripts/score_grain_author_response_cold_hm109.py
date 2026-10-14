#!/usr/bin/env python3
"""Cold numeric author scoring: no model, truth, probability laws or intervention."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction.math_author_response_observability import cold_scores


def main():
    """Read only the two explicitly admitted numeric files and save author accounts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--histories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (
        np.load(args.observations, allow_pickle=False) as obs,
        np.load(args.histories, allow_pickle=False) as histories,
    ):
        out = cold_scores(obs["observations"], obs["groups"], histories["addresses"])
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **out)
    print("Cold HM109 scored explicit event histories only", flush=True)


if __name__ == "__main__":
    main()
