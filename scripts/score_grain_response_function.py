#!/usr/bin/env python3
"""Cold numeric scoring: no text, model, persona, task metadata or answers."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction.math_response_function import cold_readings


def main():
    """Apply an independently saved response program to numeric article features."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (
        np.load(args.features, allow_pickle=False) as data,
        np.load(args.program, allow_pickle=False) as program,
    ):
        readings = cold_readings(
            data["features"], program["origin"], program["coefficients"]
        )
    with args.output.open("xb") as handle:
        np.savez_compressed(handle, readings=readings)
    print(f"Cold numeric readings: {readings.shape}")


if __name__ == "__main__":
    main()
