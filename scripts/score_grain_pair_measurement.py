#!/usr/bin/env python3
"""Cold numeric scoring from a frozen instrument and one current packet only."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_pair_measurement as M


def serial(value):
    """Keep missing values null and addressed arrays in human-readable JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    return value


def main():
    """No dataset, source ID catalogue, scale labels or model access at runtime."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instrument", type=Path, required=True)
    p.add_argument("--packet", type=Path, required=True)
    a = p.parse_args()
    q = M.score(json.loads(a.instrument.read_text()), json.loads(a.packet.read_text()))
    print(json.dumps(serial(q), allow_nan=False))


if __name__ == "__main__":
    main()
