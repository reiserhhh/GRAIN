#!/usr/bin/env python3
"""Numerical cold readback of the complete same-regime behavioral packet."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction.math_same_regime_reader import cold


def main():
    """Read local mathematical accounts, with no model, credential or API use."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("packet", type=Path)
    args = p.parse_args()
    print(json.dumps(cold(json.loads(args.packet.read_text())), allow_nan=False))


if __name__ == "__main__":
    main()
