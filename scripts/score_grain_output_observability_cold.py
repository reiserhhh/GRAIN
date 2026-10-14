#!/usr/bin/env python3
"""Cold numeric response-geometry accounts; no model or current psychological labels."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_output_observability import cold


def main():
    """Recompute registered cross-view accounts from identified numeric observations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(cold(json.loads(args.input.read_text())), allow_nan=False))


if __name__ == "__main__":
    main()
