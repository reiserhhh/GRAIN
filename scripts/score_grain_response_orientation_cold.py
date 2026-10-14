#!/usr/bin/env python3
"""Cold signed-response accounts from numeric GRAIN/source observations only."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_response_orientation import cold


def main():
    """Read a qualified record and emit all target/query/output-channel accounts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(cold(json.loads(args.input.read_text())), allow_nan=False))


if __name__ == "__main__":
    main()
