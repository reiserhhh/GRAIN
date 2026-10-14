#!/usr/bin/env python3
"""Cold signed item-pattern accounts,not human personality predictions."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_item_configuration import cold


def main():
    """Read identified numeric records and return every preserved target/query part."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(cold(json.loads(args.input.read_text())), allow_nan=False))


if __name__ == "__main__":
    main()
