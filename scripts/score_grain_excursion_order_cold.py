#!/usr/bin/env python3
"""Cold numeric complete-path scoring; neither neural inference nor held labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_excursion_order as M


def main():
    """Read a frozen SOURCE bundle and attested event records; emit nulls faithfully."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text())
    for line in args.input.read_text().splitlines():
        print(
            json.dumps(
                M.json_safe(M.score_record(bundle, json.loads(line))), allow_nan=False
            )
        )


if __name__ == "__main__":
    main()
