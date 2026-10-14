#!/usr/bin/env python3
"""Cold nested path scoring without held outcomes, text models or refitting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_excursion_order as E
from grain_research.construction import math_path_reference as M


def main():
    """Stream outcome-free event records; explicit missing results remain null."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text())
    with args.input.open() as file:
        for line in file:
            if line.strip():
                result = M.score_record(bundle, json.loads(line))
                print(json.dumps(E.json_safe(result), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
