#!/usr/bin/env python3
"""Numerical source-profile reader:bundle and addressed vectors only."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction import math_profile_reader as M


def main():
    """Run the frozen bank independently of source labels and native responses."""
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("bundle", "input", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    b = json.loads(args.bundle.read_text())
    M.validate_bundle(b)
    with args.input.open() as source, args.output.open("x") as sink:
        for line in source:
            sink.write(
                json.dumps(M.cold_record(b, json.loads(line)), allow_nan=False) + "\n"
            )


if __name__ == "__main__":
    main()
