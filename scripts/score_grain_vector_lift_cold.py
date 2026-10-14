#!/usr/bin/env python3
"""Read a fixed source vector function in an independent numeric-only process."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction import math_vector_lift as V


def main():
    """Consume only a sealed bundle and addressed numerical event records."""
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("bundle", "input", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    args = p.parse_args()
    b = json.loads(args.bundle.read_text())
    V.validate_bundle(b)
    with args.input.open() as source, args.output.open("x") as sink:
        for line in source:
            sink.write(
                json.dumps(V.cold_record(b, json.loads(line)), allow_nan=False) + "\n"
            )


if __name__ == "__main__":
    main()
