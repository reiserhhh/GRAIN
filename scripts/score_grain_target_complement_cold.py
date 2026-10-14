#!/usr/bin/env python3
"""Cold complementary target readings, with no model or application labels."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_target_complement as M


def main():
    """Consume frozen fields and numerical events in an independent process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.bundle, allow_pickle=False) as z:
        if (
            set(z.files) != {"fields", "semantic", "version"}
            or str(z["version"]) != M.VERSION
        ):
            raise ValueError("Unexpected cold bundle or version")
        fields, semantic = z["fields"], z["semantic"]
    seen = set()
    with args.input.open() as source, args.output.open("x") as sink:
        for line in source:
            row = json.loads(line)
            if row["frame_key"] in seen:
                raise ValueError("Duplicate cold frame")
            seen.add(row["frame_key"])
            sink.write(
                json.dumps(M.cold_record(fields, semantic, row), allow_nan=False) + "\n"
            )
    print(json.dumps({"records": len(seen), "new_model_calls": 0}))


if __name__ == "__main__":
    main()
