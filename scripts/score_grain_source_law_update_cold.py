#!/usr/bin/env python3
"""Independent numeric source-law reader; no source labels or model outcomes."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_source_law_update as M


def main():
    """Score all registered source comparisons without selecting a target field."""
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    with np.load(a.bundle, allow_pickle=False) as z:
        if set(z.files) != {"fields", "version"} or z["version"].item() != M.VERSION:
            raise ValueError("Invalid numeric bundle")
        fields = z["fields"]
    count, missing = 0, 0
    with a.output.open("x") as sink:
        for line in a.input.read_text().splitlines():
            row = json.loads(line)
            value, n = M.cold_record(fields, row)
            sink.write(
                json.dumps(
                    {
                        "frame_key": row["frame_key"],
                        "view": row["view"],
                        "triples": n,
                        "retained": value.tolist(),
                        "conditional": (value * 64 / n).tolist() if n else None,
                    }
                )
                + "\n"
            )
            count += 1
            missing += n == 0
    print(json.dumps({"records": count, "missing": missing, "values": count * 960}))


if __name__ == "__main__":
    main()
