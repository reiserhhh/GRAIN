#!/usr/bin/env python3
"""Cold fixed-reference choice/geometry accounts without application labels."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction import math_choice_geometry as M


def main():
    """Read one addressed numeric stream with an immutable numeric calibration."""
    parser = argparse.ArgumentParser()
    for name in ("bundle", "input", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    keys = {"q", "A", "G", "L", "f", "total"}
    with np.load(args.bundle, allow_pickle=False) as z:
        if (
            set(z.files) != keys | {"version", "fields", "basis", "mapping"}
            or str(z["version"]) != M.VERSION
        ):
            raise ValueError("Unexpected source information in bundle")
        calibration = {k: z[k] for k in keys}
        fields, basis, mapping = z["fields"], z["basis"], z["mapping"]
    frames = supported = 0
    with args.input.open() as source, args.output.open("x") as sink:
        for line in source:
            value = M.cold_record(fields, calibration, basis, mapping, json.loads(line))
            sink.write(json.dumps(value, allow_nan=False) + "\n")
            frames += 1
            supported += value["conditional_support"]
    print(
        json.dumps(
            {
                "frames": frames,
                "conditional_supported": supported,
                "source_labels_native_answers_model_reads": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
