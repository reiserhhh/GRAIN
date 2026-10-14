#!/usr/bin/env python3
"""Cold condition-shape scorer: numeric events and a frozen bundle only."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction import math_condition_shape as M


def main():
    """Execute one cold input stream, preserving absent contrasts and replication flags."""
    p = argparse.ArgumentParser()
    for name in ("bundle", "input", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    with np.load(args.bundle, allow_pickle=False) as z:
        if (
            set(z.files) != {"version", "fields", "basis", "mapping"}
            or str(z["version"]) != M.VERSION
        ):
            raise ValueError("Unexpected source metadata or bundle")
        f, q, a = z["fields"], z["basis"], z["mapping"]
    frames = supported = dispersion = 0
    with args.input.open() as source, args.output.open("x") as sink:
        for line in source:
            value = M.cold_record(f, q, a, json.loads(line))
            sink.write(json.dumps(value, allow_nan=False) + "\n")
            frames += 1
            supported += value["conditional_support"]
            dispersion += value["dispersion_support"]
    print(
        json.dumps(
            {
                "frames": frames,
                "conditional_supported": supported,
                "dispersion_supported": dispersion,
                "source_labels_native_answers_model_reads": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
