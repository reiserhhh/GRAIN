#!/usr/bin/env python3
"""Cold condition-pairing observer: no text,labels,answers or model imports."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction import math_condition_pairing as M


def main():
    """Read only a frozen coefficient bundle and addressed numerical event frames."""
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with np.load(args.bundle, allow_pickle=False) as z:
        if set(z.files) != {"version", "fields"} or str(z["version"]) != M.VERSION:
            raise ValueError("Unexpected bundle or source metadata")
        fields = z["fields"]
    frames = supported = 0
    with args.input.open() as source, args.output.open("x") as sink:
        for line in source:
            value = M.cold_record(fields, json.loads(line))
            sink.write(json.dumps(value, allow_nan=False) + "\n")
            frames += 1
            supported += int(value.get("delta_support", False))
    print(
        json.dumps(
            {
                "frames": frames,
                "conditional_delta_supported": supported,
                "source_labels_native_answers_model_reads": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
