#!/usr/bin/env python3
"""Cold numeric scorer: no source population,labels,native answers or model import."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction import math_joint_target_atlas as M


def main():
    """Load a frozen coefficient bank and score one addressed frame at a time."""
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with np.load(args.bundle, allow_pickle=False) as z:
        if (
            set(z.files) != {"version", "fields", "semantic"}
            or str(z["version"]) != M.VERSION
        ):
            raise ValueError("Unexpected cold bundle or source metadata")
        fields, semantic = z["fields"], z["semantic"]
    count = 0
    with args.input.open() as source, args.output.open("x") as sink:
        for line in source:
            value = M.cold_record(fields, semantic, json.loads(line))
            sink.write(json.dumps(value, allow_nan=False) + "\n")
            count += 1
    print(json.dumps({"frames": count, "model_or_source_reads": 0}))


if __name__ == "__main__":
    main()
