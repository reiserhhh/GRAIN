#!/usr/bin/env python3
"""Cold numeric all-target scoring with an externally supplied bundle hash."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_vector_likelihood as M
from scripts import run_grain_observed_family_hm170 as R


def main():
    """Load only frozen source coefficients and addressed application vectors."""
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if R.digest(args.bundle) != args.expected_sha256:
        raise ValueError("Caller bundle hash mismatch")
    b = R.read(args.bundle)
    if set(b) != {
        "version",
        "targets",
        "kernel",
        "kernel_sha256",
        "code_sha256",
        "variance",
        "reference",
        "interpretation",
    }:
        raise ValueError("Unexpected source bundle fields")
    if b["version"] != M.VERSION or b["targets"] != list(M.TARGETS):
        raise ValueError("Unknown source reader")
    if b["code_sha256"] != R.digest(Path(M.__file__)):
        raise ValueError("Reader implementation changed")
    kernel = args.bundle.parent / b["kernel"]
    if R.digest(kernel) != b["kernel_sha256"]:
        raise ValueError("Source observation law changed")
    with np.load(kernel, allow_pickle=False) as z:
        prototypes, labels = z["prototypes"], z["labels"]
    if prototypes.ndim != 3 or prototypes.shape[0] != 2:
        raise ValueError("Two frozen source windows required")
    if not np.array_equal(np.asarray(b["reference"]), labels.mean(0)):
        raise ValueError("Declared reference is not the source population mean")
    if (
        b["interpretation"]
        != "posterior_under_declared_source_Gaussian_law_not_human_probability"
    ):
        raise ValueError("Unknown measurement interpretation")
    with args.input.open() as stream:
        records = [json.loads(line) for line in stream]
    keys = [r["frame_key"] for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate frame identity")
    points = np.array([M.cold_point(r, prototypes.shape[-1]) for r in records])
    output = []
    for source_view in range(2):
        made = M.read(prototypes[source_view], labels, b["variance"], points)
        for i, key in enumerate(keys):
            output.append(
                {
                    "frame_key": key,
                    "source_view": source_view,
                    "posterior": made["posterior"][i].tolist(),
                    "relative": made["relative"][i].tolist(),
                    "profile": made["profile"][i].tolist(),
                    "log_odds": made["log_odds"][i].tolist(),
                }
            )
    with args.output.open("x") as stream:
        for row in output:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "frames": len(records),
                "source_views": 2,
                "target_readings": len(output) * 5,
                "new_model_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
