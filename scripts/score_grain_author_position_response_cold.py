#!/usr/bin/env python3
"""Numeric-only author position/response scoring; no outcome or model inputs."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_position_response as M


def main():
    """Score one qualified event catalogue against a frozen SOURCE reference."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    bundle = json.loads(a.bundle.read_text())
    inputs = json.loads(a.inputs.read_text())
    if set(bundle) != {"bank", "origin", "reference", "targets"} or bundle[
        "targets"
    ] != ["E", "N", "A", "C", "O"]:
        raise ValueError("Only the frozen numeric reference bundle is supported")
    if set(bundle["bank"]) != {"source_means", "codes", "variance"}:
        raise ValueError("Unexpected source inputs")
    rows = []
    for entry in inputs:
        if set(entry) != {"author", "event_ids", "vectors", "support", "counter"}:
            raise ValueError(
                "Unexpected application input; no labels or answers allowed"
            )
        result = M.score(
            bundle["bank"],
            entry["vectors"],
            entry["event_ids"],
            entry["support"],
            entry["counter"],
            bundle["origin"],
        )
        rows.append(
            {
                "author": entry["author"],
                **{
                    k: np.asarray(result[k]).tolist()
                    for k in ("absolute", "positions", "response")
                },
            }
        )
    with a.output.open("x") as f:
        json.dump(rows, f, allow_nan=False)
        f.write("\n")
    print(f"cold scored {len(rows)} addressed authors;0 model/outcome calls")


if __name__ == "__main__":
    main()
