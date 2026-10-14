#!/usr/bin/env python3
"""Recompute complete route observations using numbers, without neural loading."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction import math_continuation_routes as C
from grain_research.construction import math_late_condition as L


def score(packet):
    """Reject missing families and retain signed finite graph interactions."""
    if packet["version"] != "HM259-v1":
        raise ValueError("Wrong observation contract")
    rows = []
    for unit in packet["units"]:
        values = {}
        for route in C.ROUTES:
            value = unit["routes"][route]
            result = L.accounts(
                np.asarray(value["chosen"])[..., None],
                np.asarray(value["partition"])[..., None],
            )
            values[route] = result["effect"][:, 0]
        rows.append(
            {
                "unit": unit["unit"],
                "terms": {k: v.tolist() for k, v in C.decompose(values).items()},
            }
        )
    return {
        "rows": rows,
        "neural_runtime_loaded": any(
            k in sys.modules for k in ("torch", "transformers")
        ),
    }


def main():
    """Expose a read-only standalone numeric scoring interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(score(json.loads(args.input.read_text())), allow_nan=False))


if __name__ == "__main__":
    main()
