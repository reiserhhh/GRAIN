#!/usr/bin/env python3
"""Replay target/context probability contrasts without a neural runtime."""

import argparse
import json
import sys


def score(packet):
    """Calculate all five literal four-corner sums from numeric observations."""
    if packet["version"] != "HM258-v1":
        raise ValueError("Unknown observation contract")
    rows = []
    for unit in packet["units"]:
        ll = [
            [z - a for z, a in zip(zr, ar, strict=True)]
            for zr, ar in zip(unit["chosen"], unit["partition"], strict=True)
        ]
        if len(ll) != 2 or any(len(row) != 11 for row in ll):
            raise ValueError("Full observation family required")
        rows.append(
            {
                "unit_id": unit["unit_id"],
                "effect": [
                    ll[0][2 * q + 1]
                    - ll[0][2 * q + 2]
                    - ll[1][2 * q + 1]
                    + ll[1][2 * q + 2]
                    for q in range(5)
                ],
                "neutral": ll[0][0] - ll[1][0],
            }
        )
    return {
        "rows": rows,
        "neural_runtime_loaded": "torch" in sys.modules
        or "transformers" in sys.modules,
    }


def main():
    """Accept numeric accounts only; no raw text, model or label access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    with open(args.input) as stream:
        print(json.dumps(score(json.load(stream)), allow_nan=False))


if __name__ == "__main__":
    main()
