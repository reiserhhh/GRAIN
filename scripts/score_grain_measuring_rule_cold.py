#!/usr/bin/env python3
"""Read an explicit SOURCE measuring rule with no neural model or answer fitting."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction.math_measuring_rule import cold


def score(bundle, record):
    """Preserve both frozen vector spaces rather than selecting a better reading."""
    readings = {}
    for name in ("E5", "GRAIN15"):
        bank, event = dict(bundle), copy.deepcopy(record)
        if name == "GRAIN15":
            q = bundle["projection"]
            bank["prototypes"] = bank["prototypes"] @ q
            for item in event["events"]:
                item["vector"] = (np.asarray(item["vector"], float) @ q).tolist()
        readings[name] = cold(bank, event)
    return {"spaces": readings, "upstream_encoder_still_neural": True}


def main():
    """Accept only a numerical frozen source bank and one qualified event record."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--record", required=True, type=Path)
    args = p.parse_args()
    with np.load(args.bundle, allow_pickle=False) as z:
        bank = {
            key: z[key]
            for key in ("prototypes", "codes", "relink", "variance", "projection")
        }
    print(json.dumps(score(bank, json.loads(args.record.read_text())), allow_nan=False))


if __name__ == "__main__":
    main()
