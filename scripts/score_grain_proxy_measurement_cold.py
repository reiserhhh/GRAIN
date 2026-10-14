#!/usr/bin/env python3
"""Cold bridge scoring; imports no generator, hidden regimes or neural runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_proxy_measurement as M


def main():
    """Read only the observed bundle and new vector/choice input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    seal = json.loads((args.bundle.parent / "observed_seal.json").read_text())
    for path in (args.bundle, args.input):
        if hashlib.sha256(path.read_bytes()).hexdigest() != seal[path.name]:
            raise ValueError("Unsealed cold input or bundle")
    bundle = json.loads(args.bundle.read_text())
    profiles = []
    with np.load(args.input, allow_pickle=False) as data:
        if set(data.files) != {"ids", "conditions", "vectors", "weights"}:
            raise ValueError("Cold input must not contain hidden/application outcomes")
        for index, identifier in enumerate(data["ids"]):
            profiles.append(
                {
                    "id": str(identifier),
                    "reading": M.cold_profile(
                        bundle,
                        data["conditions"][index],
                        data["vectors"][index],
                        data["weights"][index],
                    ),
                }
            )
    with args.out.open("x") as stream:
        json.dump(
            {"profiles": profiles, "no_generator_or_application_outcome_read": True},
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
    print(json.dumps({"profiles": len(profiles), "out": str(args.out)}))


if __name__ == "__main__":
    main()
