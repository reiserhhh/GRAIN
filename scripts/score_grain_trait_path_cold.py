#!/usr/bin/env python3
"""Read frozen numerical event ledgers without source or application labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_trait_path as F


def main():
    """Authenticate one source bundle and write immutable cold observations."""
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if hashlib.sha256(a.bundle.read_bytes()).hexdigest() != a.sha256:
        raise ValueError("Bundle identity mismatch")
    bundle = json.loads(a.bundle.read_text())
    F.verify_law(bundle)
    count = 0
    with a.input.open() as source, a.output.open("x") as dest:
        for line in source:
            value = F.cold_author(bundle, json.loads(line))
            dest.write(json.dumps(value, allow_nan=False) + "\n")
            count += 1
    print(
        json.dumps(
            {"windows": count, "labels_received": False, "new_neural_forwards": 0}
        )
    )


if __name__ == "__main__":
    main()
