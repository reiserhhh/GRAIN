#!/usr/bin/env python3
"""Two-input numerical evidence scorer; never imports a transformer or text reader."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction.math_author_evidence_merge import cold_score


def main():
    """Write per-evidence/head scores and partitions to an exclusive new file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", required=True, type=Path)
    parser.add_argument("--carriers", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reader = json.loads(args.reader.read_text())
    with np.load(args.carriers, allow_pickle=False) as data:
        f, z = cold_score(reader, data)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, readings=f, log_mass=z)
    print("COLD SCORED", len(f), "numeric records; no neural or fit")


if __name__ == "__main__":
    main()
