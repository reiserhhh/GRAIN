#!/usr/bin/env python3
"""Score numeric ordered events from one frozen source bundle, without an LLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_path_completion as M


def main():
    """Stream cold records, rejecting outcomes and repeated unordered events."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text())
    stream = args.input.open() if args.input else sys.stdin
    try:
        for line in stream:
            row = M.score_record(bundle, json.loads(line))
            print(
                json.dumps(
                    {k: M.serializable(v) for k, v in row.items() if k != "profiles"},
                    allow_nan=False,
                )
            )
    finally:
        if args.input:
            stream.close()


if __name__ == "__main__":
    main()
