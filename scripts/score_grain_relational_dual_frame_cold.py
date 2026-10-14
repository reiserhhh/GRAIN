#!/usr/bin/env python3
"""Cold numeric scorer: vector batch plus sealed frame, no questions or labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_relational_dual_frame import score_vectors


def main():
    """Create numeric coordinates without loading tokenizer/model/target answers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.frame, allow_pickle=False) as z:
        frame = dict(z)
    with np.load(args.input, allow_pickle=False) as z:
        x = z["vectors"]
    scored = score_vectors(x, frame)
    with args.out.open("xb") as stream:
        np.savez_compressed(stream, **scored)
    if any(
        n in sys.modules for n in ("torch", "transformers", "sentence_transformers")
    ):
        raise RuntimeError("Cold scoring unexpectedly loaded a neural runtime")
    print(f"Cold numeric score: {len(x)} rows; no neural runtime", flush=True)


if __name__ == "__main__":
    main()
