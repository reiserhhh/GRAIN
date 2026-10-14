#!/usr/bin/env python3
"""Cold numeric observation package; no tokenizer, text, model or answer input."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_history_routing import read_current


def score(bundle, data):
    """Retain absolute readings, joint channels and known-condition contributions."""
    p = np.asarray(data["probabilities"], float)
    c = np.asarray(data["condition_index"], int)
    if (
        p.ndim != 2
        or len(c) != p.shape[1]
        or np.any(p < 0)
        or not np.allclose(p.sum(1), 1)
    ):
        raise ValueError(
            "Known observation probabilities and condition indices required"
        )
    readings = np.array(
        [
            read_current(k, v, int(data["endpoint"]), bundle)
            for k, v in zip(data["k"], data["v"], strict=True)
        ]
    )
    absolute = np.einsum("un,nmq->umq", p, readings)
    indexed = np.array(
        [
            np.einsum("un,nq->uq", p[:, c == i], readings[c == i, 1])
            for i in range(int(c.max()) + 1)
        ]
    )
    return {
        "readings": readings,
        "absolute": absolute,
        "condition_contributions": indexed,
        "joint": absolute.reshape(len(p), -1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior cold outputs")
    with np.load(args.input, allow_pickle=False) as z:
        result = score(json.loads(args.bundle.read_text()), dict(z))
    if any(
        n in sys.modules for n in ("torch", "transformers", "sentence_transformers")
    ):
        raise RuntimeError("Cold numeric process loaded neural runtime")
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **result)
    print("HM127 cold24observations/two virtual laws;no neural runtime")
