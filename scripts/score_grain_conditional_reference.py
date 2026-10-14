#!/usr/bin/env python3
"""Cold-score condition profiles from local frozen E5 vectors and JSON reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.conditional_reference import completion_envelope, score_reference
from scripts.prepare_grain_predecessor_reference import digest, write


def restore_reference(value):
    """Restore JSON arrays, rejecting malformed or unfrozen reference mass."""
    ref = value.copy()
    for key in (
        "weights",
        "common5_weights",
        "mean",
        "source_authors_per_cell",
        "source_indices",
    ):
        ref[key] = np.asarray(ref[key])
    size = len(ref["keys"])
    if size == 0 or len(set(ref["keys"])) != size or ref["mean"].shape[0] != size:
        raise ValueError("Invalid reference cells")
    for key in ("weights", "common5_weights"):
        p = ref[key]
        if (
            p.shape != (size,)
            or not np.isfinite(p).all()
            or np.any(p < 0)
            or not np.isclose(p.sum(), 1)
        ):
            raise ValueError("Invalid reference weights")
    if not np.isfinite(ref["mean"]).all():
        raise ValueError("Invalid source profile")
    return ref


def score_packet(conditions, vectors, reference, direction, targets, target="full"):
    """Normalize specified encoder vectors and emit finite JSON-ready scores."""
    h = np.asarray(vectors, float)
    if h.ndim != 2 or h.shape[1] != direction.shape[0] or not np.isfinite(h).all():
        raise ValueError("Finite vectors in frozen encoder coordinates required")
    norms = np.linalg.norm(h, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError("Zero vector cannot be treated as neutral evidence")
    scores = (h / norms[:, None]) @ direction
    s = score_reference(conditions, scores, reference, target)
    low, high, _, _ = completion_envelope(s["center"], s["missing"], direction, targets)
    result = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in s.items()}
    result.update(
        {
            "target": target,
            "lower": low.tolist(),
            "upper": high.tolist(),
            "envelope_type": "fixed empirical observed-cell means; unit-ball missing-cell completion, NOT sampling CI",
            "supported_target_warning": "Its reference distribution is author-specific when coverage<1; do not treat it as a common-target score",
            "vector_requirement": "Same frozen E5 revision/prefix/token context as construction; dimensional equality alone is insufficient",
        }
    )
    return result


def main():
    """Score one author locally; no model fitting, inference calls or labels."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--directions", type=Path, required=True)
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="NPZ with conditions and vectors for one author",
    )
    p.add_argument("--target", choices=["full", "common5"], default="full")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ref = restore_reference(json.loads(args.reference.read_text()))
    with np.load(args.input, allow_pickle=False) as z:
        c, h = z["conditions"], z["vectors"]
    with np.load(args.directions, allow_pickle=False) as z:
        result = score_packet(c, h, ref, z["direction"], z["targets"], args.target)
    result["inputs"] = {
        str(path): digest(path)
        for path in (args.reference, args.directions, args.input)
    }
    result["scorer_sha256"] = digest(Path(__file__))
    write(args.output, result)
    print(
        json.dumps(
            {
                "status": "SCORED",
                "target": args.target,
                "coverage": result["coverage"],
                "point_available": result["point"] is not None,
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
