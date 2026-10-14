#!/usr/bin/env python3
"""Cold author scoring from the opposite historical SOURCE bank."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_pair_criterion as M


def score(path, record, source_bank, space):
    """Read SOURCE criteria only for scoring; reject evaluated-author overlap."""
    if set(record) != {"author_cache_id", "sampling_regime", "pairs"}:
        raise ValueError("Only addressed numerical pairs, no criterion or answer")
    if record["sampling_regime"] != "next_cached_record_within_registered_window":
        raise ValueError("This operator is not calibrated for another sampling regime")
    with np.load(path, allow_pickle=False) as z:
        mask = z["folds"] == source_bank
        if (
            not mask.any()
            or int(record["author_cache_id"]) in z["source_indices"][mask]
        ):
            raise ValueError("Disjoint source bank required")
        bundle = {k: z[k][mask] for k in ("prototypes", "codes", "relink")}
        bundle |= {k: z[k] for k in ("projection", "variance")}
    result = M.cold_author(bundle, record["pairs"], space)
    return {
        "space": space,
        "source_bank": source_bank,
        "windows": M.WINDOWS,
        "laws_and_components": M.SERIES,
        "targets": ("E", "N", "A", "C", "O"),
        **{k: v.tolist() for k, v in result.items()},
        "overlapping_pairs_averaged_not_multiplied": True,
        "evaluated_author_criterion_used": False,
    }


def main():
    """Emit fixed arithmetic-mean readings from numeric evidence in a new process."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--record", required=True, type=Path)
    p.add_argument("--source-bank", required=True, choices=(0, 1), type=int)
    p.add_argument("--space", required=True, choices=("E5", "GRAIN15"))
    a = p.parse_args()
    print(
        json.dumps(
            score(a.bundle, json.loads(a.record.read_text()), a.source_bank, a.space),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
