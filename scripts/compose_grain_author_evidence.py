#!/usr/bin/env python3
"""Compose observed evidence with an explicit author law and measurement object."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction.math_author_evidence_merge import compose


def score_profiles(observations, declaration, mode):
    """Score declared profiles; never silently infer author weights from attention."""
    f, z = observations["readings"], observations["log_mass"]
    targets = declaration["targets"]
    if f.ndim != 3 or z.shape != f.shape or len(targets) != f.shape[1]:
        raise ValueError("Registered target/head observations required")
    atom = declaration["atom_convention"]
    if atom not in ("token_sum", "unit_mean"):
        raise ValueError("Explicit token-sum or unit-mean atom convention required")
    if mode not in ("declared_expectation", "calibrated_partition"):
        raise ValueError("Explicit measurement object required")
    if mode == "calibrated_partition" and not declaration.get(
        "partition_scale_contract"
    ):
        raise ValueError("Cross-evidence partition scale contract required")
    counts = np.asarray(declaration["atom_counts"], float)
    if (
        counts.shape != (len(f),)
        or not np.isfinite(counts).all()
        or (counts <= 0).any()
    ):
        raise ValueError("Positive declared record atom counts required")
    scores, names = [], set()
    for profile in declaration["profiles"]:
        name = profile["name"]
        ids = np.asarray(profile["record_indices"])
        if (
            name in names
            or ids.ndim != 1
            or not np.issubdtype(ids.dtype, np.integer)
            or len(ids) == 0
            or len(set(ids.tolist())) != len(ids)
            or np.min(ids) < 0
            or np.max(ids) >= len(f)
        ):
            raise ValueError(
                "Distinct named profiles and unique existing records required"
            )
        names.add(name)
        mass = z[ids] - (np.log(counts[ids, None, None]) if atom == "unit_mean" else 0)
        expected, merged, gap = compose(f[ids], mass, profile["weights"])
        head_values = expected if mode == "declared_expectation" else merged
        scores.append(
            {
                "profile": name,
                "score_by_target": dict(
                    zip(targets, head_values.sum(-1).tolist(), strict=True)
                ),
                "partition_minus_expectation_by_target": dict(
                    zip(targets, gap.sum(-1).tolist(), strict=True)
                ),
                "evidence_records": len(ids),
            }
        )
    if not scores:
        raise ValueError("No profiles; missing observations are not zero scores")
    return {
        "measurement_object": mode,
        "atom_convention": atom,
        "scores": scores,
        "partition_is_not_author_choice": True,
        "not_psychologically_calibrated": True,
        "new_neural_calls": 0,
        "new_fits": 0,
    }


def main():
    """Use a cold numeric observation file and explicit author-weight declaration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--declaration", required=True, type=Path)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("declared_expectation", "calibrated_partition"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with np.load(args.observations, allow_pickle=False) as z:
        result = score_profiles(z, json.loads(args.declaration.read_text()), args.mode)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print("COMPOSED", len(result["scores"]), "profiles;", args.mode)


if __name__ == "__main__":
    main()
