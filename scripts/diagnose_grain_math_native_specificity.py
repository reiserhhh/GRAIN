#!/usr/bin/env python3
"""Describe target-specific versus shared responses after frozen HM10B1."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json, write_json


def correlation(x, y):
    """Return a descriptive Pearson correlation, or null for a constant input."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 1 or not np.isfinite([x, y]).all():
        raise ValueError("matched finite one-dimensional readouts required")
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def run(source, out):
    """Compute paired descriptive contrasts without fitting or new observations."""
    if out.exists():
        raise FileExistsError("preserve prior specificity diagnostic")
    hashes = read_json(source / "03/artifact_hashes.json")
    for name, expected in hashes.items():
        if digest(source / name) != expected:
            raise ValueError("primary packet changed")
    manifest = read_json(source / "00/manifest.json")
    with (source / "02/interactions.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (source / "02/content_incidence_pairs.csv").open(newline="") as stream:
        pairs = list(csv.DictReader(stream))
    queries = manifest["spec"]["queries"]
    stats, lookup = [], {}
    for query in queries:
        selected = [r for r in rows if r["query"] == query]
        keyed = {(r["cube_id"], r["incidence"]): r for r in selected}
        if len(keyed) != len(selected):
            raise ValueError("duplicate matched cube")
        lookup[query] = keyed
        affine = np.array([float(r["affine_interaction"]) for r in selected])
        total = np.array([float(r["total"]) for r in selected])
        action = np.array([float(r["representation_action"]) for r in selected])
        link = np.array([float(r["link_only"]) for r in selected])
        groups = []
        for frame in range(4):
            for actor in ("self", "other"):
                group = [
                    float(r["native_delta"])
                    for r in pairs
                    if r["query"] == query
                    and r["identical_raw"] == "False"
                    and int(r["frame"]) == frame
                    and r["actor"] == actor
                ]
                if len(group) != 32:
                    raise ValueError("same-word incidence group support changed")
                groups.append(
                    {
                        "frame": frame,
                        "actor": actor,
                        "pairs": len(group),
                        "rms": float(np.sqrt(np.mean(np.asarray(group) ** 2))),
                    }
                )
        stats.append(
            {
                "query": query,
                "affine_native_joint_r": correlation(affine, total),
                "affine_native_joint_rmse": float(
                    np.sqrt(np.mean((affine - total) ** 2))
                ),
                "opposing_link_and_representation": int(np.sum(link * action < 0)),
                "joint_rows": len(selected),
                "incidence_by_frame_actor": groups,
            }
        )
    cross = []
    for i, query in enumerate(queries):
        for other in queries[i + 1 :]:
            left, right = lookup[query], lookup[other]
            if set(left) != set(right):
                raise ValueError("cross-query cube pairing changed")
            keys = sorted(left)
            x = [float(left[k]["representation_action"]) for k in keys]
            y = [float(right[k]["representation_action"]) for k in keys]
            dx = np.array([json.loads(left[k]["logit_interaction"]) for k in keys])
            dy = np.array([json.loads(right[k]["logit_interaction"]) for k in keys])
            denominator = np.sqrt(np.sum(dx**2) * np.sum(dy**2))
            cross.append(
                {
                    "queries": [query, other],
                    "scalar_joint_action_r": correlation(x, y),
                    "pooled_natural_interaction_cosine": float(
                        np.sum(dx * dy) / denominator
                    )
                    if denominator
                    else None,
                    "rows": len(keys),
                }
            )
    result = {
        "status": "DESCRIPTIVE_RESULT_AFTER_COMPLETE",
        "new_model_calls": 0,
        "target_refits": 0,
        "not_new_independent_replication": True,
        "correlations_are_not_personality_validity": True,
        "query_stats": stats,
        "cross_query_diagnostic": cross,
        "primary_hashes_checked": len(hashes),
        "source_manifest_sha256": digest(source / "03/artifact_hashes.json"),
        "code_sha256": digest(Path(__file__)),
    }
    out.mkdir(parents=True)
    write_json(out / "summary.json", result)
    print(json.dumps(result), flush=True)


def main():
    """Run one explicit result-after descriptive companion, preserving all data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/native_joint_r3",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "results/grain_math_response_v1/native_specificity_diagnostic_r1",
    )
    args = parser.parse_args()
    run(args.source, args.out)


if __name__ == "__main__":
    main()
