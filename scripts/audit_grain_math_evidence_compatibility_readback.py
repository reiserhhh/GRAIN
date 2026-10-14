#!/usr/bin/env python3
"""Result-after HM67 mean-line, summary and toy readback; no scientific rerun."""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_evidence_compatibility import OUT, bound_check, write


def audit(out):
    """Verify reported mechanisms using a reshape-based alternate cube traversal."""
    dest = out / "04_summary_readback"
    if dest.exists():
        raise FileExistsError("Preserve previous readbacks")
    bound_check(out)
    dest.mkdir()
    write(
        dest / "manifest.json",
        {
            "script_sha256": digest(Path(__file__)),
            "output_seal_sha256": digest(out / "output_seal.json"),
            "result_after_not_independent_replication": True,
        },
    )
    with np.load(out / "observation_tables.npz", allow_pickle=False) as f:
        observations = {k: f[k].copy() for k in f.files}
    with np.load(out / "source_lines.npz", allow_pickle=False) as f:
        source = {k: f[k].copy() for k in f.files}
    with np.load(out / "synthetic_material.npz", allow_pickle=False) as f:
        toys = {k: f[k].copy() for k in f.files}
    maximum, count = 0.0, 0

    def compare(a, b):
        """Count actual matching values, not empty counter assertions."""
        nonlocal maximum, count
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Readback shape/finiteness mismatch")
        maximum = max(maximum, float(np.max(np.abs(a - b), initial=0)))
        count += a.size

    def direct_edges(value):
        """Take tensor-axis finite differences rather than address/bit stepping."""
        tensor = value.reshape((2,) * 6 + (-1,))
        return np.asarray(
            [
                (np.take(tensor, 1, axis=j) - np.take(tensor, 0, axis=j)).reshape(
                    32, -1
                )
                for j in range(6)
            ]
        )

    def directional_fraction(value, lines):
        """Compute full normal energy with a rowwise Gram identity."""
        edges = direct_edges(value)
        units = lines / np.linalg.norm(lines, axis=1, keepdims=True)
        parallel = np.einsum("jcd,jd->jc", edges, units)
        normal = edges - parallel[:, :, None] * units[:, None]
        total = float(np.square(edges).sum())
        return float(np.square(normal).sum() / total), normal

    for key, value in observations.items():
        if key.endswith("_own_lines"):
            compare(
                value,
                direct_edges(observations[key.removesuffix("_own_lines")]).mean(1),
            )
    for stage, value in source.items():
        compare(value, direct_edges(observations[f"f0_{stage}"]).mean(1))
    summary = read_json(out / "summary.json")
    with (out / "square_compatibility.csv").open(newline="") as f:
        squares = list(csv.DictReader(f))
    for row in summary["stage_summaries"]:
        key = f"f{row['family']}_{row['stage']}"
        lines = (
            source[row["stage"]]
            if row["reference"] == "source_family0"
            else observations[key + "_own_lines"]
        )
        fraction, normal = directional_fraction(observations[key], lines)
        compare(fraction, row["normal_fraction"])
        compare(np.square(normal).sum(), row["normal_energy"])
        selected = [
            r
            for r in squares
            if all(str(row[k]) == r[k] for k in ("family", "stage", "reference"))
        ]
        compare(
            sum(float(r["projection_bound"]) for r in selected) / 5,
            row["projection_global_bound"],
        )
        compare(
            sum(float(r["angle_bound"]) for r in selected) / 5,
            row["angle_global_bound"],
        )
        if row["projection_global_bound"] > row["normal_energy"] + 1e-9:
            raise ValueError("Global edge-count bound failed")
    with (out / "synthetic_mechanisms.csv").open(newline="") as f:
        toy_rows = list(csv.DictReader(f))
    for row in toy_rows:
        key = f"{row['world']}_{row['seed']}_{row['stage']}"
        value = toys[key]
        edges = direct_edges(value)
        fraction, _ = directional_fraction(value, edges.mean(1))
        compare(fraction, float(row["normal_fraction"]))
        tensor = value.reshape((2,) * 6 + (-1,))
        energy = 0.0
        for j, k in itertools.combinations(range(6), 2):
            delta = np.take(tensor, 1, axis=k) - np.take(tensor, 0, axis=k)
            second = np.take(delta, 1, axis=j) - np.take(delta, 0, axis=j)
            energy += float(np.square(second).sum())
        compare(energy, float(row["interaction_energy"]))
    if maximum > 1e-9 or len(toy_rows) != 24:
        raise ValueError("Readback discrepancy or incomplete toy material")
    result = {
        "status": "VERIFIED_RESULT_AFTER_HM67_SUMMARY_AND_TOYS",
        "actual_comparisons": count,
        "maximum_alternate_error": maximum,
        "toy_stage_records_verified": len(toy_rows),
        "global_edge_multiplicity": 5,
        "no_new_calls_or_fits": True,
        "same_agent_not_independent_replication": True,
    }
    write(dest / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    audit(OUT)
