#!/usr/bin/env python3
"""NAV-08: separate construction functions from a fixed reference-author population."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.functional_profiles import FixedAuthorCoordinates
from grain_research.vector_mechanisms.navigation import (
    QUERIES,
    digest,
    quality,
    read_json,
    write_json,
)
from scripts.run_grain_reference_panel import DEFAULT


def analyze(packet, out):
    """Freeze author norms, test new-batch invariance, and resample only references."""
    if out.exists():
        raise FileExistsError("Choose a new reference analysis directory")
    if read_json(packet / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Finish reference panel collection")
    if (
        digest(packet / "predictions.csv")
        != read_json(packet / "prediction_freeze.json")["sha256"]
    ):
        raise ValueError("Predictions changed")
    world = pd.DataFrame(read_json(packet / "private_worlds.json")).drop(
        columns="texts"
    )
    predictions = pd.read_csv(packet / "predictions.csv")
    frame = predictions.merge(world, on="world_id", validate="many_to_one")
    if (
        len(frame) != 432
        or frame.duplicated(["author_id", "view", "query", "readout"]).any()
    ):
        raise ValueError("Reference panel is incomplete or duplicated")
    out.mkdir(parents=True)
    positions, bootstrap_rows, stability, source_norms, conditions = [], [], [], {}, []
    maximum_batch_error = 0.0
    for arm, g in frame.groupby("readout"):
        values = (
            g.pivot(
                index=["author_id", "view", "role"], columns="query", values="score"
            )
            .loc[:, list(QUERIES)]
            .sort_index()
        )
        ref = values[values.index.get_level_values("role") == "reference"]
        held = values[values.index.get_level_values("role") == "evaluation"]
        train = ref.groupby(level="author_id").mean()
        if len(train) != 12 or len(held) != 24:
            raise ValueError("Expected12 reference authors/12 held authors with2 views")
        norm = FixedAuthorCoordinates().fit(train.to_numpy())
        z = norm.transform(values.to_numpy())
        source_norms[arm] = {
            "queries": list(QUERIES),
            "center": norm.center.tolist(),
            "scale": norm.scale.tolist(),
            "active": norm.active.tolist(),
            "n_reference_authors": len(train),
            "construction": "author average over two views, no current-batch fit",
        }
        position = (
            pd.DataFrame(z, index=values.index, columns=values.columns)
            .reset_index()
            .assign(readout=arm)
        )
        positions.append(position)
        batch = norm.transform(
            np.vstack([values.to_numpy(), np.ones((100, len(QUERIES))) * 1000000])
        )[: len(values)]
        single = np.vstack([norm.transform(x[None, :])[0] for x in values.to_numpy()])
        maximum_batch_error = max(
            maximum_batch_error,
            float(abs(z - batch).max()),
            float(abs(z - single).max()),
        )
        a, b = [held.xs(v, level="view").droplevel("role").sort_index() for v in (0, 1)]
        if not a.index.equals(b.index):
            raise ValueError("Held author pairing changed")
        for q in QUERIES:
            stability.append(
                {
                    "readout": arm,
                    "query": q,
                    **quality(a[q], b[q]),
                    "unit": "author paired views",
                }
            )
        held_mean = held.groupby(level="author_id").mean().to_numpy()
        baseline = norm.transform(held_mean)
        rng = np.random.default_rng(20260909)
        boot = []
        for _ in range(2000):
            ix = rng.integers(len(train), size=len(train))
            resampled = FixedAuthorCoordinates().fit(train.to_numpy()[ix])
            if not resampled.active.all():
                continue
            boot.append(resampled.transform(held_mean))
        samples = np.asarray(boot)
        lower, upper = np.quantile(samples, [0.025, 0.975], axis=0)
        for j, q in enumerate(QUERIES):
            bootstrap_rows.append(
                {
                    "readout": arm,
                    "query": q,
                    "reference_authors": 12,
                    "evaluation_authors": 12,
                    "valid_bootstraps": len(boot),
                    "mean_position_interval_width": float(
                        np.mean(upper[:, j] - lower[:, j])
                    ),
                    "mean_absolute_position_shift": float(
                        np.mean(abs(samples[:, :, j] - baseline[:, j]))
                    ),
                    "maximum_position_interval_width": float(
                        np.max(upper[:, j] - lower[:, j])
                    ),
                }
            )
        for ref_view in (0, 1):
            other = FixedAuthorCoordinates().fit(
                ref.xs(ref_view, level="view").to_numpy()
            )
            dz = other.transform(held_mean) - baseline
            for j, q in enumerate(QUERIES):
                conditions.append(
                    {
                        "readout": arm,
                        "query": q,
                        "reference_view": ref_view,
                        "position_rms_shift": float(np.sqrt(np.mean(dz[:, j] ** 2))),
                        "max_abs_shift": float(abs(dz[:, j]).max()),
                    }
                )
    positions = pd.concat(positions, ignore_index=True)
    positions.to_csv(out / "private_author_positions.csv", index=False)
    comparison = []
    held = (
        positions.query("role=='evaluation'")
        .groupby(["readout", "author_id"])[list(QUERIES)]
        .mean()
    )
    for q in QUERIES:
        real = held.xs("R_real", level="readout")[q]
        synth = held.xs("R_synth", level="readout")[q].reindex(real.index)
        comparison.append(
            {
                "query": q,
                "comparison": "real_vs_synthetic_construction_function",
                **quality(real, synth),
            }
        )
    pd.DataFrame(comparison).to_csv(
        out / "construction_reference_positions.csv", index=False
    )
    pd.DataFrame(stability).to_csv(out / "cross_view_stability.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(out / "reference_sensitivity.csv", index=False)
    pd.DataFrame(conditions).to_csv(
        out / "reference_material_sensitivity.csv", index=False
    )
    write_json(out / "author_norms.json", source_norms)
    write_json(
        out / "batch_invariance.json",
        {
            "maximum_old_author_shift": maximum_batch_error,
            "tested_rows": 144,
            "added_extreme_rows_per_arm": 100,
            "single_row_equals_batch": maximum_batch_error == 0,
            "norm_never_refit_on_evaluation_batch": True,
        },
    )
    write_json(
        out / "reference_manifest.json",
        {
            "status": "COMPLETE_FIXED_REFERENCE_PACKET",
            "queries": list(QUERIES),
            "reference_authors": 12,
            "evaluation_authors": 12,
            "views_per_author": 2,
            "construction_references": "two real or two synthetic texts per query, unchanged",
            "author_norm_reference": "separate fixed12-author source, two views averaged per author",
            "author_norm_sha256": digest(out / "author_norms.json"),
            "source_manifest_sha256": digest(packet / "manifest.json"),
            "prediction_sha256": digest(packet / "predictions.csv"),
            "script_sha256": digest(Path(__file__)),
            "scope": "local exploratory reference population, not general population or clinical norms",
            "human_labels_read": False,
            "MEPS_used": False,
        },
    )
    print(pd.DataFrame(stability).round(4).to_string(index=False), flush=True)
    print(pd.DataFrame(bootstrap_rows).round(4).to_string(index=False), flush=True)
    print({"batch_error": maximum_batch_error}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet", type=Path, default=DEFAULT)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    analyze(a.packet, a.out or a.packet / "analysis")
