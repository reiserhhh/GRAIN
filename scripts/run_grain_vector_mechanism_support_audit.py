#!/usr/bin/env python3
"""Diagnose frozen GRAIN support refusals without relaxing the score gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.multicomment import finite_population_mean_radius
from scripts.run_grain_vector_mechanism_controls import write_json
from suica_core.v7_geometry import GeometryBundle, score_geometry_bundle


def main():
    """Vary only observed unit count and calculate mean/std radius contributions."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.parent_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["stage"] != "base":
        raise ValueError("Needs GRAIN unit baseline")
    if (
        json.loads((a.parent_dir / "run_status.json").read_text(encoding="utf-8"))[
            "status"
        ]
        != "COMPLETE"
    ):
        raise ValueError("Incomplete GRAIN baseline")
    units = np.load(a.parent_dir / "unit_vectors.npz")["grain_units"]
    bundle = GeometryBundle.from_dict(
        json.loads(
            (
                ROOT
                / "results/v7_geometry/g1_corrected_v2_full_20260715/geometry_bundle.json"
            ).read_text(encoding="utf-8")
        )
    )
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if any(a.output_dir.iterdir()):
        raise ValueError("Use new output directory")
    center, w = np.asarray(bundle.feature_center), np.asarray(bundle.metric_whitener)
    mean_ix = np.array([n.startswith("mean::") for n in bundle.feature_names])
    results = []
    for count in (3, 6, 12, 24, 32, 36):
        indices = np.linspace(0, 35, count, dtype=int)
        h = units[:, indices]
        mean, std = h.mean(1), h.std(1)
        raw = np.stack(
            [
                mean[:, int(name.split("_")[-1])]
                if name.startswith("mean::")
                else std[:, int(name.split("_")[-1])]
                for name in bundle.feature_names
            ],
            axis=1,
        )
        score = score_geometry_bundle(bundle, raw, unit_counts=np.full(len(raw), count))
        d = raw - center
        wm = (d * mean_ix) @ w
        ws = (d * ~mean_ix) @ w
        for i, uid in enumerate(manifest["authors"]):
            results.append(
                {
                    "author_id": uid,
                    "units": count,
                    "status": score["status"][i],
                    "radius": score["reference_radius"][i],
                    "mean_squared_component": float(wm[i] @ wm[i]),
                    "std_squared_component": float(ws[i] @ ws[i]),
                    "mean_std_cross": float(2 * wm[i] @ ws[i]),
                }
            )
        assert np.allclose(
            np.sum((d @ w) ** 2, axis=1),
            np.sum(wm**2, axis=1) + np.sum(ws**2, axis=1) + 2 * np.sum(wm * ws, axis=1),
        )
    frame = pd.DataFrame(results)
    frame.to_csv(a.output_dir / "private_radius_by_author.csv", index=False)
    aggregate = []
    for n, g in frame.groupby("units"):
        aggregate.append(
            {
                "units": int(n),
                "ready": int((g.status == "GEOMETRY_PROFILE_READY").sum()),
                "authors": len(g),
                "radius_median": float(g.radius.median()),
                "radius_min": float(g.radius.min()),
                "radius_max": float(g.radius.max()),
                "mean_squared_median": float(g.mean_squared_component.median()),
                "std_squared_median": float(g.std_squared_component.median()),
                "cross_median": float(g.mean_std_cross.median()),
            }
        )
    pd.DataFrame(aggregate).to_csv(a.output_dir / "support_by_volume.csv", index=False)
    mean_metric = (w @ w.T)[np.ix_(mean_ix, mean_ix)]
    mean_center = center[mean_ix]
    rng, moments = np.random.default_rng(109), []
    for count in (3, 6, 12, 24, 32, 36):
        base, inflation = finite_population_mean_radius(
            units, count, mean_center, mean_metric
        )
        monte_carlo = []
        for h in units:
            picks = np.argsort(rng.random((500, len(h))), axis=1)[:, :count]
            d = h[picks].mean(1) - mean_center
            monte_carlo.append(
                float(np.einsum("bi,ij,bj->b", d, mean_metric, d).mean())
            )
        monte_carlo = np.asarray(monte_carlo)
        predicted = base + inflation
        moments.append(
            {
                "subset_units": count,
                "pool_units": 36,
                "authors": len(units),
                "draws_per_author": 500,
                "expected_radius_mean": float(predicted.mean()),
                "monte_carlo_radius_mean": float(monte_carlo.mean()),
                "expected_sampling_inflation_mean": float(inflation.mean()),
                "finite_pool_radius_mean": float(base.mean()),
                "mean_absolute_relative_mc_error": float(
                    np.mean(abs(monte_carlo - predicted) / predicted)
                ),
            }
        )
    pd.DataFrame(moments).to_csv(
        a.output_dir / "finite_population_moment_check.csv", index=False
    )
    write_json(
        a.output_dir / "manifest.json",
        {
            "parent_identity": manifest["identity"],
            "bundle_identity": material_hash(bundle.to_dict()),
            "threshold_unchanged": True,
            "claim": "Volume-only sensitivity, no questionnaire or threshold fitting",
        },
    )
    print(pd.DataFrame(aggregate).to_string(index=False))
    print(pd.DataFrame(moments).to_string(index=False))


if __name__ == "__main__":
    main()
