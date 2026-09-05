#!/usr/bin/env python3
"""Separate local controllability, approximation quality and natural task validity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from grain_research.vector_mechanisms.sensitivity import region_derivative_stats
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.summarize_grain_vector_natural_patch import aggregate


def main():
    """Retain unmatched controls and numerical floors rather than selecting victories."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--gradient-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    parent, status = (
        read(a.parent_dir / "manifest.json"),
        read(a.parent_dir / "run_status.json"),
    )
    gm, gs = (
        read(a.gradient_dir / "manifest.json"),
        read(a.gradient_dir / "run_status.json"),
    )
    if (
        status["status"] != "COMPLETE"
        or gs["status"] != "COMPLETE"
        or parent["gradient_identity"] != gm["identity"]
    ):
        raise ValueError("Need complete linked capture and interventions")
    frame = pd.read_csv(a.parent_dir / "interventions.csv")
    if len(frame) != 2880 or frame.author_id.nunique() != 6 or status["jobs"] != 48:
        raise ValueError("Unexpected experimental population")
    if status["restore_max_error"] != 0 or status["self_max_error"] != 0:
        raise ValueError("Intervention controls failed")
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a new analysis directory")
    archive_source(ROOT, out / "source_snapshot.zip")
    frame["abs_delta"] = frame.delta.abs()
    frame["abs_prediction"] = frame.predicted_delta.abs()
    frame["absolute_linear_error"] = frame.linear_error.abs()
    frame["relative_linear_error"] = frame.absolute_linear_error / np.maximum(
        np.maximum(frame.abs_delta, frame.abs_prediction),
        0.25,
    )
    frame["resolved_sign"] = (frame.abs_delta >= 0.125) & (
        frame.abs_prediction >= 0.125
    )
    frame["sign_agreement"] = np.where(
        frame.resolved_sign,
        np.sign(frame.delta) == np.sign(frame.predicted_delta),
        np.nan,
    )
    frame["zero_actual_displacement"] = (frame.effect_norm == 0).astype(float)
    keys = ["task", "site", "family", "method", "rank", "alpha"]
    aggregate(
        frame,
        keys,
        [
            "delta",
            "abs_delta",
            "oriented_delta",
            "predicted_delta",
            "absolute_linear_error",
            "relative_linear_error",
            "resolved_sign",
            "sign_agreement",
            "decision_flip",
            "relative_effect_norm",
            "zero_actual_displacement",
        ],
    ).to_csv(out / "effects.csv", index=False)
    correlations = []
    for key, sub in frame.groupby(keys):
        correlation = None
        if sub.delta.std() > 0 and sub.predicted_delta.std() > 0:
            correlation = float(np.corrcoef(sub.delta, sub.predicted_delta)[0, 1])
        correlations.append(
            dict(zip(keys, key, strict=True))
            | {
                "rows": len(sub),
                "authors": sub.author_id.nunique(),
                "pearson_predicted_vs_actual_descriptive": correlation,
            }
        )
    pd.DataFrame(correlations).to_csv(
        out / "linear_prediction_correlations.csv", index=False
    )
    pairing = [
        "case_id",
        "author_id",
        "donor_id",
        "rule",
        "task",
        "site",
        "rank",
        "family",
    ]
    pairs = []
    for mode in ("natural", "matched"):
        target = frame[
            (frame.family == "current_subspace")
            & (frame.method == "sensitivity_" + mode)
            & (frame.alpha == 1)
        ]
        random = frame[frame.method.str.startswith("random_" + mode + "_")]
        random = random.groupby(pairing, as_index=False).agg(
            random_delta=("delta", "mean"),
            random_abs=("abs_delta", "mean"),
            random_oriented=("oriented_delta", "mean"),
            random_norm_error=("norm_match_max_relative_error", "max"),
        )
        joined = target.merge(random, on=pairing, validate="one_to_one")
        if len(joined) != 192:
            raise ValueError(
                "Every sensitivity candidate needs its two random controls"
            )
        joined["comparison_mode"] = mode
        joined["oriented_advantage"] = joined.oriented_delta - joined.random_oriented
        joined["absolute_advantage"] = joined.abs_delta - joined.random_abs
        joined["norm_accepted"] = joined.random_norm_error <= 0.01
        if mode == "matched":
            joined["norm_accepted"] &= joined.norm_match_max_relative_error <= 0.01
        pairs.append(joined)
    paired = pd.concat(pairs, ignore_index=True)
    paired.to_csv(out / "random_comparison_pairs.csv", index=False)
    aggregate(
        paired,
        ["task", "site", "rank", "comparison_mode"],
        [
            "oriented_advantage",
            "absolute_advantage",
            "norm_accepted",
            "random_norm_error",
        ],
    ).to_csv(out / "random_comparisons_all.csv", index=False)
    aggregate(
        paired[paired.norm_accepted],
        ["task", "site", "rank", "comparison_mode"],
        [
            "oriented_advantage",
            "absolute_advantage",
        ],
    ).to_csv(out / "random_comparisons_norm_qualified.csv", index=False)
    # Recompute all region norms in float64 from the frozen gradients.
    region_rows = []
    for ji, job in enumerate(gm["jobs"]):
        for site in gm["sites"]:
            cache = np.load(a.gradient_dir / "private_cache" / f"j{ji:02d}_{site}.npz")
            for row in region_derivative_stats(
                cache["state"], cache["gradient"], job["regions"]
            ):
                region_rows.append(
                    {k: job[k] for k in ("case_id", "author_id", "task")}
                    | {"site": site}
                    | row
                )
    region = pd.DataFrame(region_rows)
    region.to_csv(out / "gradient_regions_recomputed.csv", index=False)
    aggregate(
        region,
        ["task", "site", "region"],
        [
            "gradient_energy_fraction",
            "gradient_norm_per_sqrt_token",
            "gradient_norm",
            "tokens",
        ],
    ).to_csv(out / "gradient_region_summary.csv", index=False)
    sums = region.groupby(["case_id", "task", "site"]).gradient_energy_fraction.sum()
    if np.max(np.abs(sums - 1)) > 1e-10:
        raise ValueError("Gradient region partition does not sum to one")
    geometry = pd.read_csv(a.parent_dir / "geometry.csv")
    aggregate(
        geometry,
        ["site"],
        [
            "target_gradient_rank4_energy",
            "rank2_subspace_overlap",
            "rank4_subspace_overlap",
        ],
    ).to_csv(out / "subspace_geometry.csv", index=False)
    access = frame[
        (frame.family != "current_subspace") & (frame.method == "target_gradient")
    ].copy()
    access["dose"] = access.alpha.abs()
    signs = ["case_id", "author_id", "task", "site", "family", "dose"]
    positive = access[access.alpha > 0].set_index(signs)
    negative = access[access.alpha < 0].set_index(signs)
    if not positive.index.equals(negative.index):
        negative = negative.reindex(positive.index)
    asymmetry = (
        positive[["delta", "predicted_delta"]] + negative[["delta", "predicted_delta"]]
    ).reset_index()
    asymmetry["abs_even_effect"] = asymmetry.delta.abs()
    asymmetry["rounding_linear_even"] = asymmetry.predicted_delta
    aggregate(
        asymmetry,
        ["task", "site", "family", "dose"],
        [
            "delta",
            "abs_even_effect",
            "rounding_linear_even",
        ],
    ).to_csv(out / "signed_dose_asymmetry.csv", index=False)
    collateral = frame.pivot(
        index=[
            "case_id",
            "author_id",
            "rule",
            "site",
            "family",
            "method",
            "rank",
            "alpha",
        ],
        columns="task",
        values="delta",
    ).reset_index()
    collateral["target_abs"] = collateral.disposition.abs()
    collateral["collateral_abs"] = collateral.current_guidance.abs()
    aggregate(
        collateral,
        ["site", "family", "method", "rank", "alpha"],
        [
            "target_abs",
            "collateral_abs",
        ],
    ).to_csv(out / "collateral_effects.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "parent_identity": parent["identity"],
        "gradient_identity": gm["identity"],
        "code_sha256": code_fingerprint(ROOT),
        "capture_runtime": gs,
        "intervention_runtime": status,
        "authors": 6,
        "comments": 12,
        "norm_qualified_comparisons": int(paired.norm_accepted.sum()),
        "all_comparisons": len(paired),
        "maximum_random_row_match_error": float(
            frame.norm_match_max_relative_error.max()
        ),
        "external_labels_used": False,
        "limitations": [
            "Local derivatives in one pinned coordinate system are not causal variance fractions",
            "Sensitivity direction selected from target output; no independent personality discovery",
            "Six author clusters with many repeated interventions; descriptive unadjusted intervals",
            "First-order prediction tested against actual BF16 perturbations; numerical floors retained",
            "Historical-position access is not necessity or exhaustive mediation of natural history",
            "No independent mechanism-holdout or external personality result in this package",
        ],
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
