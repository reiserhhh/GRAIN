#!/usr/bin/env python3
"""NAV-12: compare fixed contextual charts and signed actor/opportunity interactions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import (
    BASE,
    OUTPUT,
    digest,
    quality,
    read_json,
    write_json,
)


def factorial_differences(frame):
    """Return first differences and mixed finite differences, without declaring them noise."""
    rows = []
    keys = ["query", "method", "split", "scene_id", "realization"]
    for key, g in frame.groupby(keys):
        if len(g) != 4:
            raise ValueError("A complete actor-by-opportunity rectangle is required")
        v = g.set_index(["actor", "opportunity"]).score
        if not v.index.is_unique:
            raise ValueError("Duplicate factorial cell")
        sv, sa, ov, oa = [
            float(v.loc[k])
            for k in (
                ("self", "voluntary"),
                ("self", "assigned"),
                ("other", "voluntary"),
                ("other", "assigned"),
            )
        ]
        for effect, value in {
            "actor_voluntary": ov - sv,
            "actor_assigned": oa - sa,
            "opportunity_self": sa - sv,
            "opportunity_other": oa - ov,
            "actor_x_opportunity": oa - sa - ov + sv,
        }.items():
            rows.append(
                dict(zip(keys, key, strict=True)) | {"effect": effect, "delta": value}
            )
    return pd.DataFrame(rows)


def run(source, reference, out):
    """Evaluate already-frozen mean and observable-affinity gates, never outcome-fit them."""
    if out.exists():
        raise FileExistsError("Choose a new atlas analysis ID")
    predictions = source / "private_predictions.csv"
    if digest(predictions) != read_json(source / "prediction_freeze.json")["sha256"]:
        raise ValueError("Contextual prediction freeze changed")
    frame = pd.read_csv(predictions)
    worlds = pd.DataFrame(read_json(reference / "worlds.json"))
    factors = worlds[["world_id", "scene_id", "actor", "opportunity", "realization"]]
    values = frame.merge(factors, on="world_id", validate="many_to_one")
    if len(values) != len(frame):
        raise ValueError("Missing event coordinates")
    truth = pd.read_csv(reference / "native_and_function.csv").query(
        "reference=='R_synth'"
    )
    target_parts = []
    for column, name in (("score", "full_function"), ("native_score", "native")):
        part = truth[["world_id", "query", column]].rename(columns={column: "score"})
        part["method"] = name
        part = part.merge(
            worlds[["world_id", "split"]], on="world_id", validate="many_to_one"
        )
        target_parts.append(part.merge(factors, on="world_id", validate="many_to_one"))
    diffs = factorial_differences(pd.concat([values, *target_parts], ignore_index=True))
    keys = ["query", "split", "scene_id", "realization", "effect"]
    targets = (
        diffs[diffs.method == "full_function"]
        .drop(columns="method")
        .rename(columns={"delta": "target_delta"})
    )
    joined = diffs[~diffs.method.isin(["full_function", "native"])].merge(
        targets, on=keys, validate="many_to_one"
    )
    summaries = []
    for key, g in joined.groupby(["query", "method", "split", "effect"]):
        summaries.append(
            dict(zip(["query", "method", "split", "effect"], key, strict=True))
            | {
                **quality(g.target_delta, g.delta),
                "independent_scenes": g.scene_id.nunique(),
                "mean_abs_target": float(abs(g.target_delta).mean()),
                "mean_abs_predicted": float(abs(g.delta).mean()),
                "sign_agreement": float(
                    (np.sign(g.target_delta) == np.sign(g.delta)).mean()
                ),
            }
        )
    overlap = []
    for key, g in values.groupby(["query", "cut", "split"]):
        v = g.pivot(index="world_id", columns="pooling", values="score")
        gate = (
            g[g.pooling == "affinity"]
            .set_index("world_id")
            .loc[v.index, "chart0_weight"]
        )
        overlap.append(
            dict(zip(["query", "cut", "split"], key, strict=True))
            | {
                "n": len(v),
                "chart_correlation": quality(v.chart0, v.chart1)["r"],
                "mean_abs_chart_disagreement": float(abs(v.chart0 - v.chart1).mean()),
                "affinity_minus_mean_mae": float(abs(v.affinity - v["mean"]).mean()),
                "gate_weight_min": float(gate.min()),
                "gate_weight_max": float(gate.max()),
                "gate_input": "construction-reference versus current evidence-hidden cosine; no human labels",
            }
        )
    out.mkdir(parents=True)
    diffs.to_csv(out / "conditional_signed_terms.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "response_transfer.csv", index=False)
    pd.DataFrame(overlap).to_csv(out / "overlap_consistency.csv", index=False)
    write_json(
        out / "local_charts.json",
        {
            "charts_per_query_per_cut": 2,
            "cuts": [0, 3, 7],
            "gate": "softmax(8*cos(reference_mean, current_mean))",
            "gate_fit_to_outcomes": False,
            "methods": ["chart0", "chart1", "mean", "affinity"],
            "source_prediction_sha256": digest(predictions),
            "source_manifest_sha256": digest(source / "manifest.json"),
            "code_sha256": digest(Path(__file__)),
            "scope": "controlled worlds; three held scene skeletons, repeated renderings",
            "continuity": "soft weighting is continuous in finite nonzero-norm vectors; tokenization remains discrete",
        },
    )
    write_json(
        out / "summary.json",
        {
            "status": "COMPLETE_FIRST_LOCAL_ATLAS",
            "conditional_terms": len(diffs),
            "mean_vs_gate_tested": True,
            "chart_disagreement_retained": True,
            "new_model_calls": 0,
            "new_human_label_fitting": False,
        },
    )
    print(
        pd.DataFrame(summaries)
        .query("split=='held_scene' and effect=='actor_x_opportunity'")
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source", type=Path, default=BASE / "navigation_v1/NAV-10/context_worlds_r2"
    )
    p.add_argument("--reference", type=Path, default=OUTPUT)
    p.add_argument(
        "--out", type=Path, default=BASE / "navigation_v1/NAV-12/context_atlas_r1"
    )
    a = p.parse_args()
    run(a.source, a.reference, a.out)
