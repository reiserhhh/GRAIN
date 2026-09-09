#!/usr/bin/env python3
"""NEXT13: fit fixed scalar/component maps, freeze, then analyze source links."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.factor_catalog import association
from grain_research.vector_mechanisms.factor_path_routes import paired_delta
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from grain_research.vector_mechanisms.scalar_component import (
    ARMS,
    COMPONENTS,
    construct_oof,
)
from grain_research.vector_mechanisms.target_accounting import (
    projection_account,
    projection_draws,
)
from scripts.analyze_grain_teacher_targets import interval, verify_digest

PLAN = ROOT / "docs/GRAIN_LOGIC_SCALAR_COMPONENT_PLAN_V1.md"
READING = BASE / "logic_reading_levels_v1"
M29 = BASE / "all_factor_source_m29_v1"
S9 = BASE / "source_remainder_s9_r1"
QUERIES = ("TF_cont", "neuroticism")
KEYS = ["author_id", "query"]
CONTRASTS = (
    ("cubic_author", "affine_author"),
    ("cubic_view", "affine_view"),
    ("components_author", "affine_author"),
    ("components_view", "affine_view"),
    ("cubic_view", "cubic_author"),
    ("components_view", "components_author"),
)


def load_views():
    """Read no Y; reconcile material-level targets and the fixed fold membership."""
    paths = [
        READING / "private_source_oof.csv",
        M29 / "private_predictions.csv",
        BASE / "dynamic_path_channels_v1/private_path_views.csv",
        S9 / "private_components_before_outcomes.csv",
    ]
    verify_digest(
        paths[0], read(READING / "prediction_freeze.json")["files"][paths[0].name]
    )
    verify_digest(paths[1], read(M29 / "prediction_freeze.json")["sha256"])
    verify_digest(paths[3], read(S9 / "components_frozen.json")["sha256"])
    old = pd.read_csv(paths[0], dtype={"author_id": str})
    old = old[(old.arm == "pooled") & old["query"].isin(QUERIES)]
    if (
        old.duplicated(KEYS + ["view"]).any()
        or old.groupby(KEYS).fold.nunique().ne(1).any()
    ):
        raise ValueError("Frozen OOF membership/folds changed")
    views = pd.read_csv(paths[2], dtype={"author_id": str})
    data = old[KEYS + ["view", "fold", "actual_c"]].merge(
        views, on=KEYS + ["view"], how="left", validate="one_to_one"
    )
    if len(data) != 828 or not np.allclose(
        data.actual_c, data.configuration, rtol=0, atol=1e-12
    ):
        raise ValueError("Current reference-disjoint per-view panel changed")
    fixed = pd.read_csv(paths[1], dtype={"author_id": str}).set_index(KEYS)
    avg = data.groupby(KEYS)[["score", *COMPONENTS]].mean()
    if not np.allclose(avg, fixed.loc[avg.index, avg.columns], rtol=0, atol=1e-12):
        raise ValueError("Fixed author function replay failed")
    anchor = data.score - data[list(COMPONENTS)].sum(axis=1)
    if anchor.groupby(data["query"]).agg(lambda x: x.max() - x.min()).gt(1e-10).any():
        raise ValueError("Function anchor is not constant within query")
    packets, records = [], []
    for p in sorted((S9 / "private_packets").glob("author_*.json")):
        if p.name.startswith("._"):
            continue
        d = read(p)
        if len(d["views"]) != 3 or {v["view"] for v in d["views"]} != {0, 1, 2}:
            raise ValueError("Incomplete native observer views")
        packets.append(p)
        records.extend(
            {
                "author_id": str(d["author_id"]),
                "query": d["query"],
                "view": v["view"],
                "teacher": v["new_teacher"],
            }
            for v in d["views"]
        )
    observed = pd.DataFrame(records)
    source = pd.read_csv(paths[3], dtype={"author_id": str}).set_index(KEYS)
    means = observed.groupby(KEYS).teacher.mean()
    if (
        len(packets) != 277
        or observed.duplicated(KEYS + ["view"]).any()
        or not np.allclose(
            means, source.loc[means.index].new_teacher, rtol=0, atol=1e-12
        )
    ):
        raise ValueError("S9 per-view native replay failed")
    data = data.merge(observed, on=KEYS + ["view"], how="left", validate="one_to_one")
    if not np.isfinite(data.teacher).all() or data.groupby(KEYS).size().ne(3).any():
        raise ValueError("Native material join incomplete")
    for q, g in data.groupby("query"):
        if g.author_id.nunique() != (63 if q == "TF_cont" else 213) or set(
            g.fold
        ) != set(range(5)):
            raise ValueError("Unexpected author count or folds")
    manifest = {
        "files": {str(p.relative_to(ROOT)): sha(p) for p in paths + packets},
        "authors": 276,
        "views": len(data),
        "new_neural_calls": 0,
        "plan_sha256": sha(PLAN),
        "script_sha256": sha(__file__),
        "core_sha256": sha(
            ROOT / "grain_research/vector_mechanisms/scalar_component.py"
        ),
        "human_labels_read": False,
        "arms": list(ARMS),
        "penalty": 0.01,
    }
    return data.drop(columns="actual_c"), manifest


def construct(out):
    """Persist a complete model-only prediction freeze before source Y analysis."""
    start = time.monotonic()
    data, manifest = load_views()
    out.mkdir(parents=True, exist_ok=False)
    data.to_csv(out / "private_inputs.csv", index=False)
    predictions, models, jensen = construct_oof(data)
    if (
        len(predictions) != 5796
        or len(models) != 60
        or jensen.closure_error.max() > 1e-10
    ):
        raise ValueError("Fixed coverage or Jensen identity failed")
    predictions.to_csv(out / "private_view_predictions.csv", index=False)
    authors = (
        predictions.groupby(KEYS + ["fold", "arm"])[
            ["teacher", "prediction", "training_mean", "extrapolated"]
        ]
        .mean()
        .reset_index()
    )
    authors.to_csv(out / "private_author_predictions.csv", index=False)
    jensen.to_csv(out / "private_jensen.csv", index=False)
    write(out / "private_models.json", {"models": models})
    write(out / "manifest.json", manifest)
    write(
        out / "prediction_freeze.json",
        {
            "files": {
                name: sha(out / name)
                for name in (
                    "private_inputs.csv",
                    "private_view_predictions.csv",
                    "private_author_predictions.csv",
                    "private_jensen.csv",
                    "private_models.json",
                )
            },
            "human_labels_read": False,
            "new_neural_calls": 0,
            "model_fits": len(models),
            "max_jensen_error": float(jensen.closure_error.max()),
            "seconds": time.monotonic() - start,
        },
    )
    print(
        {
            "status": "PREDICTIONS_FROZEN",
            "authors": 276,
            "fits": len(models),
            "new_neural_calls": 0,
            "seconds": round(time.monotonic() - start, 3),
        }
    )


def analyze(out):
    """Evaluate every frozen arm; never select parameters from source criteria."""
    freeze = read(out / "prediction_freeze.json")
    if freeze["human_labels_read"] is not False:
        raise ValueError("Expected model-only prediction freeze")
    for name, digest in freeze["files"].items():
        verify_digest(out / name, digest)
    if (out / "summary.json").exists():
        raise ValueError("Preserve completed analysis")
    criterion_path = M29 / "private_external_links.csv"
    verify_digest(criterion_path, read(READING / "manifest.json")["criterion_sha256"])
    source = pd.read_csv(criterion_path, dtype={"author_id": str})
    if source.duplicated(KEYS).any():
        raise ValueError("Duplicate source criterion")
    authors = pd.read_csv(
        out / "private_author_predictions.csv", dtype={"author_id": str}
    )
    data = authors.merge(
        source[KEYS + ["y_true"]], on=KEYS, how="left", validate="many_to_one"
    )
    if not np.isfinite(data.y_true).all() or len(data) != 1932:
        raise ValueError("Missing matched source criteria")
    data.to_csv(out / "private_criterion_links.csv", index=False)
    metrics, comparisons, folds, projections = [], [], [], []
    for qi, (q, group) in enumerate(data.groupby("query", sort=True)):
        pivot = group.pivot(
            index="author_id", columns="arm", values="prediction"
        ).sort_index()
        if set(pivot) != set(ARMS) or not np.isfinite(pivot).all().all():
            raise ValueError("Complete fixed arms required")
        truth = group[group.arm == "raw_f"].set_index("author_id").loc[pivot.index]
        y, t = truth.y_true.to_numpy(), truth.teacher.to_numpy()
        rng = np.random.default_rng(20260910 + qi)
        ix = rng.integers(0, len(y), (1999, len(y)))
        for arm in ARMS:
            p = pivot[arm].to_numpy()
            rows = group[group.arm == arm].set_index("author_id").loc[pivot.index]
            mse = float(np.mean((t - p) ** 2))
            metrics.append(
                {
                    "query": q,
                    "arm": arm,
                    "n": len(y),
                    "teacher_r": association(p, t)["pearson_r"],
                    "teacher_mse": mse,
                    "teacher_mae": float(np.mean(np.abs(t - p))),
                    "mse_over_training_mean": mse
                    / float(np.mean((t - truth.training_mean) ** 2)),
                    "prediction_sd": float(p.std()),
                    "out_of_unit_range": float(np.mean((p < 0) | (p > 1))),
                    "extrapolated_view_fraction": float(rows.extrapolated.mean()),
                    **{
                        "criterion_" + k: v
                        for k, v in association(p, y, binary=q == "TF_cont").items()
                    },
                }
            )
            account = projection_account(t, p, y)
            boot = projection_draws(t, p, y, ix)
            projections.append(
                {
                    "query": q,
                    "arm": arm,
                    **account,
                    **{
                        "non_affine_" + k: v
                        for k, v in interval(boot["non_affine_contribution"]).items()
                    },
                }
            )
        for candidate, reference in CONTRASTS:
            a, b = pivot[candidate].to_numpy(), pivot[reference].to_numpy()
            gain = (t - b) ** 2 - (t - a) ** 2
            base = {
                "query": q,
                "candidate": candidate,
                "reference": reference,
                "n": len(y),
            }
            teacher_stat = paired_delta(t, a, b, draws=1999)
            gain_interval = interval(gain[ix].mean(1))
            comparisons.append(
                {
                    **base,
                    "endpoint": "teacher",
                    **teacher_stat,
                    "mse_gain": float(gain.mean()),
                    **{"mse_gain_" + k: v for k, v in gain_interval.items()},
                }
            )
            comparisons.append(
                {
                    **base,
                    "endpoint": "criterion",
                    **paired_delta(y, a, b, binary=q == "TF_cont", draws=1999),
                }
            )
            for fold in sorted(truth.fold.unique()):
                keep = truth.fold.to_numpy() == fold
                folds.append(
                    {
                        **base,
                        "fold": int(fold),
                        "n": int(keep.sum()),
                        "teacher_mse_gain": float(gain[keep].mean()),
                    }
                )
    jensen = pd.read_csv(out / "private_jensen.csv")
    rows = []
    for (q, arm), part in jensen.groupby(["query", "arm"]):
        rows.append(
            {
                "query": q,
                "arm": arm,
                "n": len(part),
                "gap_mean": float(part.direct.mean()),
                "gap_rms": float(np.sqrt(np.mean(part.direct**2))),
                "second_rms": float(np.sqrt(np.mean(part.second**2))),
                "third_rms": float(np.sqrt(np.mean(part.third**2))),
                "max_identity_error": float(part.closure_error.max()),
            }
        )
    for name, values in (
        ("metrics", metrics),
        ("paired_comparisons", comparisons),
        ("fold_comparisons", folds),
        ("projection_accounts", projections),
        ("jensen_accounts", rows),
    ):
        pd.DataFrame(values).to_csv(out / (name + ".csv"), index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "prediction_freeze_sha256": sha(out / "prediction_freeze.json"),
            "criterion_sha256": sha(criterion_path),
            "analysis_script_sha256": sha(__file__),
            "human_criterion_fits": 0,
            "queries": 2,
            "arms": len(ARMS),
            "pointwise_draws": 1999,
            "new_neural_calls": 0,
        },
    )
    print(
        pd.DataFrame(metrics)[
            [
                "query",
                "arm",
                "teacher_r",
                "teacher_mse",
                "criterion_pearson_r",
                "criterion_auc",
            ]
        ].to_string(index=False)
    )


def main():
    """Use explicit construct/analyze phases and new ignored output directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("construct", "analyze"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError("Individual models and predictions must remain ignored")
    if args.mode == "construct":
        if out.exists():
            raise ValueError("Use a new output directory")
        construct(out)
    else:
        analyze(out)


if __name__ == "__main__":
    main()
