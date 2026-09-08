#!/usr/bin/env python3
"""Change only function-fitting order using the frozen M12 feature bank."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.conditional_baseline import fixed_ridge
from grain_research.vector_mechanisms.level_response import correlation
from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.run_grain_conditional_baseline import FAMILIES, SOURCE
from scripts.run_grain_real_author_path import BASE, read, sha, write
from scripts.run_grain_source_remainder import covariance

PREVIOUS = BASE / "conditional_baseline_construction_v1"
OUT = BASE / "material_first_baseline_v1"
PLAN = ROOT / "docs/GRAIN_MATERIAL_FIRST_BASELINE_PLAN_V1.md"


def material_predictions(features, targets, folds):
    """Fit all views of train authors, never any view of the predicted author."""
    x, y, f = map(np.asarray, (features, targets, folds))
    if x.ndim != 3 or y.shape != x.shape[:2] or f.shape != (len(x),):
        raise ValueError(
            "Author-by-view features/targets and one fold per author required"
        )
    pred = np.empty(y.shape)
    for fold in np.unique(f):
        train, test = f != fold, f == fold
        if not train.any() or not test.any():
            raise ValueError("Nonempty disjoint author folds required")
        pred[test] = fixed_ridge(
            x[train].reshape(-1, x.shape[-1]),
            y[train].reshape(-1),
            x[test].reshape(-1, x.shape[-1]),
            alpha=1.0,
        ).reshape(test.sum(), x.shape[1])
    return pred


def main():
    """Freeze this new construction and predictions before outcome comparison."""
    if OUT.exists():
        raise ValueError("Use the completed M13 packet, do not overwrite results")
    OUT.mkdir(parents=True)
    old_manifest = read(PREVIOUS / "manifest.json")
    design = read(PREVIOUS / "private_design.json")
    score_path = PREVIOUS / "private_predictions_before_outcomes.csv"
    if sha(score_path) != read(PREVIOUS / "predictions_frozen.json")["sha256"]:
        raise ValueError("M12 predictions changed")
    files = [
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/conditional_baseline.py",
    ]
    write(
        OUT / "manifest.json",
        {
            "protocol": "GRAIN_MATERIAL_FIRST_BASELINE_V1",
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "parent_manifest_sha256": sha(PREVIOUS / "manifest.json"),
            "design_sha256": sha(PREVIOUS / "private_design.json"),
            "parent_predictions_sha256": sha(score_path),
            "alpha": 1.0,
            "new_neural_calls": 0,
            "human_outcomes_for_fitting": False,
        },
    )
    previous = pd.read_csv(score_path, dtype={"author_id": str}).set_index(
        ["author_id", "query", "method"]
    )
    output, metrics, variance_rows = [], [], []
    for query in ("TF_cont", "neuroticism"):
        rows = [r for r in design if r["query"] == query]
        folds = np.array([r["fold"] for r in rows])
        targets, bank = [], {k: [] for k in FAMILIES}
        for row in rows:
            path = SOURCE / "private_packets" / f"author_{row['index']:03d}.json"
            if sha(path) != old_manifest["sources"][str(path.relative_to(ROOT))]:
                raise ValueError("Frozen b targets changed")
            targets.append([v["anchor"] for v in read(path)["views"]])
            features = PREVIOUS / "private_features" / path.with_suffix(".npz").name
            if sha(features) != read(features.with_suffix(".json"))["arrays_sha256"]:
                raise ValueError("M12 feature bank changed")
            with np.load(features, allow_pickle=False) as a:
                for method in FAMILIES:
                    bank[method].append(a[method])
        targets = np.asarray(targets)
        between = float(np.var(targets.mean(1)))
        within = float(np.mean(np.var(targets, axis=1)))
        variance_rows.append(
            {
                "query": query,
                "between_author_means": between,
                "within_author_views": within,
                "between_share": between / (between + within),
            }
        )
        for method in FAMILIES:
            predictions = material_predictions(bank[method], targets, folds)
            by_author = predictions.mean(1)
            old = np.asarray(
                [
                    previous.loc[(r["author_id"], query, method), "anchor_hat"]
                    for r in rows
                ]
            )
            new_mse = float(np.mean((by_author - targets.mean(1)) ** 2))
            old_mse = float(np.mean((old - targets.mean(1)) ** 2))
            metrics.append(
                {
                    "query": query,
                    "method": method,
                    "authors": len(rows),
                    "views": targets.size,
                    "anchor_r": correlation(targets.mean(1), by_author),
                    "view_anchor_r_descriptive": correlation(
                        targets.ravel(), predictions.ravel()
                    ),
                    "anchor_rmse": float(np.sqrt(new_mse)),
                    "mse_over_author_first": new_mse / old_mse,
                }
            )
            for i, row in enumerate(rows):
                old_row = previous.loc[(row["author_id"], query, method)]
                output.append(
                    {
                        "author_id": row["author_id"],
                        "query": query,
                        "fold": row["fold"],
                        "method": method,
                        "anchor": float(targets[i].mean()),
                        "anchor_hat": float(by_author[i]),
                        "old_anchor_hat": float(old[i]),
                        "PQ": float(old_row.PQ),
                        "native": float(old_row.native),
                        "constructed": float(
                            by_author[i] + old_row.PQ - old_row.base_ref
                        ),
                        "old_constructed": float(old_row.constructed),
                    }
                )
    frame = pd.DataFrame(output)
    frame.to_csv(OUT / "private_predictions_before_outcomes.csv", index=False)
    write(
        OUT / "predictions_frozen.json",
        {
            "sha256": sha(OUT / "private_predictions_before_outcomes.csv"),
            "human_labels_used": False,
        },
    )
    pd.DataFrame(metrics).to_csv(OUT / "anchor_fidelity.csv", index=False)
    pd.DataFrame(variance_rows).to_csv(OUT / "target_variation.csv", index=False)
    source_path = SOURCE / "private_anchor_source_connection.csv"
    if sha(source_path) != old_manifest["source_outcome_sha256"]:
        raise ValueError("Exposed source linkage changed")
    source = pd.read_csv(source_path, dtype={"author_id": str})
    frame = frame.merge(
        source[["author_id", "query", "y_true"]],
        on=["author_id", "query"],
        validate="many_to_one",
    )
    external = []
    for (query, method), group in frame.groupby(["query", "method"]):
        for name in ("PQ", "native", "old_constructed", "constructed"):
            row = scalar_metrics(
                group.y_true.to_numpy(),
                group[name].to_numpy(),
                binary=query == "TF_cont",
            )
            row.update(
                query=query,
                method=method,
                readout=name,
                cov_y=covariance(group.y_true.to_numpy(), group[name].to_numpy()),
            )
            external.append(row)
    frame.to_csv(OUT / "private_source_connection.csv", index=False)
    pd.DataFrame(external).to_csv(OUT / "source_metrics.csv", index=False)
    write(
        OUT / "summary.json",
        {
            "status": "COMPLETE",
            "anchor_metrics": metrics,
            "target_variation": variance_rows,
            "source_metrics": external,
            "neural_calls": 0,
        },
    )
    print(pd.DataFrame(metrics).to_string(index=False))
    print(pd.DataFrame(variance_rows).to_string(index=False))


if __name__ == "__main__":
    main()
