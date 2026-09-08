#!/usr/bin/env python3
"""M22B: apply frozen CPU readers to separate-author natural evidence edits."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.nonlinear_readout import METHODS
from grain_research.vector_mechanisms.readout_cache_cpu import (
    BASE,
    corr,
    pooled_features,
    read,
    sha,
    write,
)

CONDITIONS = ("full", "e1", "e2", "joint")


def natural_interaction_parts(full, first, second, joint, parallelogram):
    """Separate reader curvature from feature-map closure by an exact identity."""
    full, first, second, joint, parallelogram = [
        np.asarray(v, float) for v in (full, first, second, joint, parallelogram)
    ]
    if len({v.shape for v in (full, first, second, joint, parallelogram)}) != 1:
        raise ValueError("Paired reader values required")
    return {
        "natural": joint - first - second + full,
        "reader": parallelogram - first - second + full,
        "feature": joint - parallelogram,
    }


def panel_data(folder):
    """Read fixed S1 edits/embeddings without loading a neural library or labels."""
    manifest = read(folder / "manifest.json")
    material_path, encoder_path = (
        folder / "private_material.json",
        folder / "private_encoder.npz",
    )
    if (
        sha(material_path) != manifest["material_sha256"]
        or sha(encoder_path) != manifest["encoder_sha256"]
    ):
        raise ValueError("S1 frozen material changed")
    materials = read(material_path)
    with np.load(encoder_path, allow_pickle=False) as z:
        encoded, poles = z["comment_vectors"], z["poles"]
    metadata, targets, comments = [], [], []
    for path in sorted((folder / "private_packets").glob("panel_*.json")):
        packet = read(path)
        row = materials[packet["material_row"]]
        if any(packet[k] != row[k] for k in ("author_id", "panel_index", "view")):
            raise ValueError("S1 author/view join changed")
        if sha(path.with_suffix(".npz")) != packet["state_sha256"]:
            raise ValueError("S1 teacher packet changed")
        vectors = encoded[packet["material_row"]]
        e1, e2 = set(row["e1"]), set(row["e2"])
        if e1 & e2:
            raise ValueError("S1 edit sets must remain disjoint")
        keep = {
            "full": list(range(12)),
            "e1": [i for i in range(12) if i not in e1],
            "e2": [i for i in range(12) if i not in e2],
            "joint": [i for i in range(12) if i not in e1 | e2],
        }
        for q, query in enumerate(("TF_cont", "neuroticism")):
            scores = packet["queries"][query]["readouts"]
            metadata.append(
                {
                    "author_id": row["author_id"],
                    "view": row["view"],
                    "query": query,
                    "q": q,
                }
            )
            targets.append(
                [
                    scores[c]["conditional_score"] if c in scores else np.nan
                    for c in CONDITIONS
                ]
            )
            comments.append([vectors[keep[c]] for c in CONDITIONS])
    meta = pd.DataFrame(metadata)
    if meta.duplicated(["author_id", "view", "query"]).any():
        raise ValueError("Duplicate S1 observation")
    if (
        not (meta.groupby(["author_id", "query"]).size() == 2).all()
        or meta.author_id.nunique() != 24
    ):
        raise ValueError("Require the complete24-author two-view panel")
    return meta, np.asarray(targets), comments, poles


def author_interval(y, p, draws):
    """Bootstrap already-aggregated authors, not dependent view rows."""
    rng = np.random.default_rng(731)
    values = []
    for _ in range(draws):
        ix = rng.integers(len(y), size=len(y))
        value = corr(y[ix], p[ix])
        if value is not None:
            values.append(value)
    return {
        "r": corr(y, p),
        "r_lower": float(np.quantile(values, 0.025)) if values else None,
        "r_upper": float(np.quantile(values, 0.975)) if values else None,
    }


def main():
    """Load locally generated readers and verified S1 data; never refit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    args = parser.parse_args()
    out, source = args.output_dir.resolve(), args.readout_dir.resolve()
    if (
        not out.is_relative_to(BASE)
        or not source.is_relative_to(BASE)
        or args.bootstrap_draws < 1
    ):
        raise ValueError("Use private local directories and a positive bootstrap count")
    if read(source / "summary.json")["status"] != "COMPLETE":
        raise ValueError("Complete M22 required")
    if (
        sha(source / "private_oof_before_outcomes.csv")
        != read(source / "oof_frozen.json")["sha256"]
    ):
        raise ValueError("M22 predictions changed")
    if (
        sha(ROOT / "grain_research/vector_mechanisms/nonlinear_readout.py")
        != read(source / "manifest.json")["module_sha256"]
    ):
        raise ValueError("Reader implementation changed")
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    panel = BASE / "multiquery_sensitivity_s1_r1"
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_M22B_NATURAL_EDIT_CONNECTION_V1",
            "script_sha256": sha(__file__),
            "plan_sha256": sha(ROOT / "docs/GRAIN_NONLINEAR_EDIT_CONNECTION_M22B.md"),
            "readout_manifest_sha256": sha(source / "manifest.json"),
            "panel_manifest_sha256": sha(panel / "manifest.json"),
            "new_fits": 0,
            "new_neural_calls": 0,
            "source_labels_read": False,
            "selection": "all fixed readers; M22 inner teacher CV choice",
            "aggregation": "two views for level/E1/E2; joint eight authors view0 only",
        },
    )
    meta, teacher, comments, poles = panel_data(panel)
    panel_authors = set(meta.author_id)
    rows, hashes, identity_errors = [], {}, []
    with threadpool_limits(limits=2):
        for q, query in enumerate(("TF_cont", "neuroticism")):
            mask = meta["query"] == query
            part = meta.loc[mask].reset_index(drop=True)
            indices, y = np.flatnonzero(mask), teacher[mask]
            for family in ("mean", "query"):
                features = [
                    pooled_features([comments[i][c] for i in indices], q, poles, family)
                    for c in range(4)
                ]
                features.append(features[1] + features[2] - features[0])
                predictions = {m: [] for m in (*METHODS, "inner_chosen")}
                for fold in range(5):
                    path = (
                        source
                        / "private_models"
                        / f"source_all_{family}_{query}_observed_{fold}.joblib"
                    )
                    hashes[str(path.relative_to(ROOT))] = sha(path)
                    model = joblib.load(path)
                    if panel_authors & set(model["train_authors"]):
                        raise ValueError("Edit author overlaps reader training")
                    fold_predictions = {
                        m: np.column_stack([r.predict(x) for x in features])
                        for m, r in model["readers"].items()
                    }
                    fold_predictions["inner_chosen"] = fold_predictions[model["chosen"]]
                    for m, collected in predictions.items():
                        collected.append(fold_predictions[m])
                for method, values in predictions.items():
                    p = np.mean(values, axis=0)
                    terms = natural_interaction_parts(*p.T)
                    identity_errors.append(
                        float(
                            np.max(
                                np.abs(
                                    terms["natural"]
                                    - terms["reader"]
                                    - terms["feature"]
                                )
                            )
                        )
                    )
                    for i, author in enumerate(part.author_id):
                        for condition in (*CONDITIONS, "interaction"):
                            if condition == "interaction":
                                target = y[i, 3] - y[i, 1] - y[i, 2] + y[i, 0]
                                prediction = terms["natural"][i]
                            else:
                                c = CONDITIONS.index(condition)
                                target = y[i, c] if c == 0 else y[i, c] - y[i, 0]
                                prediction = p[i, c] if c == 0 else p[i, c] - p[i, 0]
                            if not np.isfinite(target):
                                continue
                            rows.append(
                                {
                                    "author_id": author,
                                    "view": int(part.iloc[i]["view"]),
                                    "query": query,
                                    "family": family,
                                    "method": method,
                                    "condition": condition,
                                    "teacher": float(target),
                                    "prediction": float(prediction),
                                    "reader_interaction": float(terms["reader"][i]),
                                    "feature_interaction": float(terms["feature"][i]),
                                }
                            )
    frame = pd.DataFrame(rows)
    keys = ["query", "family", "method", "condition", "author_id"]
    counts = frame.groupby(keys).size().reset_index(name="views")
    expected_views = np.where(counts["condition"].isin(["joint", "interaction"]), 1, 2)
    if not np.array_equal(counts["views"], expected_views):
        raise ValueError("Expected two regular views or one frozen joint view")
    authors = frame.groupby(keys, as_index=False)[
        ["teacher", "prediction", "reader_interaction", "feature_interaction"]
    ].mean()
    metrics = []
    for keys, group in authors.groupby(["query", "family", "method", "condition"]):
        y, p = group.teacher.to_numpy(), group.prediction.to_numpy()
        zero = np.mean(y**2)
        metrics.append(
            {
                "query": keys[0],
                "family": keys[1],
                "method": keys[2],
                "condition": keys[3],
                "authors": len(group),
                **author_interval(y, p, args.bootstrap_draws),
                "mse": float(np.mean((y - p) ** 2)),
                "mse_over_zero": float(np.mean((y - p) ** 2) / zero)
                if zero > 1e-14
                else None,
                "sign_agreement": float(np.mean(np.sign(y) == np.sign(p))),
                "reader_interaction_rms": float(
                    np.sqrt(np.mean(group.reader_interaction**2))
                ),
                "feature_interaction_rms": float(
                    np.sqrt(np.mean(group.feature_interaction**2))
                ),
            }
        )
    table = pd.DataFrame(metrics)
    table.to_csv(out / "metrics.csv", index=False)
    frame.to_csv(out / "private_view_predictions.csv", index=False)
    authors.to_csv(out / "private_author_predictions.csv", index=False)
    write(out / "private_model_hashes.json", hashes)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "seconds": time.monotonic() - start,
            "new_fits": 0,
            "new_neural_calls": 0,
            "external_labels_read": False,
            "training_author_overlap": 0,
            "max_identity_error": max(identity_errors),
            "metrics": table.astype(object)
            .where(pd.notna(table), None)
            .to_dict("records"),
        },
    )
    (out / "report.md").write_text(
        "# M22B natural-edit connection\n\nNo new fits, labels or neural calls. S1 authors excluded from M22 training.\nNatural interaction = readout interaction + feature-map closure.\n\n"
        + table.to_markdown(index=False, floatfmt=".4f")
        + "\n",
        encoding="utf-8",
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
