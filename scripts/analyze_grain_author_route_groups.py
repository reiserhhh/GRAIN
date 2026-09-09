#!/usr/bin/env python3
"""NAV-06: freeze groups in one GRAIN text view, compare evidence action in another."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import (
    BASE,
    QUERIES,
    digest,
    quality,
    read_json,
    write_json,
)
from grain_research.vector_mechanisms.reference_groups import (
    distance_relation,
    group_permutation,
)

DEFAULT = BASE / "navigation_v1/NAV-06/author_groups_r1"


def prepare(out):
    """Fit two descriptive groups on reference authors' view1 only, without edits."""
    if out.exists():
        raise FileExistsError("Choose a fresh group packet")
    old = BASE / "functional_author_construction_m25_m26_v1"
    overlap = BASE / "shared_author_profiles_m25_v1/private_overlap.json"
    norms = read_json(old / "author_reference.json")
    if (
        digest(old / "author_reference.json")
        != read_json(old / "manifest.json")["author_reference_sha256"]
    ):
        raise ValueError("Old GRAIN author reference changed")
    if digest(overlap) != read_json(overlap.parent / "manifest.json")["overlap_sha256"]:
        raise ValueError("View-overlap audit changed")
    if any(r["common_exact_texts"] for r in read_json(overlap)):
        raise ValueError(
            "Independent group/action views require zero exact-text overlap"
        )
    raw = pd.read_csv(old / "private_author_raw.csv", dtype={"author_id": str})
    cols = norms["coordinates"]
    raw[cols] = (raw[cols] - norms["center"]) / norms["scale"] * norms["active"]
    given = pd.read_csv(old / "private_author_positions.csv", dtype={"author_id": str})
    ident = ["author_id", "view", "role"]
    lhs = raw.set_index(ident).sort_index()
    rhs = given.set_index(ident).sort_index()
    if not lhs.index.equals(rhs.index) or not np.allclose(
        lhs[cols], rhs[cols], atol=1e-12
    ):
        raise ValueError("Saved GRAIN coordinates do not reproduce")
    source = raw.query("view==1").sort_values("author_id")
    train = source[source.role == "reference"]
    held = source[source.role == "evaluation"]
    if (
        len(train) != 12
        or len(held) != 12
        or set(train.author_id) & set(held.author_id)
    ):
        raise ValueError("Expected disjoint reference/evaluation authors")
    out.mkdir(parents=True)
    groups, models = [], {}
    with threadpool_limits(limits=2):
        for name, features in (("level3", cols[:3]), ("profile9", cols)):
            km = KMeans(n_clusters=2, n_init=10, random_state=20260909).fit(
                train[features]
            )
            labels = km.predict(held[features])
            ordering = np.argsort(km.cluster_centers_[:, 0])
            labels = np.argsort(ordering)[labels]
            models[name] = {
                "features": features,
                "centers": km.cluster_centers_[ordering].tolist(),
                "train_inertia": float(km.inertia_),
            }
            for (_, row), label in zip(held.iterrows(), labels, strict=True):
                groups.append(
                    {"author_id": row.author_id, "partition": name, "group": int(label)}
                )
    pd.DataFrame(groups).to_csv(out / "private_groups.csv", index=False)
    held[["author_id", *cols]].to_csv(out / "private_group_view.csv", index=False)
    write_json(out / "group_models.json", models)
    write_json(
        out / "group_freeze.json",
        {
            "status": "GROUPS_FROZEN_BEFORE_RESPONSE_ACCESS",
            "files": {
                n: digest(out / n)
                for n in (
                    "private_groups.csv",
                    "private_group_view.csv",
                    "group_models.json",
                )
            },
            "source_sha256": {
                str(p.relative_to(ROOT)): digest(p)
                for p in (
                    old / "private_author_raw.csv",
                    old / "private_author_positions.csv",
                    old / "author_reference.json",
                    overlap,
                )
            },
            "script_sha256": digest(Path(__file__)),
            "human_labels_used": False,
            "group_view": 1,
            "response_view": 0,
            "reference_authors": 12,
            "evaluation_authors": 12,
            "definition": "old GRAIN functional levels primary, nine-coordinate profile secondary; not psychological classes",
            "group_count": 2,
            "seed": 20260909,
            "permutation_draws": 4999,
        },
    )
    print(
        pd.DataFrame(groups).groupby(["partition", "group"]).size().to_string(),
        flush=True,
    )


def analyze(out):
    """Measure group-associated response geometry without fitting on action outcomes."""
    m = read_json(out / "group_freeze.json")
    if m["script_sha256"] != digest(Path(__file__)):
        raise ValueError("Frozen group protocol changed")
    for n, sha in m["files"].items():
        if digest(out / n) != sha:
            raise ValueError("Frozen group data changed")
    target = out / "analysis"
    if target.exists():
        raise FileExistsError("Preserve completed group analysis")
    source = BASE / "navigation_v1/NAV-04/natural_evidence_r1"
    summary = read_json(source / "analysis/summary.json")
    if digest(source / "predictions.csv") != summary["prediction_sha256"]:
        raise ValueError("Source evidence prediction freeze changed")
    responses = pd.read_csv(
        source / "analysis/paired_responses.csv", dtype={"scene_id": str}
    )
    worlds = pd.DataFrame(read_json(source / "private_worlds.json"))
    authors = worlds[["scene_id", "author_id"]].astype(str).drop_duplicates()
    if authors.scene_id.duplicated().any() or len(authors) != 12:
        raise ValueError("Evidence scene-to-author join changed")
    responses = responses.merge(authors, on="scene_id", validate="many_to_one")
    groups = pd.read_csv(out / "private_groups.csv", dtype={"author_id": str})
    profiles = pd.read_csv(
        out / "private_group_view.csv", dtype={"author_id": str}
    ).set_index("author_id")
    models = read_json(out / "group_models.json")
    metrics, kernels, contrasts = [], [], []
    target.mkdir()
    for readout in ("native", "R_synth"):
        rows = responses[responses.readout == readout]
        matrix = rows.pivot(
            index="author_id", columns=["query", "axis"], values="score_delta"
        ).sort_index()
        wanted = [
            (q, a) for q in QUERIES for a in ("actor", "endorsement", "time", "removal")
        ]
        matrix = matrix[wanted]
        if matrix.shape != (12, 12) or matrix.isna().any().any():
            raise ValueError("Expected complete12-author/12-response matrix")
        for name, model in models.items():
            labels = (
                groups[groups.partition == name]
                .set_index("author_id")
                .loc[matrix.index, "group"]
            )
            feature = profiles.loc[matrix.index, model["features"]]
            counts = labels.value_counts()
            if len(counts) != 2:
                metrics.append(
                    {
                        "readout": readout,
                        "partition": name,
                        "status": "ONE_HELD_GROUP_NO_CONTRAST",
                    }
                )
                continue
            metrics.append(
                {
                    "readout": readout,
                    "partition": name,
                    "status": "MEASURED",
                    "n": 12,
                    "min_group_n": int(counts.min()),
                    **group_permutation(
                        matrix.to_numpy(), labels.to_numpy(), draws=4999
                    ),
                }
            )
            kernels.append(
                {
                    "readout": readout,
                    "partition": name,
                    "n": 12,
                    **distance_relation(
                        feature.to_numpy(), matrix.to_numpy(), draws=4999
                    ),
                }
            )
            for label in sorted(set(labels)):
                selected = set(labels[labels == label].index)
                for (q, axis), g in rows[rows.author_id.isin(selected)].groupby(
                    ["query", "axis"]
                ):
                    contrasts.append(
                        {
                            "readout": readout,
                            "partition": name,
                            "group": int(label),
                            "query": q,
                            "axis": axis,
                            "n": len(g),
                            "mean_delta": float(g.score_delta.mean()),
                            "mean_abs_delta": float(abs(g.score_delta).mean()),
                        }
                    )
    paired = responses[responses.readout == "native"][
        ["author_id", "query", "axis", "score_delta"]
    ].rename(columns={"score_delta": "native"})
    paired = responses[responses.readout == "R_synth"].merge(
        paired, on=["author_id", "query", "axis"], validate="one_to_one"
    )
    fidelity = []
    for name in models:
        joined = paired.merge(
            groups[groups.partition == name], on="author_id", validate="many_to_one"
        )
        for (group, q), g in joined.groupby(["group", "query"]):
            fidelity.append(
                {
                    "partition": name,
                    "group": int(group),
                    "query": q,
                    "authors": g.author_id.nunique(),
                    **quality(g.native, g.score_delta),
                }
            )
    for name, table in (
        ("group_condition_effects.csv", metrics),
        ("group_distance_relation.csv", kernels),
        ("group_response_means.csv", contrasts),
        ("group_function_fidelity.csv", fidelity),
    ):
        pd.DataFrame(table).to_csv(target / name, index=False)
    write_json(
        target / "summary.json",
        {
            "status": "COMPLETE_CROSS_VIEW_GROUP_PACKET",
            "group_freeze_sha256": digest(out / "group_freeze.json"),
            "response_sha256": digest(source / "analysis/paired_responses.csv"),
            "world_sha256": digest(source / "private_worlds.json"),
            "script_sha256": digest(Path(__file__)),
            "math_sha256": digest(
                ROOT / "grain_research/vector_mechanisms/reference_groups.py"
            ),
            "unit": "author; pairwise distances are not independent samples",
            "human_labels_used": False,
            "new_model_calls": 0,
        },
    )
    print(pd.DataFrame(metrics).round(5).to_string(index=False), flush=True)
    print(pd.DataFrame(kernels).round(5).to_string(index=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "analyze"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    a = p.parse_args()
    prepare(a.out) if a.action == "prepare" else analyze(a.out)
