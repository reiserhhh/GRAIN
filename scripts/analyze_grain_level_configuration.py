#!/usr/bin/env python3
"""Cache-only whole-value remainder and GRAIN/reference configuration connections."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.level_response import correlation, paired_quality
from scripts.analyze_grain_multiquery_sensitivity import load_panel
from scripts.run_grain_level_response import FULL, SOURCE, verify
from scripts.run_grain_multiquery_sensitivity import QUERIES
from scripts.run_grain_real_author_path import BASE, read, sha, write


def distance_relation(x, y, draws=1000):
    """Describe author-distance correspondence; permute authors rather than pairs."""
    dx, dy = pdist(x), pdist(y)
    observed = correlation(dx, dy)
    matrix = squareform(dy)
    triangle = np.triu_indices(len(y), 1)
    rng = np.random.default_rng(20260908)
    null = []
    for _ in range(draws):
        order = rng.permutation(len(y))
        value = correlation(dx, matrix[np.ix_(order, order)][triangle])
        if value is not None:
            null.append(value)
    return {
        "authors": len(x),
        "pairs": len(dx),
        "distance_r": observed,
        "two_sided_author_permutation_p": (
            1 + sum(abs(v) >= abs(observed) for v in null)
        )
        / (1 + len(null))
        if observed is not None
        else None,
    }


def main():
    """Connect fixed predictions without new fitting, neural calls or questionnaire reads."""
    out = BASE / "complete_level_response_v1"
    verify(out)
    if read(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete frozen construction required")
    _, records = load_panel(SOURCE)
    lookup = {
        (row["author_id"], row["view"]): (row, packet, states)
        for row, packet, states in records
    }
    design = read(out / "private_design.json")
    authors = [d["author_id"] for d in design if d["query"] == QUERIES[0]]
    if len(authors) != len(set(authors)) or len(authors) != 12:
        raise ValueError("Twelve distinct authors required")
    predictions = pd.read_csv(out / "private_predictions.csv")
    with np.load(out / "reference.npz", allow_pickle=False) as a:
        ref = dict(a)
    references = [
        d["author_id"]
        for d in read(FULL / "private_inputs.json")
        if d["role"] == "reference"
    ]
    center, scale = [], []
    for q in QUERIES:
        values = [lookup[u, 0][1]["queries"][q]["readouts"] for u in references]
        for kind in ("level", "natural"):
            a = np.array(
                [
                    v["full"]["conditional_score"]
                    if kind == "level"
                    else v["e1"]["conditional_score"] - v["full"]["conditional_score"]
                    for v in values
                ]
            )
            center.append(float(a.mean()))
            scale.append(max(float(a.std()), 1e-6))
    config, native, other, rows, anchors = [], [], [], [], []
    overlap = 0
    for i, uid in enumerate(authors):
        row0, _, _ = lookup[uid, 0]
        row1, next_packet, _ = lookup[uid, 1]
        overlap += len(set(row0["texts"]) & set(row1["texts"]))
        predicted, actual, view1 = [], [], []
        for q in QUERIES:
            for kind in ("level", "natural"):
                hit = predictions[
                    (predictions["index"] == i)
                    & (predictions["query"] == q)
                    & (predictions.target == kind)
                    & (predictions.method == "PQ")
                ]
                if len(hit) != 1:
                    raise ValueError("Ambiguous author/query prediction")
                predicted.append(float(hit.predicted.iloc[0]))
                actual.append(float(hit.actual.iloc[0]))
                scores = next_packet["queries"][q]["readouts"]
                view1.append(
                    scores["full"]["conditional_score"]
                    if kind == "level"
                    else scores["e1"]["conditional_score"]
                    - scores["full"]["conditional_score"]
                )
            packet = read(out / "private_packets" / f"unit_{i:02d}_{q}.json")
            b = float(ref[f"{q}__b"])
            centered = predicted[-2] - b
            actual_anchor = packet["anchor_score"] - packet["full"]
            baseline_shift = packet["anchor_score"] - b
            anchor_error = actual_anchor + centered
            if not np.isclose(
                packet["full"] - predicted[-2],
                baseline_shift - anchor_error,
                atol=1e-12,
                rtol=0,
            ):
                raise ValueError("Anchor baseline/transport error ledger failed")
            anchors.append(
                {
                    "index": i,
                    "query": q,
                    "baseline_shift": baseline_shift,
                    "negative_transport_error": -anchor_error,
                    "full": packet["full"],
                    "centered_readout": centered,
                    "total_remainder": packet["full"] - predicted[-2],
                }
            )
        config.append(predicted)
        native.append(actual)
        other.append(view1)
    config, native, other = map(np.asarray, (config, native, other))
    for j, q in enumerate(QUERIES):
        for k, kind in enumerate(("level", "natural")):
            c = j * 2 + k
            for name, values in (
                ("constructed_view0", config),
                ("native_view0", native),
            ):
                rows.append(
                    dict(
                        query=q,
                        target=kind,
                        method=name,
                        **paired_quality(other[:, c], values[:, c], center[c]),
                    )
                )
    profiles = BASE / "vm05_profile_material_r1/private_profile_features.npz"
    with np.load(profiles, allow_pickle=False) as a:
        ids = {u: i for i, u in enumerate(a["author_ids"].tolist())}
        indices = [ids[u] for u in authors]
        grain = a["cue_mask_whitened"][indices].astype(float)
    coordinates = {
        "constructed_view0": (config - center) / scale,
        "native_view0": (native - center) / scale,
        "native_view1": (other - center) / scale,
        "grain_36_comment_overlap": grain,
    }
    relations = []
    for a, b in (
        ("constructed_view0", "native_view0"),
        ("constructed_view0", "native_view1"),
        ("native_view0", "native_view1"),
        ("grain_36_comment_overlap", "constructed_view0"),
        ("grain_36_comment_overlap", "native_view0"),
    ):
        relations.append(
            dict(left=a, right=b, **distance_relation(coordinates[a], coordinates[b]))
        )
    ap = pd.DataFrame(anchors)
    component = []
    for q, g in ap.groupby("query"):
        y = g.full.to_numpy()
        for column in (
            "centered_readout",
            "baseline_shift",
            "negative_transport_error",
            "total_remainder",
        ):
            v = g[column].to_numpy()
            component.append(
                {
                    "query": q,
                    "component": column,
                    "mean": float(v.mean()),
                    "rms": float(np.sqrt(np.mean(v**2))),
                    "covariance_with_full": float(
                        np.mean((v - v.mean()) * (y - y.mean()))
                    ),
                    "correlation_with_full": correlation(v, y),
                }
            )
    ap.to_csv(out / "private_anchor_ledger.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "cross_view_metrics.csv", index=False)
    pd.DataFrame(relations).to_csv(out / "configuration_distances.csv", index=False)
    pd.DataFrame(component).to_csv(out / "anchor_component_metrics.csv", index=False)
    np.savez_compressed(
        out / "private_author_configuration.npz",
        author_ids=np.asarray(authors),
        center=center,
        scale=scale,
        **coordinates,
    )
    write(
        out / "configuration_summary.json",
        {
            "status": "COMPLETE",
            "neural_calls": 0,
            "labels_read": False,
            "authors": 12,
            "profile_sha256": sha(profiles),
            "reference_authors": 2,
            "normalization": "two-reference native level/response SD, floor 1e-6; provisional norm, not population standardization",
            "grain_scope": "frozen geometry, 36 comments including mechanism material; not independent text validation",
            "view0_view1_duplicate_texts": overlap,
            "cross_view": rows,
            "relations": relations,
            "anchor_components": component,
        },
    )
    print(pd.DataFrame(relations).to_string(index=False))
    print(pd.DataFrame(component).to_string(index=False))


if __name__ == "__main__":
    main()
