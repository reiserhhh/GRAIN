#!/usr/bin/env python3
"""M28/M31: nine exact function cards, transport tests and cross-path ledger."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.direct_traits import AXES, BIG5
from grain_research.vector_mechanisms.functional_profiles import (
    FixedAuthorCoordinates,
    FunctionalQuery,
    profile_pair_auc,
)
from grain_research.vector_mechanisms.readout_cache_cpu import (
    BASE,
    corr,
    read,
    sha,
    write,
)
from scripts.analyze_grain_functional_covariance import PARTS, covariance_ledger
from scripts.analyze_grain_functional_profiles import paired_metrics, response_table

M25 = BASE / "shared_author_profiles_m25_v1"
M8 = BASE / "complete_level_response_v1"
QUERIES = (*BIG5, *AXES)


def load(out):
    """Verify old and newly completed query states; never open human outcomes."""
    m = read(out / "manifest.json")
    if read(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("M28 collection is not complete")
    if (
        sha(out / "reference.npz")
        != read(out / "reference_manifest.json")["array_sha256"]
    ):
        raise ValueError("New functions changed")
    with np.load(M8 / "reference.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k.split("__")[0] in QUERIES}
    with np.load(out / "reference.npz", allow_pickle=False) as z:
        arrays.update(dict(z))
    rows, hashes = [], {}
    for folder, digits in ((out, 4), (M25, 3)):
        manifest = read(folder / "manifest.json")
        if sha(folder / "private_design.json") != manifest["input_sha256"]:
            raise ValueError("State design changed")
        for d in read(folder / "private_design.json"):
            if d["query"] not in QUERIES:
                continue
            path = folder / "private_packets" / f"cell_{d['index']:0{digits}d}.json"
            meta = read(path)
            if any(
                meta[k] != d[k]
                for k in ("author_id", "view", "query", "condition", "role")
            ):
                raise ValueError("State metadata join mismatch")
            if sha(path.with_suffix(".npz")) != meta["array_sha256"]:
                raise ValueError("State array changed")
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
                rows.append((meta, z["P"].astype(float), z["Q"].astype(float)))
            hashes[str(path.relative_to(ROOT))] = sha(path)
    for p, digest in m["inputs_sha256"].items():
        if sha(ROOT / p) != digest:
            raise ValueError("Old reference or provenance changed")
    if len(rows) != 1296:
        raise ValueError("Expected 24 authors x 2 views x 3 conditions x 9 queries")
    functions = {
        q: FunctionalQuery(
            *(arrays[f"{q}__{key}"] for key in ("Z", "h", "G", "g", "b"))
        )
        for q in QUERIES
    }
    return rows, functions, arrays, hashes


def summarize_states(rows, functions):
    """Apply the frozen additive score formula; no fitted scale or clipping."""
    parts = []
    for d, P, Q in rows:
        r = {
            k: d[k]
            for k in ("author_id", "view", "query", "condition", "role", "native_score")
        }
        r.update(functions[d["query"]].parts(P, Q))
        r["question_tokens"] = len(P)
        parts.append(r)
    frame = pd.DataFrame(parts)
    if frame.duplicated(["author_id", "view", "query", "condition"]).any():
        raise ValueError("Duplicate expanded state")
    return frame, response_table(parts)


def quality(frame, events):
    """Treat authors as resampling units; isolate complete levels and natural edits."""
    metrics, ablations = [], []
    for q in QUERIES:
        for kind, group in (
            (
                "level",
                frame[
                    (frame["query"] == q)
                    & (frame.condition == "full")
                    & (frame.role == "evaluation")
                ],
            ),
            (
                "response",
                events[(events["query"] == q) & (events.role == "evaluation")],
            ),
        ):
            native = group.native_score.to_numpy()
            metrics.append(
                {
                    "query": q,
                    "kind": kind,
                    "native_sd": float(np.std(native)),
                    **paired_metrics(
                        native, group.score.to_numpy(), group.author_id.to_numpy()
                    ),
                }
            )
            anchor = functions_anchor = float(
                (
                    group.score - group[["mean", "configuration", "final"]].sum(axis=1)
                ).mean()
            )
            if kind == "response" and abs(functions_anchor) > 1e-8:
                raise ValueError("Response includes a nonzero intercept")
            for name, cols in {
                "mean": ["mean"],
                "configuration": ["configuration"],
                "final": ["final"],
                "P": ["mean", "configuration"],
                "P_Q": ["mean", "configuration", "final"],
            }.items():
                pred = group[cols].sum(axis=1).to_numpy() + anchor
                ablations.append(
                    {
                        "query": q,
                        "kind": kind,
                        "arm": name,
                        "r": corr(native, pred),
                        "rmse": float(np.sqrt(np.mean((native - pred) ** 2))),
                        "predicted_sd": float(np.std(pred)),
                        "native_sd": float(np.std(native)),
                    }
                )
    return pd.DataFrame(metrics), pd.DataFrame(ablations)


def covariance_atlas(events):
    """Retain signed M/C/Q/remainder sharing for every query pair and condition."""
    groups, arrays, index = {}, {}, None
    for q in QUERIES:
        group = events[events["query"] == q].copy()
        group["native_remainder"] = group.native_score - group[
            ["mean", "configuration", "final"]
        ].sum(axis=1)
        ref = (
            group[group.role == "reference"]
            .groupby(["view", "condition"])[list(PARTS)]
            .mean()
        )
        held = (
            group[group.role == "evaluation"]
            .set_index(["author_id", "view", "condition"])
            .sort_index()
        )
        centers = np.array([ref.loc[(v, c)].to_numpy() for _, v, c in held.index])
        arrays[q], groups[q] = held[list(PARTS)].to_numpy() - centers, held.index
        index = held.index.to_frame(index=False)
    if any(not groups[q].equals(groups[QUERIES[0]]) for q in QUERIES):
        raise ValueError("Unmatched cross-query conditions")
    summary, detail = [], []
    for l, r in itertools.combinations(QUERIES, 2):
        for stratum in ("pooled", "0/e1", "0/e2", "1/e1", "1/e2"):
            keep = (
                np.ones(len(index), bool)
                if stratum == "pooled"
                else (index.view.astype(str) + "/" + index.condition).to_numpy()
                == stratum
            )
            a, b = arrays[l][keep], arrays[r][keep]
            matrix, total, error = covariance_ledger(a, b)
            scale = float(np.std(a.sum(1)) * np.std(b.sum(1)))
            base = {
                "left": l,
                "right": r,
                "stratum": stratum,
                "events": int(keep.sum()),
            }
            summary.append(
                {
                    **base,
                    "native_r": corr(a.sum(1), b.sum(1)),
                    "functional_r": corr(a[:, :3].sum(1), b[:, :3].sum(1)),
                    "configuration_r": corr(a[:, 1], b[:, 1]),
                    "native_covariance": total,
                    "accounting_error": error,
                    "functional_r_units": float(matrix[:3, :3].sum() / scale)
                    if scale > 1e-14
                    else None,
                }
            )
            for i, j in itertools.product(range(4), repeat=2):
                detail.append(
                    {
                        **base,
                        "left_part": PARTS[i],
                        "right_part": PARTS[j],
                        "covariance": float(matrix[i, j]),
                        "native_r_contribution": float(matrix[i, j] / scale)
                        if scale > 1e-14
                        else None,
                    }
                )
    return pd.DataFrame(summary), pd.DataFrame(detail)


def run(source, out):
    """Write all per-query results and arrays before any external scale linkage."""
    out.mkdir(parents=True, exist_ok=False)
    rows, functions, arrays, hashes = load(source)
    frame, events = summarize_states(rows, functions)
    frame.to_csv(out / "private_parts.csv", index=False)
    events.to_csv(out / "private_responses.csv", index=False)
    write(
        out / "prediction_freeze.json",
        {
            "private_parts_sha256": sha(out / "private_parts.csv"),
            "human_labels_read": False,
        },
    )
    metrics, ablations = quality(frame, events)
    metrics.to_csv(out / "native_function_metrics.csv", index=False)
    ablations.to_csv(out / "frozen_component_ablations.csv", index=False)
    summary, detail = covariance_atlas(events)
    summary.to_csv(out / "cross_query_relations.csv", index=False)
    detail.to_csv(out / "signed_path_covariance.csv", index=False)
    np.savez_compressed(out / "nine_query_functions.npz", **arrays)
    levels = (
        frame[frame.condition == "full"]
        .pivot(index=["author_id", "view", "role"], columns="query", values="score")
        .loc[:, list(QUERIES)]
    )
    ref = (
        levels.reset_index()
        .query("role == 'reference'")
        .groupby("author_id")[list(QUERIES)]
        .mean()
    )
    norm = FixedAuthorCoordinates().fit(ref.to_numpy())
    coords = pd.DataFrame(
        norm.transform(levels.to_numpy()), columns=QUERIES, index=levels.index
    ).reset_index()
    coords.to_csv(out / "private_author_positions.csv", index=False)
    held = coords[coords.role == "evaluation"]
    a = held[held.view == 0].set_index("author_id").sort_index()
    b = held[held.view == 1].set_index("author_id").loc[a.index]
    auc, _ = profile_pair_auc(a[list(QUERIES)], b[list(QUERIES)])
    write(
        out / "author_reference.json",
        {
            "queries": list(QUERIES),
            "center": norm.center.tolist(),
            "scale": norm.scale.tolist(),
            "active": norm.active.tolist(),
        },
    )
    write(
        out / "summary.json",
        {
            "status": "COMPLETE_M28",
            "queries": list(QUERIES),
            "states": len(frame),
            "authors": frame.author_id.nunique(),
            "function_reference_authors": 2,
            "norm_reference_authors": len(ref),
            "evaluation_authors": len(a),
            "nine_level_same_author_auc": auc,
            "max_covariance_accounting_error": float(summary.accounting_error.max()),
            "human_labels_read": False,
            "new_fits": 0,
            "state_metadata_sha256": hashes,
            "functions_sha256": sha(out / "nine_query_functions.npz"),
            "observational_scope": "legacy common evidence edits, query-specific affine function transport",
        },
    )
    print(
        metrics[
            ["query", "kind", "r", "r_lower", "r_upper", "mse_over_zero", "native_sd"]
        ]
        .round(4)
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(2):
        run(args.state_dir, args.output_dir)
