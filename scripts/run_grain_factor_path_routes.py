#!/usr/bin/env python3
"""GFP01/03/04: frozen GRAIN factors, all-nine query links and source-only CV."""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.cohort import ID_SEED
from grain_research.vector_mechanisms.direct_traits import AXES, BIG5
from grain_research.vector_mechanisms.factor_catalog import association, bh_adjust
from grain_research.vector_mechanisms.factor_path_routes import (
    ARMS,
    SEED,
    aligned_directions,
    design_matrix,
    historical_overlap,
    metrics,
    nested_predictions,
    paired_delta,
)
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from suica_core.v7_observations import ObservationSpec, canonicalize_comments
from suica_core.v7_operator_controls import score_operator_from_comments
from suica_core.v7_psychometric import read_factor_bundle

PLAN = ROOT / "docs/GRAIN_FACTOR_PATH_ALL_ROUTES_PLAN_V1.md"
FROZEN = ROOT / "results/v7_operator_boundary_audit/e0_full_20260714"
SOURCE = BASE / "all_factor_source_m29_v1"
MATERIAL = BASE / "source_remainder_s9_r1/private_material.json"
LEVEL = BASE / "complete_level_response_v1"
OPERATORS = ("whole", "native", "fixed_128_cross")


def factor_names():
    """Keep factor identities separate across observation operators."""
    return [f"{op}:F{i:02d}" for op in OPERATORS for i in range(1, 5)]


def prepare(out):
    """Bind source materials, frozen factor provenance and disjoint native panel."""
    if out.exists():
        raise ValueError("Use a new output directory; never overwrite a packet")
    if sha(MATERIAL) != read(MATERIAL.parent / "manifest.json")["material_sha256"]:
        raise ValueError("Source material changed")
    sm = read(SOURCE / "manifest.json")
    if read(SOURCE / "external_summary.json")["status"] != "COMPLETE_M29":
        raise ValueError("All-nine source functions incomplete")
    if (
        sha(SOURCE / "private_predictions.csv")
        != read(SOURCE / "prediction_freeze.json")["sha256"]
    ):
        raise ValueError("Frozen function scores changed")
    if sha(SOURCE / "private_design.json") != sm["input_sha256"]:
        raise ValueError("Source design changed")
    rows = read(MATERIAL)
    ids = {r["author_id"] for r in rows}
    if len(rows) != 277 or len(ids) != 277:
        raise ValueError("Expected fixed 277-author source")
    previous = pd.read_csv(FROZEN / "scores_native.csv", dtype={"user_id": str})
    overlap = historical_overlap(previous.user_id, ids, seed=ID_SEED)
    natural = [
        d
        for d in read(LEVEL / "private_design.json")
        if d["query"] in ("TF_cont", "neuroticism")
    ]
    if {d["author_id"] for d in natural} & ids or len(natural) != 24:
        raise ValueError("Native intervention panel must be 12 disjoint authors")
    if historical_overlap(
        previous.user_id, {d["author_id"] for d in natural}, seed=ID_SEED
    ):
        raise ValueError("Native panel overlaps historical factor sample")
    functions = Path(sm["functions"]) / "nine_query_functions.npz"
    if not functions.is_absolute():
        functions = ROOT / functions
    if sha(functions) != sm["functions_sha256"]:
        raise ValueError("Functional coefficients changed")
    files = [
        MATERIAL,
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/factor_path_routes.py",
        FROZEN / "artifacts/common_source_comment_representation.joblib",
        FROZEN / "scores_native.csv",
        SOURCE / "private_design.json",
        SOURCE / "private_predictions.csv",
        SOURCE / "private_external_links.csv",
        functions,
        LEVEL / "private_design.json",
        LEVEL / "manifest.json",
    ] + [FROZEN / "operator_bundles" / f"{op}_factor_bundle.json" for op in OPERATORS]
    out.mkdir(parents=True)
    write(out / "private_excluded_authors.json", sorted(overlap))
    with zipfile.ZipFile(
        out / "code_snapshot.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in [
            PLAN,
            Path(__file__),
            ROOT / "grain_research/vector_mechanisms/factor_path_routes.py",
        ]:
            archive.write(path, str(path.relative_to(ROOT)))
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_FACTOR_PATH_ALL_ROUTES_V1",
            "seed": SEED,
            "authors": len(ids - overlap),
            "input_authors": len(ids),
            "excluded_historical_authors": len(overlap),
            "exclusion_sha256": sha(out / "private_excluded_authors.json"),
            "identity_join": "stable_key(raw_user_id, cohort.ID_SEED)[:24]",
            "views": 3,
            "queries": 9,
            "factors": 12,
            "reference_author_overlap": 0,
            "native_source_author_overlap": 0,
            "native_units": 24,
            "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "functions": str(functions.relative_to(ROOT)),
            "meps_used_for_fitting": False,
            "status_scope": "exploratory already-exposed source CV; disjoint factor reference",
            "factor_reference": {
                "discovery": 101,
                "calibration": 29,
                "confirmation": 50,
            },
        },
    )
    write(out / "status.json", {"status": "PREPARED", "completed_routes": []})


def score_material(out):
    """Run real frozen factor posteriors on each of the exact upstream views."""
    rows, mapping = [], []
    excluded = set(read(out / "private_excluded_authors.json"))
    for r in read(MATERIAL):
        if r["author_id"] in excluded:
            continue
        for view, v in enumerate(r["views"]):
            key = f"{r['author_id']}::view{view}"
            mapping.append(
                {
                    "user_id": key,
                    "author_id": r["author_id"],
                    "view": view,
                    "source_group": r["source_group"],
                    "input_comments": len(v["texts"]),
                }
            )
            for text in v["texts"]:
                rows.append({"author": key, "body": text})
    raw = pd.DataFrame(rows)
    comments = canonicalize_comments(
        raw, order_col=None, condition_col=None, min_tokens=12
    )
    comments["split"] = "confirmation"
    meta = pd.DataFrame(mapping)
    runtime = joblib.load(
        FROZEN / "artifacts/common_source_comment_representation.joblib"
    )
    result, support = meta.copy(), []
    for op in OPERATORS:
        bundle = read_factor_bundle(
            FROZEN / "operator_bundles" / f"{op}_factor_bundle.json"
        )
        units, _, scores = score_operator_from_comments(
            comments,
            spec=ObservationSpec(**bundle.operator),
            representation=runtime,
            bundle=bundle,
        )
        if len(scores) != len(meta) or not scores.score_status.eq("SCORED").all():
            raise ValueError(f"Incomplete supported factor observations: {op}")
        cols = [f"SU7-FC-{i:04d}@v1_z" for i in range(1, 5)]
        renamed = scores[["user_id", *cols]].rename(
            columns={c: f"{op}:F{i:02d}" for i, c in enumerate(cols, 1)}
        )
        result = result.merge(renamed, on="user_id", validate="one_to_one")
        scores.assign(operator=op).to_csv(
            out / f"private_support_{op}.csv", index=False
        )
        support.append(
            {
                "operator": op,
                "views_scored": len(scores),
                "units": len(units),
                "median_units": float(scores.n_units.median()),
                "median_tokens": float(scores.n_tokens.median()),
                "min_units": int(scores.n_units.min()),
                "max_units": int(scores.n_units.max()),
                "bundle_sha256": sha(
                    FROZEN / "operator_bundles" / f"{op}_factor_bundle.json"
                ),
            }
        )
    result.to_csv(out / "private_factor_views.csv", index=False)
    averaged = (
        result.groupby(["author_id", "source_group"], sort=True)[factor_names()]
        .mean()
        .reset_index()
    )
    averaged.to_csv(out / "private_factor_authors.csv", index=False)
    pd.DataFrame(support).to_csv(out / "material_support.csv", index=False)
    reliability = []
    for factor in factor_names():
        for group, subset in result.groupby("source_group"):
            pivot = subset.pivot(index="author_id", columns="view", values=factor)
            for a, b in ((0, 1), (0, 2), (1, 2)):
                reliability.append(
                    {
                        "factor": factor,
                        "source_group": group,
                        "view_a": a,
                        "view_b": b,
                        **association(pivot[a], pivot[b]),
                    }
                )
    pd.DataFrame(reliability).to_csv(out / "factor_view_consistency.csv", index=False)
    write(
        out / "factor_freeze.json",
        {
            "sha256": sha(out / "private_factor_authors.csv"),
            "views_sha256": sha(out / "private_factor_views.csv"),
            "labels_read_during_scoring": False,
            "input_comments": len(raw),
            "retained_comments": len(comments),
            "comment_personality_term_rate": float(
                comments.has_personality_term.mean()
            ),
            "additional_masking": False,
            "raw_timestamp_available": False,
        },
    )
    return averaged


def links_and_predictions(out, grain):
    """Complete all association cells and train-only nested source prediction arms."""
    data = pd.read_csv(SOURCE / "private_external_links.csv", dtype={"author_id": str})
    if data.duplicated(["author_id", "query"]).any() or set(data["query"]) != {
        *BIG5,
        *AXES,
    }:
        raise ValueError("Incomplete source query matrix")
    expected = data[data.author_id.isin(set(grain.author_id))].copy()
    joined = expected.merge(
        grain, on=["author_id", "source_group"], validate="many_to_one"
    )
    if len(joined) != len(expected) or len(set(joined.author_id)) != len(grain):
        raise ValueError("Missing author links")
    frozen = pd.read_csv(SOURCE / "private_predictions.csv", dtype={"author_id": str})
    check = joined.merge(
        frozen,
        on=["author_id", "query", "source_group"],
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    for term in ("score", "mean", "configuration", "final"):
        if not np.allclose(check[term + "_a"], check[term + "_b"], rtol=0, atol=1e-12):
            raise ValueError("Criterion join modified frozen path scores")
    joined.to_csv(out / "private_joined.csv", index=False)
    path, external, comparisons, all_metrics, oof, choices = [], [], [], [], [], []
    for query, sub in joined.groupby("query", sort=True):
        sub = sub.sort_values("author_id").reset_index(drop=True)
        y = sub.y_true.to_numpy(float)
        binary = query in AXES
        for factor in factor_names():
            external.append(
                {
                    "factor": factor,
                    "query": query,
                    **association(sub[factor], y, binary=binary),
                }
            )
            for term in ("score", "mean", "configuration", "final", "prediction"):
                path.append(
                    {
                        "factor": factor,
                        "query": query,
                        "term": term,
                        **association(sub[factor], sub[term]),
                    }
                )
        g, p = (
            sub[factor_names()].to_numpy(),
            sub[["mean", "configuration", "final"]].to_numpy(),
        )
        preds = {
            "frozen_function": sub.score.to_numpy(),
            "native": sub.prediction.to_numpy(),
        }
        permutation = np.random.default_rng(SEED).permutation(len(y))
        for arm in ARMS:
            x = design_matrix(g, p, arm)
            pred, fold, selected = nested_predictions(x, y, sub.author_id)
            preds[arm] = pred
            perm_pred, _, _ = nested_predictions(x, y[permutation], sub.author_id)
            all_metrics.append(
                {
                    "query": query,
                    "arm": arm,
                    "control": "permuted_labels",
                    **metrics(y[permutation], perm_pred, binary=binary),
                }
            )
            choices.extend({"query": query, "arm": arm, **s} for s in selected)
            oof.extend(
                {
                    "author_id": a,
                    "query": query,
                    "arm": arm,
                    "fold": int(f),
                    "y_true": float(t),
                    "prediction": float(v),
                    "permuted_y": float(py),
                    "permuted_prediction": float(pp),
                }
                for a, f, t, v, py, pp in zip(
                    sub.author_id, fold, y, pred, y[permutation], perm_pred, strict=True
                )
            )
        for arm, pred in preds.items():
            all_metrics.append(
                {
                    "query": query,
                    "arm": arm,
                    "control": "real_labels",
                    **metrics(y, pred, binary=binary),
                }
            )
        for a, b in (("joint", "path"), ("joint", "grain"), ("interaction", "joint")):
            comparisons.append(
                {
                    "query": query,
                    "candidate": a,
                    "reference": b,
                    **paired_delta(y, preds[a], preds[b], binary=binary),
                }
            )
        print({"stage": "nested_cv", "query": query, "authors": len(y)}, flush=True)
    for rows, filename, group in (
        (path, "factor_path_associations.csv", "term"),
        (external, "factor_criterion_associations.csv", None),
    ):
        table = pd.DataFrame(rows)
        table["q_bh"] = (
            table.groupby(group).p.transform(bh_adjust) if group else bh_adjust(table.p)
        )
        table.to_csv(out / filename, index=False)
    pd.DataFrame(all_metrics).to_csv(out / "prediction_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "paired_prediction_deltas.csv", index=False)
    pd.DataFrame(oof).to_csv(out / "private_nested_predictions.csv", index=False)
    pd.DataFrame(choices).to_csv(out / "inner_selections.csv", index=False)


def prepare_directions(out, grain):
    """Fit two query-specific activation alignments without questionnaire labels."""
    design = read(SOURCE / "private_design.json")
    native = [f"native:F{i:02d}" for i in range(1, 5)]
    stored, summaries = {}, []
    source = grain.set_index("author_id")
    for query in ("TF_cont", "neuroticism"):
        ids, states, hashes = [], [], {}
        for d in design:
            if d["query"] != query or d["author_id"] not in source.index:
                continue
            path = SOURCE / "private_packets" / f"unit_{d['index']:04d}.json"
            packet = read(path)
            if packet["author_id"] != d["author_id"] or packet["query"] != query:
                raise ValueError("Source state identity changed")
            npz = path.with_suffix(".npz")
            if sha(npz) != packet["array_sha256"]:
                raise ValueError("Source states changed")
            with np.load(npz, allow_pickle=False) as z:
                if len(z["P"]) != 3 or len(z["Q"]) != 3:
                    raise ValueError("Require three source views")
                p, q = z["P"].astype(float).mean(0), z["Q"].astype(float).mean(0)
                states.append(np.r_[p.ravel(), q.ravel()])
            ids.append(d["author_id"])
            hashes[str(npz.relative_to(ROOT))] = packet["array_sha256"]
        real, null, center, scale = aligned_directions(
            source.loc[ids, native].to_numpy(), np.asarray(states)
        )
        for k, value in (
            ("real", real),
            ("null", null),
            ("center", center),
            ("scale", scale),
        ):
            stored[f"{query}__{k}"] = value
        norms = np.linalg.norm(real, axis=1)
        cosine = (real @ real.T) / norms[:, None] / norms[None, :]
        summaries.append(
            {
                "query": query,
                "authors": len(ids),
                "p_shape": list(p.shape),
                "q_shape": list(q.shape),
                "real_norms": norms.tolist(),
                "null_norms": np.linalg.norm(null, axis=1).tolist(),
                "direction_cosines": cosine.tolist(),
                "source_state_hashes": hashes,
            }
        )
    np.savez_compressed(out / "private_directions.npz", **stored)
    write(
        out / "direction_freeze.json",
        {
            "sha256": sha(out / "private_directions.npz"),
            "amplitude": 0.25,
            "alpha": 100.0,
            "factors": native,
            "queries": summaries,
            "human_labels_used": False,
        },
    )


def run(out):
    """Execute all CPU routes once; persist failure without a silent fallback."""
    m = read(out / "manifest.json")
    if read(out / "status.json")["status"] == "CPU_COMPLETE":
        return
    for path, digest in m["input_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError("Prepared input changed: " + path)
    if sha(out / "private_excluded_authors.json") != m["exclusion_sha256"]:
        raise ValueError("Author exclusion ledger changed")
    start = time.monotonic()
    with threadpool_limits(limits=2):
        grain = score_material(out)
        print({"stage": "factor_scoring", "authors": len(grain)}, flush=True)
        prepare_directions(out, grain)
        links_and_predictions(out, grain)
    write(
        out / "status.json",
        {
            "status": "CPU_COMPLETE",
            "completed_routes": ["GFP-01", "GFP-03", "GFP-04"],
            "GFP-02": "directions_ready_native_replay_pending",
            "seconds": time.monotonic() - start,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "run", "all"), default="all")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(BASE.resolve()):
        parser.error("Private outputs must stay under ignored mechanism results")
    if args.stage in ("prepare", "all"):
        prepare(output)
    if args.stage in ("run", "all"):
        try:
            run(output)
        except Exception as error:
            write(
                output / "error.json",
                {"error_type": type(error).__name__, "message": str(error)},
            )
            raise
