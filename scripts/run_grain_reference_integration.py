#!/usr/bin/env python3
"""INT01: cached, source-only GRAIN reference/response integration experiment."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.conditional_models import (
    fit_response_kernel,
    predict_response_kernel,
)
from grain_research.vector_mechanisms.conditional_navigation import response_metrics
from grain_research.vector_mechanisms.navigation import (
    BASE,
    OUTPUT,
    QUERIES,
    digest,
    load_functions,
    read_json,
    write_json,
)
from grain_research.vector_mechanisms.reference_integration import (
    condition_profile,
    fit_anchor_response,
    fit_condition_reference,
    predict_anchor_response,
)
from scripts.analyze_grain_conditional_navigation import EARLY, NATURAL, ROUTES
from scripts.collect_grain_conditional_sources import DEFAULT as SOURCE
from scripts.fit_grain_conditional_navigation import (
    response_table,
    source_arrays,
    summarize_responses,
)
from scripts.run_grain_navigation_baseline import checked_arrays
from scripts.run_grain_source_span_extension import DEFAULT as SPAN
from scripts.run_grain_source_span_extension import OLD, load_extension
from scripts.run_grain_state_transport import load_operators

DEFAULT = BASE / "reference_integration_v1/INT-01/run_r1"
PLAN = ROOT / "docs/GRAIN_REFERENCE_MECHANISM_INTEGRATION_PLAN_V1.md"
METHODS = ("linear_value", "linear_secant", "anchor_secant")


def fit_three(gram, residual, train, edges, anchors):
    """Fit only three declared formulas; do not tune against natural outcomes."""
    return {
        "linear_value": fit_response_kernel(gram, residual, train, edges),
        "linear_secant": fit_response_kernel(gram, residual, train, edges, secant=True),
        "anchor_secant": fit_anchor_response(gram, residual, train, edges, anchors),
    }


def predict(model, cross, method):
    """Use the same frozen function for both endpoint values and differences."""
    fn = (
        predict_anchor_response
        if method == "anchor_secant"
        else predict_response_kernel
    )
    return fn(model, cross).sum(1)


def construct(out):
    """Freeze all fitted artifacts before reading any natural target scores."""
    extension = load_extension(SPAN)
    functions = load_functions(OUTPUT / "synthetic_reference.npz")
    if read_json(SOURCE / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Incomplete source collection")
    records, diagnostics, files, membership = [], [], {}, {}
    for q in QUERIES:
        x, residual, meta, edges, reader, bias = source_arrays(
            SOURCE, q, extension[q], functions[q]
        )
        gram = x @ x.T
        base = functions[q].anchor + (x @ reader + bias).sum(1)
        anchors = [i for i, row in enumerate(meta) if row["kind"] == "author_anchor"]
        authors = sorted({r["author_id"] for r in meta if r["kind"] == "conditional"})
        membership[q] = authors
        splits = [
            (
                "source_fit",
                np.arange(len(x)),
                [i for i, r in enumerate(meta) if r["kind"] == "conditional"],
            )
        ]
        for a in authors:
            train = np.array([i for i, r in enumerate(meta) if r["author_id"] != a])
            held = [
                i
                for i, r in enumerate(meta)
                if r["kind"] == "conditional" and r["author_id"] == a
            ]
            splits.append(("source_author_holdout", train, held))
        for split, train, held in splits:
            models = fit_three(gram, residual, train, edges, anchors)
            for method, model in models.items():
                scores = base[held] + predict(model, gram[held], method)
                for i, score in zip(held, scores, strict=True):
                    records.append(
                        {
                            "query": q,
                            "method": method,
                            "split": split,
                            "author_id": meta[i]["author_id"],
                            "condition": meta[i]["condition"],
                            "score": float(score),
                            "fixed": meta[i]["fixed"],
                            "native": meta[i]["native"],
                        }
                    )
                if split == "source_fit":
                    path = out / f"program_{q}_{method}.npz"
                    np.savez_compressed(path, **model)
                    files[path.name] = digest(path)
            if split == "source_fit":
                model = models["anchor_secant"]
                drift = predict(model, gram[anchors], "anchor_secant") - predict(
                    models["linear_value"], gram[anchors], "linear_value"
                )
                secant_drift = predict(
                    models["linear_secant"], gram[anchors], "linear_secant"
                ) - predict(models["linear_value"], gram[anchors], "linear_value")
                diagnostics.append(
                    {
                        "query": q,
                        "source_rows": len(x),
                        "anchors": len(anchors),
                        "anchor_correction_max": float(np.max(np.abs(drift))),
                        "old_secant_anchor_drift_max": float(
                            np.max(np.abs(secant_drift))
                        ),
                        "projected_trace_ratio": float(model["projected_trace_ratio"]),
                        "projected_min_eigenvalue": float(
                            model["projected_min_eigenvalue"]
                        ),
                        "lambda": float(model["response_lambda"]),
                    }
                )
        path = out / f"private_inputs_{q}.npz"
        np.savez_compressed(
            path, x=x, reader=reader, bias=bias, anchor=functions[q].anchor
        )
        files[path.name] = digest(path)
        print({"query": q, "source_fits": 39, "new_neural_calls": 0}, flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out / "private_source_predictions.csv", index=False)
    write_json(out / "private_membership.json", membership)
    for name in ("private_source_predictions.csv", "private_membership.json"):
        files[name] = digest(out / name)
    pd.DataFrame(diagnostics).to_csv(out / "anchor_diagnostics.csv", index=False)
    write_json(
        out / "construction_freeze.json",
        {
            "files": files,
            "source_sha256": digest(SOURCE / "collection_freeze.json"),
            "span_sha256": digest(SPAN / "construction_freeze.json"),
            "reader_sha256": digest(OUTPUT / "synthetic_reference.npz"),
            "plan_sha256": digest(PLAN),
            "methods": METHODS,
            "code_sha256": {
                str(p.relative_to(ROOT)): digest(p)
                for p in (
                    Path(__file__),
                    ROOT / "grain_research/vector_mechanisms/reference_integration.py",
                    ROOT / "grain_research/vector_mechanisms/conditional_models.py",
                    ROOT / "scripts/fit_grain_conditional_navigation.py",
                )
            },
            "human_labels_used": False,
            "natural_targets_used": False,
        },
    )
    return frame


def natural_predictions(out):
    """Replay checked early states; join teacher outcomes only after score freeze."""
    frozen = read_json(out / "construction_freeze.json")
    for name, sha in frozen["files"].items():
        if digest(out / name) != sha:
            raise ValueError("Source program changed")
    membership = read_json(out / "private_membership.json")
    worlds = {w["world_id"]: w for w in read_json(NATURAL / "private_worlds.json")}
    design = read_json(NATURAL / "private_design.json")
    _, old, _ = load_operators(OLD)
    records, batch = [], []
    for q in QUERIES:
        with np.load(out / f"private_inputs_{q}.npz", allow_pickle=False) as z:
            x, reader, bias, anchor = z["x"], z["reader"], z["bias"], float(z["anchor"])
        selected, inputs = [], []
        for row in design:
            if row["query"] != q:
                continue
            w = worlds[row["world_id"]]
            if w["author_id"] in membership[q]:
                raise ValueError("Source/evaluation author overlap")
            meta, z = checked_arrays(
                EARLY / "private_units" / f"unit_{row['index']:04d}.json"
            )
            if any(
                meta["identity"][k] != row[k] for k in ("index", "world_id", "query")
            ):
                raise ValueError("Early state identity mismatch")
            selected.append(row)
            inputs.append(z["residual_mean"] - old[q]["mean_r"])
        inputs = np.asarray(inputs)
        cross, base = inputs @ x.T, anchor + (inputs @ reader + bias).sum(1)
        for method in METHODS:
            with np.load(out / f"program_{q}_{method}.npz", allow_pickle=False) as z:
                model = dict(z)
            scores = base + predict(model, cross, method)
            solo = base[0] + predict(model, cross[:1], method)[0]
            batch.append(
                {
                    "query": q,
                    "method": method,
                    "max_batch_change": abs(float(scores[0] - solo)),
                }
            )
            for row, score in zip(selected, scores, strict=True):
                w = worlds[row["world_id"]]
                records.append(
                    {
                        "query": q,
                        "method": method,
                        "world_id": row["world_id"],
                        "split": "natural_exposed",
                        "author_id": w["author_id"],
                        "condition": w["condition"],
                        "score": float(score),
                    }
                )
    p = out / "private_natural_predictions.csv"
    frame = pd.DataFrame(records)
    frame.to_csv(p, index=False)
    pd.DataFrame(batch).to_csv(out / "batch_invariance.csv", index=False)
    write_json(
        out / "prediction_freeze.json",
        {
            "sha256": digest(p),
            "construction_sha256": digest(out / "construction_freeze.json"),
            "natural_targets_joined": False,
            "new_neural_calls": 0,
        },
    )
    target_path = NATURAL / "predictions.csv"
    if digest(target_path) != read_json(NATURAL / "prediction_freeze.json")["sha256"]:
        raise ValueError("Natural target changed")
    target = (
        pd.read_csv(target_path)
        .pivot(index=["query", "world_id"], columns="readout", values="score")
        .reset_index()
        .rename(columns={"R_synth": "fixed"})
    )
    return frame.merge(
        target[["query", "world_id", "fixed", "native"]],
        on=["query", "world_id"],
        validate="many_to_one",
    )


def cube(frame, field, conditions):
    """Require complete aligned conditions, never fill a missing score with zero."""
    tab = frame.pivot(index="author_id", columns="condition", values=field).reindex(
        columns=conditions
    )
    if tab.isna().any().any():
        raise ValueError("Incomplete author condition panel")
    return tab.to_numpy()[:, :, None]


def matched_comparison(out, natural):
    """INT02 cache census: compare native, proxy and programs on identical rows."""
    routepath = ROUTES / "natural/predictions.csv"
    if (
        digest(routepath)
        != read_json(routepath.parent / "prediction_freeze.json")["sha256"]
    ):
        raise ValueError("Old comparator changed")
    routes = pd.read_csv(routepath)
    names = {"B_token": "B0_token", "B_context4": "B4_context", "A_source": "A_source"}
    routes = routes[(routes.reference == "R_synth") & routes.method.isin(names)].copy()
    routes["method"] = routes.method.map(names)
    meta = natural.drop_duplicates(["query", "world_id"])[
        ["query", "world_id", "author_id", "condition", "split", "fixed", "native"]
    ]
    # Old caches also contain split/scene metadata. Keep only identity and score.
    routes = routes[["query", "world_id", "method", "score"]].merge(
        meta, on=["query", "world_id"], validate="many_to_one"
    )
    for (_, _), panel in routes.groupby(["query", "method"]):
        if len(panel) != 96 or panel[["query", "world_id"]].duplicated().any():
            raise ValueError("Missing or duplicate same-material comparator")
    baseline = natural[natural.method == "linear_value"].copy()
    ideal = []
    for method, field in (("native", "native"), ("fixed_f", "fixed")):
        frame = baseline.copy()
        frame["score"], frame["method"] = frame[field], method
        ideal.append(frame)
    frame = pd.concat([natural, routes, *ideal], ignore_index=True)
    rows = []
    for (q, method), g in frame.groupby(["query", "method"]):
        for scope, panel in (
            ("all_endpoints", g),
            ("original", g[g.condition == "original"]),
            (
                "author_mean",
                g.groupby("author_id")[["score", "fixed", "native"]].mean(),
            ),
        ):
            for target in ("native", "fixed"):
                rows.append(
                    {
                        "query": q,
                        "method": method,
                        "scope": scope,
                        "axis": "",
                        "target": target,
                        "authors": 12,
                    }
                    | response_metrics(panel[target], panel.score)
                )
    response = summarize_responses(response_table(frame)).rename(
        columns={"split": "scope"}
    )
    response["scope"] = "finite_response"
    pd.concat([pd.DataFrame(rows), response], ignore_index=True).to_csv(
        out / "target_location_comparison.csv", index=False
    )
    write_json(
        out / "comparison_provenance.json",
        {
            "route_sha256": digest(routepath),
            "natural_sha256": digest(NATURAL / "predictions.csv"),
            "new_predictions_sha256": digest(out / "private_natural_predictions.csv"),
            "new_layers_tested": False,
            "same_material_rows_per_query_method": 96,
            "purpose": "Target/readout comparison; no fresh layer sweep or human criterion",
        },
    )


def analyze(out, source, natural):
    """Separate individual positions, condition response and fixed-reference units."""
    all_rows = pd.concat([source, natural], ignore_index=True)
    responses = response_table(all_rows)
    responses.to_csv(out / "private_responses.csv", index=False)
    summarize_responses(responses).to_csv(out / "response_metrics.csv", index=False)
    levels, profiles, intervals = [], [], []
    for (q, split, method), g in all_rows.groupby(["query", "split", "method"]):
        for scope, panel in (
            ("all_endpoints", g),
            ("original", g[g.condition == "original"]),
            (
                "author_mean",
                g.groupby("author_id")[["score", "fixed", "native"]].mean(),
            ),
        ):
            for target in ("fixed", "native"):
                levels.append(
                    {
                        "query": q,
                        "split": split,
                        "method": method,
                        "scope": scope,
                        "target": target,
                        "authors": g.author_id.nunique(),
                    }
                    | response_metrics(panel[target], panel.score)
                )
    pd.DataFrame(levels).to_csv(out / "level_metrics.csv", index=False)
    # One common target reference plus separate versioned program references.
    for q in QUERIES:
        s = source[(source["query"] == q) & (source.split == "source_fit")]
        e = natural[natural["query"] == q]
        conditions = sorted(s.condition.unique())
        for method in METHODS:
            sc, ec = s[s.method == method], e[e.method == method]
            values, held = cube(sc, "score", conditions), cube(ec, "score", conditions)
            ref = fit_condition_reference(
                values,
                np.ones(len(conditions)) / len(conditions),
                reference_id=f"INT01-{q}-{method}",
            )
            prof = condition_profile(held, ref)
            solo = condition_profile(held[:1], ref)
            path = out / f"reference_{q}_{method}.npz"
            np.savez_compressed(
                path,
                condition_mean=ref["condition_mean"],
                weights=ref["weights"],
                level_scale=ref["level_scale"],
                conditions=np.array(conditions),
            )
            profiles.append(
                {
                    "query": q,
                    "method": method,
                    "reconstruction_max": prof["reconstruction_max_error"],
                    "centered_shape_max": prof["shape_weighted_mean_max"],
                    "profile_batch_change": float(
                        np.nanmax(
                            abs(
                                prof["standardized_level"][:1]
                                - solo["standardized_level"]
                            )
                        )
                    ),
                    "level_second_moment": float(np.mean(prof["level"] ** 2)),
                    "conditional_shape_second_moment": float(
                        np.mean(prof["response_shape"] ** 2)
                    ),
                    "source_reference_scale": float(ref["level_scale"][0]),
                }
            )
    pd.DataFrame(profiles).to_csv(out / "reference_profile_checks.csv", index=False)
    rng = np.random.default_rng(20260909)
    for scope, frame, keys in (
        ("endpoint", natural, ["query", "author_id", "condition"]),
        (
            "response",
            responses[responses.split == "natural_exposed"],
            ["query", "author_id", "axis"],
        ),
    ):
        base = frame[frame.method == "linear_value"][keys + ["score"]].rename(
            columns={"score": "base"}
        )
        paired = frame.merge(base, on=keys, validate="many_to_one")
        for (q, method), g in paired.groupby(["query", "method"]):
            for target in ("fixed", "native"):
                delta = (g.score - g[target]) ** 2 - (g.base - g[target]) ** 2
                cluster = delta.groupby(g.author_id).mean().to_numpy()
                draws = cluster[
                    rng.integers(len(cluster), size=(2000, len(cluster)))
                ].mean(1)
                lo, hi = np.quantile(draws, [0.025, 0.975])
                intervals.append(
                    {
                        "query": q,
                        "method": method,
                        "scope": scope,
                        "target": target,
                        "authors": len(cluster),
                        "delta_mse_vs_value": float(cluster.mean()),
                        "ci_low": float(lo),
                        "ci_high": float(hi),
                    }
                )
    pd.DataFrame(intervals).to_csv(out / "paired_author_intervals.csv", index=False)
    write_json(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "queries": QUERIES,
            "methods": METHODS,
            "source_authors": 12,
            "evaluation_authors": 12,
            "source_rows_per_query": 156,
            "human_labels_used": False,
            "MEPS_used": False,
            "new_neural_calls": 0,
            "source_evaluation_author_overlap": 0,
            "bootstrap_unit": "author, conditional on frozen fits",
            "scope": "Exposed real-text edits; model evidence response, not human repeated behavior",
            "next": "INT02 target-aware level/response comparison, not an automatic tuning grid",
        },
    )


def main():
    """Run one bounded fresh packet, keeping derived private material ignored."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() or not out.is_relative_to(
        (BASE / "reference_integration_v1").resolve()
    ):
        raise ValueError("Choose a fresh ignored reference-integration directory")
    out.mkdir(parents=True)
    start = time.monotonic()
    try:
        with threadpool_limits(limits=2):
            source = construct(out)
            natural = natural_predictions(out)
            analyze(out, source, natural)
            matched_comparison(out, natural)
        write_json(
            out / "run_status.json",
            {"status": "COMPLETE", "seconds": time.monotonic() - start},
        )
    except Exception as exc:
        write_json(
            out / "run_status.json",
            {
                "status": "ERROR",
                "type": type(exc).__name__,
                "seconds": time.monotonic() - start,
            },
        )
        raise
    print(read_json(out / "run_status.json"), flush=True)


if __name__ == "__main__":
    main()
