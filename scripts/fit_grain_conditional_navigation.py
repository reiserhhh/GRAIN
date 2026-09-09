#!/usr/bin/env python3
"""CON01: fixed value/secant x linear/quadratic comparison, without human labels."""

from __future__ import annotations

import argparse
import sys
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
from grain_research.vector_mechanisms.state_geometry import linear_reader
from grain_research.vector_mechanisms.state_transport import pack_state
from scripts.analyze_grain_conditional_navigation import EARLY, NATURAL, PAIRS, ROUTES
from scripts.collect_grain_conditional_sources import DEFAULT as SOURCE
from scripts.run_grain_navigation_baseline import checked_arrays
from scripts.run_grain_source_span_extension import DEFAULT as SPAN
from scripts.run_grain_source_span_extension import OLD, load_extension
from scripts.run_grain_state_transport import load_operators

DEFAULT = BASE / "conditional_navigation_v1/CON-01/fit_r1"
METHODS = {
    "linear_value": (1, False),
    "linear_secant": (1, True),
    "quadratic_value": (2, False),
    "quadratic_secant": (2, True),
}


def source_arrays(source, query, extension, function):
    """Recover exact old construction anchors plus newly measured conditional endpoints."""
    frozen = read_json(source / "collection_freeze.json")
    if digest(source / "manifest.json") != frozen["manifest_sha256"]:
        raise ValueError("Source collection changed")
    manifest = read_json(source / "manifest.json")
    if any(
        digest(source / name) != sha for name, sha in manifest["input_sha256"].items()
    ):
        raise ValueError("Source endpoint design changed")
    design = [
        d for d in read_json(source / "private_design.json") if d["query"] == query
    ]
    authors = read_json(SPAN / "private_membership.json")["reference_authors"]
    u = extension["train_x"]
    x0 = extension["mean_x"] + extension["scale"] * u
    k = u @ u.T
    kc = k - extension["col1"][None] - extension["col1"][:, None] + extension["grand1"]
    # Multiplying the original fitted equation recovers its60 target residuals.
    residual = (kc + float(extension["alpha"]) * np.eye(len(u))) @ extension[
        "coef1"
    ] + extension["mean_r"]
    l, bias = linear_reader(function)
    xx, rr = [*x0], [*(residual @ l)]
    rows = [{"kind": "synthetic", "author_id": "", "condition": ""} for _ in range(48)]
    rows += [
        {"kind": "author_anchor", "author_id": a, "condition": ""} for a in authors
    ]
    if len(x0) != 60:
        raise ValueError("Expected48 synthetic and12 author-mean anchors")
    for d in design:
        path = source / "private_packets" / f"unit_{d['index']:04d}.json"
        if digest(path) != frozen["packets"][path.name]:
            raise ValueError("Source endpoint marker changed")
        meta, z = checked_arrays(path)
        if any(meta[k] != d[k] for k in ("index", "query", "author_id", "condition")):
            raise ValueError("Source endpoint mismatch")
        x, y = pack_state(z["P_early"], z["Q_early"]), pack_state(z["P"], z["Q"])
        xx.append(x)
        rr.append((y - x) @ l)
        rows.append(
            {
                "kind": "conditional",
                "author_id": d["author_id"],
                "condition": d["condition"],
                "source_index": d["source_index"],
                "native": meta["native"]["conditional_score"],
                "fixed": function.anchor + float((y @ l + bias).sum()),
            }
        )
    x, r = np.asarray(xx), np.asarray(rr)
    if len(rows) != 156:
        raise ValueError("Expected156 construction endpoints per query")
    index = {
        (row["author_id"], row["condition"]): i
        for i, row in enumerate(rows)
        if row["kind"] == "conditional"
    }
    edges = [(index[a, low], index[a, high]) for a in authors for low, high, _ in PAIRS]
    return x, r, rows, edges, l, bias


def response_table(predictions):
    """Keep author-held and condition-held finite responses distinct."""
    rows = []
    for key, g in predictions.groupby(["query", "split", "method", "author_id"]):
        lookup = g.set_index("condition")
        if lookup.index.duplicated().any():
            raise ValueError("Duplicate endpoint prediction")
        for low, high, axis in PAIRS:
            if low not in lookup.index or high not in lookup.index:
                continue
            a, b = lookup.loc[low], lookup.loc[high]
            rows.append(
                dict(zip(("query", "split", "method", "author_id"), key, strict=True))
                | {
                    "axis": axis,
                    "score": b.score - a.score,
                    "fixed": b.fixed - a.fixed,
                    "native": b.native - a.native,
                }
            )
    return pd.DataFrame(rows)


def summarize_responses(responses):
    """Report per-axis teacher/native response metrics on the declared clusters."""
    rows = []
    for key, g in responses.groupby(["query", "split", "method", "axis"]):
        ident = dict(zip(("query", "split", "method", "axis"), key, strict=True))
        for target in ("fixed", "native"):
            rows.append(
                ident
                | {"target": target, "authors": g.author_id.nunique()}
                | response_metrics(g[target], g.score)
            )
    return pd.DataFrame(rows)


def fit_and_freeze(out, source):
    """Source-only author/condition holdouts, followed by one final fit per variant."""
    if read_json(source / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Finish source collection first")
    extension, functions = (
        load_extension(SPAN),
        load_functions(OUTPUT / "synthetic_reference.npz"),
    )
    out.mkdir(parents=True)
    rows, diagnostics, files = [], [], {}
    for q in QUERIES:
        x, residual, meta, edges, reader, bias = source_arrays(
            source, q, extension[q], functions[q]
        )
        gram, indices = x @ x.T, np.arange(len(x))
        base = functions[q].anchor + (x @ reader + bias).sum(1)
        authors = sorted({r["author_id"] for r in meta if r["kind"] == "conditional"})
        splits = []
        for author in authors:
            train = [i for i, r in enumerate(meta) if r["author_id"] != author]
            test = [
                i
                for i, r in enumerate(meta)
                if r["author_id"] == author and r["kind"] == "conditional"
            ]
            splits.append(("author_holdout", train, test))
        for condition, counterpart in (("denied", "endorsed"), ("past", "today")):
            train = [i for i, r in enumerate(meta) if r["condition"] != condition]
            test = [
                i
                for i, r in enumerate(meta)
                if r["condition"] in (condition, counterpart)
            ]
            splits.append(("holdout_" + condition, train, test))
        for split, train, test in splits:
            for method, (degree, secant) in METHODS.items():
                model = fit_response_kernel(
                    gram, residual, train, edges, degree=degree, secant=secant
                )
                pred = base[test] + predict_response_kernel(model, gram[test]).sum(1)
                for i, score in zip(test, pred, strict=True):
                    rows.append(
                        {
                            "query": q,
                            "split": split,
                            "method": method,
                            "author_id": meta[i]["author_id"],
                            "condition": meta[i]["condition"],
                            "score": float(score),
                            "fixed": meta[i]["fixed"],
                            "native": meta[i]["native"],
                        }
                    )
        path = out / f"private_inputs_{q}.npz"
        np.savez_compressed(
            path, x=x, reader=reader, bias=bias, anchor=np.asarray(functions[q].anchor)
        )
        files[path.name] = digest(path)
        for method, (degree, secant) in METHODS.items():
            model = fit_response_kernel(
                gram, residual, indices, edges, degree=degree, secant=secant
            )
            path = out / f"program_{q}_{method}.npz"
            np.savez_compressed(path, **model)
            files[path.name] = digest(path)
            diagnostics.append(
                {
                    "query": q,
                    "method": method,
                    "construction_rows": len(x),
                    "dimensions": x.shape[1],
                    "edges": int(model["edge_count"]),
                    "lambda": float(model["lambda"]),
                    "alpha": 1.0,
                }
            )
        print(
            {"phase": "SOURCE_FIT", "query": q, "holdouts": len(splits), "programs": 4},
            flush=True,
        )
    predictions = pd.DataFrame(rows)
    predictions.to_csv(out / "private_source_predictions.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out / "construction_diagnostics.csv", index=False)
    responses = response_table(predictions)
    responses.to_csv(out / "private_source_responses.csv", index=False)
    summarize_responses(responses).to_csv(
        out / "source_response_metrics.csv", index=False
    )
    write_json(
        out / "construction_freeze.json",
        {
            "files": files,
            "source_sha256": digest(source / "collection_freeze.json"),
            "anchor_sha256": digest(SPAN / "construction_freeze.json"),
            "reader_sha256": digest(OUTPUT / "synthetic_reference.npz"),
            "plan_sha256": digest(
                ROOT / "docs/GRAIN_CONDITIONAL_NAVIGATION_PLAN_V1.md"
            ),
            "code_sha256": {
                str(p.relative_to(ROOT)): digest(p)
                for p in (
                    Path(__file__),
                    ROOT / "grain_research/vector_mechanisms/conditional_models.py",
                )
            },
            "methods": list(METHODS),
            "human_labels_used": False,
            "evaluation_targets_used_in_fit": False,
        },
    )


def evaluate_natural(out):
    """Predict all exposed natural endpoints before joining fixed/native target scores."""
    frozen = read_json(out / "construction_freeze.json")
    for name, sha in frozen["files"].items():
        if digest(out / name) != sha:
            raise ValueError("Conditional program changed")
    _, old, _ = load_operators(OLD)
    worlds = {w["world_id"]: w for w in read_json(NATURAL / "private_worlds.json")}
    design = read_json(NATURAL / "private_design.json")
    rows = []
    for q in QUERIES:
        with np.load(out / f"private_inputs_{q}.npz", allow_pickle=False) as z:
            x, reader, bias, anchor = z["x"], z["reader"], z["bias"], float(z["anchor"])
        selected, inputs = [], []
        for d in design:
            if d["query"] != q:
                continue
            meta, z = checked_arrays(
                EARLY / "private_units" / f"unit_{d['index']:04d}.json"
            )
            if any(meta["identity"][k] != d[k] for k in ("index", "world_id", "query")):
                raise ValueError("Natural early-state identity mismatch")
            selected.append(d)
            inputs.append(z["residual_mean"] - old[q]["mean_r"])
        inputs = np.asarray(inputs)
        cross, base = inputs @ x.T, anchor + (inputs @ reader + bias).sum(1)
        for method in METHODS:
            with np.load(out / f"program_{q}_{method}.npz", allow_pickle=False) as z:
                prediction = base + predict_response_kernel(dict(z), cross).sum(1)
            for d, score in zip(selected, prediction, strict=True):
                w = worlds[d["world_id"]]
                rows.append(
                    {
                        "query": q,
                        "world_id": d["world_id"],
                        "split": "natural_exposed",
                        "method": method,
                        "author_id": w["author_id"],
                        "condition": w["condition"],
                        "score": float(score),
                    }
                )
    path = out / "private_natural_predictions.csv"
    predictions = pd.DataFrame(rows)
    predictions.to_csv(path, index=False)
    write_json(
        out / "prediction_freeze.json",
        {
            "sha256": digest(path),
            "construction_sha256": digest(out / "construction_freeze.json"),
            "natural_targets_joined": False,
            "human_labels_used": False,
            "new_neural_calls": 0,
        },
    )
    targetpath = NATURAL / "predictions.csv"
    if digest(targetpath) != read_json(NATURAL / "prediction_freeze.json")["sha256"]:
        raise ValueError("Natural targets changed")
    target = (
        pd.read_csv(targetpath)
        .pivot(index=["query", "world_id"], columns="readout", values="score")
        .reset_index()
        .rename(columns={"R_synth": "fixed"})
    )
    predictions = predictions.merge(
        target[["query", "world_id", "fixed", "native"]],
        on=["query", "world_id"],
        validate="many_to_one",
    )
    responses = response_table(predictions)
    responses.to_csv(out / "private_natural_responses.csv", index=False)
    summarize_responses(responses).to_csv(
        out / "natural_response_metrics.csv", index=False
    )
    levels = []
    for (q, method), g in predictions.groupby(["query", "method"]):
        for t in ("fixed", "native"):
            levels.append(
                {
                    "query": q,
                    "method": method,
                    "target": t,
                    "authors": g.author_id.nunique(),
                }
                | response_metrics(g[t], g.score)
            )
    pd.DataFrame(levels).to_csv(out / "natural_level_metrics.csv", index=False)
    route_path = ROUTES / "natural/predictions.csv"
    if (
        digest(route_path)
        != read_json(route_path.parent / "prediction_freeze.json")["sha256"]
    ):
        raise ValueError("Frozen comparator predictions changed")
    route = pd.read_csv(route_path)
    route = route[(route.reference == "R_synth") & (route.method == "A_source")]
    meta = predictions.drop_duplicates(["query", "world_id"])[
        ["query", "world_id", "author_id", "condition", "fixed", "native"]
    ]
    comparator = route.merge(meta, on=["query", "world_id"], validate="one_to_one")
    comparator["split"] = "natural_exposed"
    comp = response_table(comparator)
    baseline = comp[["query", "author_id", "axis", "score"]].rename(
        columns={"score": "baseline"}
    )
    paired = responses.merge(
        baseline, on=["query", "author_id", "axis"], validate="many_to_one"
    )
    intervals, rng = [], np.random.default_rng(42)
    for (q, method, axis), g in paired.groupby(["query", "method", "axis"]):
        delta = ((g.score - g.native) ** 2 - (g.baseline - g.native) ** 2).to_numpy()
        draws = delta[rng.integers(0, len(delta), size=(2000, len(delta)))].mean(1)
        lo, hi = np.quantile(draws, [0.025, 0.975])
        intervals.append(
            {
                "query": q,
                "method": method,
                "axis": axis,
                "authors": len(delta),
                "delta_mse_vs_A_source": float(delta.mean()),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "scope": "paired authors; conditional on frozen fits; exposed panel",
            }
        )
    pd.DataFrame(intervals).to_csv(out / "paired_error_intervals.csv", index=False)


def main():
    """Execute fixed source holdouts and natural replay in a fresh isolated directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() or not out.is_relative_to(
        (BASE / "conditional_navigation_v1").resolve()
    ):
        raise ValueError("Choose a fresh ignored fit directory")
    with threadpool_limits(limits=2):
        fit_and_freeze(out, args.source)
        evaluate_natural(out)
    write_json(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "queries": list(QUERIES),
            "methods": list(METHODS),
            "source_authors": 12,
            "source_rows_per_query": 156,
            "holdout_schemes": [
                "leave_one_author_out_with_anchor_removed",
                "denied_endpoint_out",
                "past_endpoint_out",
            ],
            "natural_evaluation_authors": 12,
            "new_neural_calls": 0,
            "human_labels_used": False,
            "MEPS_used": False,
            "next": "Compare source held responses and natural improvements before CON02/03.",
        },
    )
    print(read_json(out / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
