#!/usr/bin/env python3
"""NAV02: one fixed source-span extension, followed by held-state/response analysis."""

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

from grain_research.vector_mechanisms.navigation import (
    BASE,
    OUTPUT,
    QUERIES,
    digest,
    quality,
    read_json,
    write_json,
)
from grain_research.vector_mechanisms.source_span import (
    author_means,
    fit_linear,
    predict_linear,
)
from grain_research.vector_mechanisms.state_geometry import (
    decompose_state_error,
    linear_reader,
    residual_basis,
)
from grain_research.vector_mechanisms.state_transport import pack_state, unpack_state
from scripts.analyze_grain_evidence_packets import compare_responses, response_rows
from scripts.run_grain_navigation_baseline import checked_arrays, save_unit
from scripts.run_grain_state_transport import DEFAULT as OLD
from scripts.run_grain_state_transport import input_units, load_operators, load_runtime
from scripts.run_grain_upstream_routes import collector_guard

DEFAULT = BASE / "navigation_v1/NAV-02/source_span_r1"
PANEL = BASE / "navigation_v1/NAV-08/author_panel_r1"
NATURAL = BASE / "navigation_v1/NAV-04/natural_evidence_r1"


def code_hashes():
    """Pin constructor and arithmetic code before fitting or held evaluation."""
    return {
        s: digest(ROOT / s)
        for s in (
            "scripts/run_grain_source_span_extension.py",
            "grain_research/vector_mechanisms/source_span.py",
            "grain_research/vector_mechanisms/state_geometry.py",
        )
    }


def verify_manifest(out):
    """Reject changes to membership, source coordinates and implementation."""
    m = read_json(out / "manifest.json")
    if m["code_sha256"] != code_hashes():
        raise ValueError("Frozen source-span implementation changed")
    for name, sha in m["sources"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Source changed: " + name)
    for name, sha in m["private_inputs"].items():
        if digest(out / name) != sha:
            raise ValueError("Construction membership changed")
    return m


def prepare(out):
    """Freeze12 reference authors,24 paired views and48 synthetic worlds per query."""
    if out.exists() or not out.resolve().is_relative_to(
        (BASE / "navigation_v1").resolve()
    ):
        raise ValueError("Use a fresh ignored navigation directory")
    pm = read_json(PANEL / "manifest.json")
    for name in ("private_worlds.json", "private_design.json"):
        if digest(PANEL / name) != pm["input_sha256"][name]:
            raise ValueError("Panel source changed")
    world = {w["world_id"]: w for w in read_json(PANEL / "private_worlds.json")}
    refs = {w["author_id"] for w in world.values() if w["role"] == "reference"}
    held = {w["author_id"] for w in world.values() if w["role"] == "evaluation"}
    if len(refs) != 12 or len(held) != 12 or refs & held:
        raise ValueError("Expected disjoint12/12 panel")
    natural_authors = {
        w["author_id"] for w in read_json(NATURAL / "private_worlds.json")
    }
    if natural_authors != held:
        raise ValueError("Natural responses are not the held panel")
    selected = []
    for d in read_json(PANEL / "private_design.json"):
        w = world[d["world_id"]]
        if w["author_id"] in refs:
            selected.append({**d, **{k: w[k] for k in ("author_id", "view", "role")}})
    if len(selected) != 72:
        raise ValueError("Expected72 missing early snapshots")
    source_path = BASE / "all_factor_source_m29_v1/private_design.json"
    if (
        digest(source_path)
        != read_json(source_path.parent / "manifest.json")["input_sha256"]
    ):
        raise ValueError("Source-only endpoint membership changed")
    source_units = read_json(source_path)
    overlap = {
        u["author_id"]
        for u in source_units
        if u["query"] in ("extraversion", "TF_cont")
    } & refs
    out.mkdir(parents=True)
    (out / "private_early").mkdir()
    write_json(out / "private_design.json", selected)
    write_json(
        out / "private_membership.json",
        {
            "reference_authors": sorted(refs),
            "evaluation_authors": sorted(held),
            "external_source_overlap": sorted(overlap),
            "rule": "equal one row per natural author after averaging two views",
        },
    )
    sources = [
        PANEL / "manifest.json",
        PANEL / "private_design.json",
        PANEL / "private_worlds.json",
        OLD / "program.json",
        source_path,
        OUTPUT / "manifest.json",
        OUTPUT / "design.json",
        OUTPUT / "worlds.json",
        NATURAL / "manifest.json",
        NATURAL / "private_worlds.json",
        ROOT / "docs/GRAIN_NAVIGATION_SOURCE_SPAN_EXTENSION_PLAN_V1.md",
    ]
    write_json(
        out / "manifest.json",
        {
            "protocol": "NAV02_SOURCE_SPAN_EXTENSION_V1",
            "code_sha256": code_hashes(),
            "sources": {str(p.relative_to(ROOT)): digest(p) for p in sources},
            "private_inputs": {
                n: digest(out / n)
                for n in ("private_design.json", "private_membership.json")
            },
            "alpha": 1.0,
            "kernel": "linear",
            "context_layers": 4,
            "construction_rows_per_query": 60,
            "synthetic_rows": 48,
            "natural_author_rows": 12,
            "construction_source": "synthetic plus unlabelled reference-author states; not toy-only",
            "human_labels_fit": False,
            "MEPS_construction": False,
            "source_endpoint_overlap_excluded": len(overlap),
        },
    )
    write_json(
        out / "run_status.json", {"status": "PREPARED", "calls": 0, "seconds": 0}
    )
    print(
        {
            "status": "PREPARED",
            "reference_authors": 12,
            "early_calls": 72,
            "external_overlap": len(overlap),
        },
        flush=True,
    )


def construct(out, seconds):
    """Collect reference-only early states and fit the single frozen extension."""
    m = verify_manifest(out)
    prior = read_json(out / "run_status.json")
    if prior["status"] == "CONSTRUCTION_COMPLETE":
        print(prior, flush=True)
        return
    p, runtime = load_runtime(OLD)
    design = read_json(out / "private_design.json")
    start = time.monotonic()
    for d in design:
        path = out / "private_early" / f"unit_{d['index']:04d}.json"
        if path.exists():
            meta, _ = checked_arrays(path)
            if any(
                meta[k] != d[k]
                for k in ("index", "query", "world_id", "author_id", "view")
            ):
                raise ValueError("Reference early identity changed")
            continue
        if time.monotonic() - start >= seconds:
            write_json(
                out / "run_status.json",
                {
                    "status": "PAUSED_RESOURCE_BOUNDARY",
                    "calls": prior["calls"] + runtime.calls,
                    "seconds": prior["seconds"] + time.monotonic() - start,
                },
            )
            return
        mask = np.asarray(d["mask"], bool)
        if (
            np.asarray(d["input_ids"])[mask].tolist()
            != p["question_tokens"][d["query"]]
        ):
            raise ValueError("Query coordinates changed")
        h = runtime.capture(d["input_ids"])[3]
        save_unit(
            path,
            {k: d[k] for k in ("index", "query", "world_id", "author_id", "view")},
            P=h[mask],
            Q=h[-1],
        )
    roles = {w["world_id"]: w["split"] for w in read_json(OUTPUT / "worlds.json")}
    syn = read_json(OUTPUT / "design.json")
    audit, files, snapshots = [], {}, {}
    with threadpool_limits(limits=2):
        for q in QUERIES:
            xx, yy, records = [], [], []
            for d in syn:
                if d["query"] != q or roles[d["world_id"]] != "construction":
                    continue
                paths = (
                    OLD / "private_early" / f"unit_{d['index']:04d}.json",
                    OUTPUT / "packets" / f"unit_{d['index']:04d}.json",
                )
                pair = []
                for path in paths:
                    meta, z = checked_arrays(path)
                    if any(meta[k] != d[k] for k in ("index", "query", "world_id")):
                        raise ValueError("Synthetic state identity changed")
                    pair.append(pack_state(z["P"], z["Q"]))
                    snapshots[str(path.relative_to(ROOT))] = digest(path)
                xx.append(pair[0])
                yy.append(pair[1])
            if len(xx) != 48:
                raise ValueError("Synthetic membership changed")
            for d in design:
                if d["query"] != q:
                    continue
                paths = (
                    out / "private_early" / f"unit_{d['index']:04d}.json",
                    PANEL / "private_packets" / f"unit_{d['index']:04d}.json",
                )
                pair = []
                for path in paths:
                    meta, z = checked_arrays(path)
                    if any(meta[k] != d[k] for k in ("index", "query", "world_id")):
                        raise ValueError("Natural state identity changed")
                    pair.append(pack_state(z["P"], z["Q"]))
                    snapshots[str(path.relative_to(ROOT))] = digest(path)
                records.append(
                    {
                        **{k: d[k] for k in ("author_id", "view", "role")},
                        "x": pair[0],
                        "y": pair[1],
                    }
                )
            authors, ax, ay = author_means(records)
            if (
                authors
                != read_json(out / "private_membership.json")["reference_authors"]
            ):
                raise ValueError("Author aggregation membership changed")
            model = fit_linear(
                np.vstack([xx, ax]), np.vstack([yy, ay]), alpha=m["alpha"]
            )
            np.savez_compressed(out / f"{q}.npz", **model)
            files[f"{q}.npz"] = digest(out / f"{q}.npz")
            audit.append(
                {
                    "query": q,
                    "rows": len(model["train_x"]),
                    "residual_rank": len(residual_basis(model)),
                    "effective_df": float(model["effective_df1"]),
                }
            )
    write_json(
        out / "construction_freeze.json",
        {
            "operators": files,
            "snapshot_metadata_sha256": snapshots,
            "manifest_sha256": digest(out / "manifest.json"),
            "human_labels_used": False,
        },
    )
    pd.DataFrame(audit).to_csv(out / "construction_diagnostics.csv", index=False)
    write_json(
        out / "run_status.json",
        {
            "status": "CONSTRUCTION_COMPLETE",
            "calls": prior["calls"] + runtime.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
        },
    )
    print(pd.DataFrame(audit).round(4).to_string(index=False), flush=True)
    print(read_json(out / "run_status.json"), flush=True)


def load_extension(out):
    """Load frozen operators without any teacher or questionnaire input."""
    verify_manifest(out)
    if read_json(out / "run_status.json")["status"] != "CONSTRUCTION_COMPLETE":
        raise ValueError("Finish construction")
    freeze = read_json(out / "construction_freeze.json")
    if freeze["manifest_sha256"] != digest(out / "manifest.json"):
        raise ValueError("Construction freeze changed")
    models = {}
    for name, sha in freeze["operators"].items():
        if digest(out / name) != sha:
            raise ValueError("Frozen operator changed")
        with np.load(out / name, allow_pickle=False) as z:
            models[Path(name).stem] = dict(z)
    return models


def evaluate(out, mode):
    """Reuse stored early inputs; freeze all generated predictions before teacher joins."""
    models = load_extension(out)
    p, old, functions = load_operators(OLD)
    units, _sources = input_units(mode)
    previous = BASE / f"navigation_v1/NAV-10/state_{mode}_r1"
    dest = out / mode
    if dest.exists():
        raise FileExistsError("Use a fresh evaluation")
    dest.mkdir()
    (dest / "private_units").mkdir()
    rows = []
    start = time.monotonic()
    with threadpool_limits(limits=2):
        for u in units:
            meta, z = checked_arrays(
                previous / "private_units" / f"unit_{u['index']:04d}.json"
            )
            if any(meta["identity"][k] != u[k] for k in ("index", "query", "world_id")):
                raise ValueError("Saved early input identity changed")
            q = u["query"]
            # This old generated control equals x+mean(R); it contains no teacher state.
            x = z["residual_mean"] - old[q]["mean_r"]
            y = predict_linear(models[q], x)
            identity = {k: u[k] for k in ("index", "query", "world_id")}
            parts = functions[q].parts(*unpack_state(y, p["state_shapes"][q]))
            row = {**identity, "readout": "source_span", **parts}
            save_unit(
                dest / "private_units" / f"unit_{u['index']:04d}.json",
                {"identity": identity, "row": row},
                estimated=y,
            )
            rows.append(row)
    pd.DataFrame(rows).to_csv(dest / "predictions.csv", index=False)
    write_json(
        dest / "prediction_freeze.json",
        {
            "sha256": digest(dest / "predictions.csv"),
            "construction_sha256": digest(out / "construction_freeze.json"),
            "teacher_inputs": False,
            "human_labels_read": False,
        },
    )
    write_json(
        dest / "run_status.json",
        {
            "status": "COMPLETE",
            "units": len(units),
            "new_neural_calls": 0,
            "seconds": time.monotonic() - start,
        },
    )
    print(read_json(dest / "run_status.json"), flush=True)


def analyze(out, mode):
    """Quantify source coverage versus coefficients after generated-score freeze."""
    models = load_extension(out)
    p, old, functions = load_operators(OLD)
    dest = out / mode
    freeze = read_json(dest / "prediction_freeze.json")
    if digest(dest / "predictions.csv") != freeze["sha256"] or freeze[
        "construction_sha256"
    ] != digest(out / "construction_freeze.json"):
        raise ValueError("Prediction freeze changed")
    analysis = dest / "analysis"
    if analysis.exists():
        raise FileExistsError("Analysis already exists")
    analysis.mkdir()
    previous = BASE / f"navigation_v1/NAV-10/state_{mode}_r1"
    src = OUTPUT if mode == "worlds" else NATURAL
    design = read_json(
        src / ("design.json" if mode == "worlds" else "private_design.json")
    )
    world = pd.DataFrame(
        read_json(src / ("worlds.json" if mode == "worlds" else "private_worlds.json"))
    ).drop(columns="texts")
    bases = {
        q: {
            "synthetic": residual_basis(old[q]),
            "source_span": residual_basis(models[q]),
        }
        for q in QUERIES
    }
    levels, geometry = [], []
    with threadpool_limits(limits=2):
        for d in design:
            index, q = d["index"], d["query"]
            om, oz = checked_arrays(
                previous / "private_units" / f"unit_{index:04d}.json"
            )
            nm, nz = checked_arrays(dest / "private_units" / f"unit_{index:04d}.json")
            tm, tz = checked_arrays(
                src
                / ("packets" if mode == "worlds" else "private_packets")
                / f"unit_{index:04d}.json"
            )
            for metadata in (om["identity"], nm["identity"], tm):
                if any(metadata[k] != d[k] for k in ("index", "query", "world_id")):
                    raise ValueError("Teacher/student identity mismatch")
            x = oz["residual_mean"] - old[q]["mean_r"]
            actual = pack_state(tz["P"], tz["Q"])
            reader, _ = linear_reader(functions[q])
            full = functions[q].parts(tz["P"], tz["Q"])["score"]
            ident = {"query": q, "world_id": d["world_id"]}
            levels.extend(
                [
                    {
                        **ident,
                        "readout": "native",
                        "score": tm["native"]["conditional_score"],
                    },
                    {**ident, "readout": "full_function", "score": full},
                ]
            )
            outside = {}
            for name, model, estimate in (
                ("synthetic", old[q], oz["transport1"]),
                ("source_span", models[q], nz["estimated"]),
            ):
                parts = functions[q].parts(
                    *unpack_state(estimate, p["state_shapes"][q])
                )
                levels.append({**ident, "readout": name, **parts})
                oracle, eout, ein, stats = decompose_state_error(
                    model, bases[q][name], x, actual, estimate
                )
                outside[name] = stats["outside_squared"]
                geometry.append(
                    {
                        **ident,
                        "readout": name,
                        "dimensions": len(x),
                        "rank": len(bases[q][name]),
                        **stats,
                        "outside_score_error": float((eout @ reader).sum()),
                        "inside_score_error": float((ein @ reader).sum()),
                        "full_function": full,
                        "estimated": parts["score"],
                        "oracle": functions[q].parts(
                            *unpack_state(oracle, p["state_shapes"][q])
                        )["score"],
                        "oracle_uses_teacher": True,
                    }
                )
            if outside["source_span"] > outside["synthetic"] + 1e-5 * max(
                outside["synthetic"], 1
            ):
                raise ValueError(
                    "Expanded affine span unexpectedly loses synthetic coverage"
                )
    frame = pd.DataFrame(levels).merge(world, on="world_id", validate="many_to_one")
    geo = pd.DataFrame(geometry).merge(
        world[["world_id", "split"]], on="world_id", validate="many_to_one"
    )
    metrics = []
    for (q, name, split), g in geo.groupby(["query", "readout", "split"]):
        metrics.append(
            {
                "query": q,
                "readout": name,
                "split": split,
                **quality(g.full_function, g.estimated),
                "oracle_r": quality(g.full_function, g.oracle)["r"],
                "rank": int(g["rank"].iloc[0]),
                "outside_fraction": float(
                    g.outside_squared.sum() / g.total_squared.sum()
                ),
                "outside_mse": float((g.outside_squared / g.dimensions).mean()),
                "inside_mse": float((g.inside_squared / g.dimensions).mean()),
                "total_mse": float((g.total_squared / g.dimensions).mean()),
                "outside_score_rmse": float(np.sqrt(np.mean(g.outside_score_error**2))),
                "inside_score_rmse": float(np.sqrt(np.mean(g.inside_score_error**2))),
            }
        )
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(analysis / "metrics.csv", index=False)
    frame.to_csv(analysis / "private_levels.csv", index=False)
    geo.to_csv(analysis / "private_geometry.csv", index=False)
    responses = response_rows(frame, "identity" if mode == "worlds" else "evidence")
    joined, effects = compare_responses(responses)
    responses.to_csv(analysis / "paired_responses.csv", index=False)
    joined.to_csv(analysis / "response_comparison.csv", index=False)
    effects.to_csv(analysis / "response_metrics.csv", index=False)
    eligible = []
    if mode == "natural":
        for q in QUERIES:
            qr = metrics[metrics["query"] == q].set_index("readout")["r"]
            e = effects[(effects["query"] == q) & (effects.readout == "source_span")]
            if (
                qr["source_span"] > qr["synthetic"]
                and qr["source_span"] > 0
                and (e["r"] > 0).any()
            ):
                eligible.append(q)
    write_json(
        analysis / "decision.json",
        {
            "status": "COMPLETE",
            "source_span_is_nested": True,
            "maximum_energy_accounting_error": float(
                geo.orthogonality_accounting_error.max()
            ),
            "model_only_queries_supporting_external_probe": eligible,
            "external_probe_all_queries": bool(eligible),
            "no_human_labels_read": True,
        },
    )
    print(
        metrics.query("split != 'construction'").round(5).to_string(index=False),
        flush=True,
    )
    print(read_json(analysis / "decision.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "construct", "evaluate", "analyze")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--mode", choices=("worlds", "natural"), default="natural")
    parser.add_argument("--seconds", type=int, default=1200)
    args = parser.parse_args()
    with collector_guard(args.out), threadpool_limits(limits=2):
        if args.action == "prepare":
            prepare(args.out)
        elif args.action == "construct":
            construct(args.out, args.seconds)
        elif args.action == "evaluate":
            evaluate(args.out, args.mode)
        else:
            analyze(args.out, args.mode)
