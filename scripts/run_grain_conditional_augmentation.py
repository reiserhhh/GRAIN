#!/usr/bin/env python3
"""CON02: one fixed E-only evidence-summary augmentation, never a model grid."""

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

from grain_research.vector_mechanisms.conditional_inputs import (
    evidence_mask,
    evidence_summaries,
)
from grain_research.vector_mechanisms.conditional_models import (
    fit_response_kernel,
    predict_response_kernel,
)
from grain_research.vector_mechanisms.conditional_navigation import response_metrics
from grain_research.vector_mechanisms.navigation import (
    BASE,
    MODEL,
    OUTPUT,
    digest,
    load_functions,
    read_json,
    tokenize_world,
    write_json,
)
from grain_research.vector_mechanisms.state_transport import pack_state
from scripts.analyze_grain_conditional_navigation import EARLY, NATURAL
from scripts.collect_grain_conditional_sources import DEFAULT as SOURCE
from scripts.fit_grain_conditional_navigation import DEFAULT as FIT
from scripts.fit_grain_conditional_navigation import (
    METHODS,
    response_table,
    source_arrays,
    summarize_responses,
)
from scripts.run_grain_navigation_baseline import checked_arrays, save_unit
from scripts.run_grain_source_span_extension import DEFAULT as SPAN
from scripts.run_grain_source_span_extension import OLD, PANEL, load_extension
from scripts.run_grain_state_transport import load_operators, load_runtime
from scripts.run_grain_upstream_routes import collector_guard

DEFAULT = BASE / "conditional_navigation_v1/CON-02/evidence_e_r1"
QUERY = "extraversion"


def prepare(out):
    """Pin all 168 missing summaries before collecting or reading new outcomes."""
    from transformers import AutoTokenizer

    if out.exists() or not out.is_relative_to(
        (BASE / "conditional_navigation_v1").resolve()
    ):
        raise ValueError("Choose a fresh ignored directory")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    units, sources = [], {}
    for kind, folder, dn, wn in (
        ("synthetic", OUTPUT, "design.json", "worlds.json"),
        ("reference", PANEL, "private_design.json", "private_worlds.json"),
        ("natural", NATURAL, "private_design.json", "private_worlds.json"),
    ):
        worlds = {w["world_id"]: w for w in read_json(folder / wn)}
        for name in (dn, wn):
            sources[str((folder / name).relative_to(ROOT))] = digest(folder / name)
        for d in read_json(folder / dn):
            w = worlds[d["world_id"]]
            if (
                d["query"] != QUERY
                or (kind == "synthetic" and w["split"] != "construction")
                or (kind == "reference" and w["role"] != "reference")
            ):
                continue
            _, spans, encoded, mask, _ = tokenize_world(tokenizer, w["texts"], QUERY)
            if encoded["input_ids"] != d["input_ids"] or mask.tolist() != d["mask"]:
                raise ValueError("Frozen prompt/token coordinates changed")
            units.append(
                {
                    "index": len(units),
                    "source_index": d["index"],
                    "kind": kind,
                    "world_id": d["world_id"],
                    "author_id": w.get("author_id", ""),
                    "view": w.get("view", -1),
                    "input_ids": d["input_ids"],
                    "mask": d["mask"],
                    "body": evidence_mask(
                        encoded["offset_mapping"], spans, mask
                    ).tolist(),
                }
            )
    counts = pd.Series([u["kind"] for u in units]).value_counts().to_dict()
    if counts != {"natural": 96, "synthetic": 48, "reference": 24}:
        raise ValueError("Unexpected fixed collection membership")
    out.mkdir(parents=True)
    (out / "private_units").mkdir()
    write_json(out / "private_design.json", units)
    write_json(
        out / "manifest.json",
        {
            "protocol": "CON02_E_FIXED_SUMMARY_V1",
            "units": 168,
            "sources": sources,
            "design_sha256": digest(out / "private_design.json"),
            "code_sha256": digest(Path(__file__)),
            "plan_sha256": digest(
                ROOT / "docs/GRAIN_CONDITIONAL_NAVIGATION_PLAN_V1.md"
            ),
            "summary_sha256": digest(
                ROOT / "grain_research/vector_mechanisms/conditional_inputs.py"
            ),
            "parent_fit_sha256": digest(FIT / "construction_freeze.json"),
            "human_labels_fit": False,
            "full_model_calls": 0,
        },
    )
    write_json(
        out / "run_status.json", {"status": "PREPARED", "calls": 0, "seconds": 0}
    )


def collect(out, seconds):
    """Collect four-block summaries with resumable committed unit markers."""
    m = read_json(out / "manifest.json")
    if m["code_sha256"] != digest(Path(__file__)) or m["design_sha256"] != digest(
        out / "private_design.json"
    ):
        raise ValueError("Frozen augmentation changed")
    if any(digest(ROOT / n) != sha for n, sha in m["sources"].items()):
        raise ValueError("Frozen materials changed")
    design = read_json(out / "private_design.json")
    prior = read_json(out / "run_status.json")
    if prior["status"] == "COMPLETE":
        return
    _, runtime = load_runtime(OLD)
    start, calls = time.monotonic(), 0
    for u in design:
        p = out / "private_units" / f"unit_{u['index']:04d}.json"
        if p.exists():
            meta, _ = checked_arrays(p)
            if meta["identity"] != {k: u[k] for k in ("index", "world_id", "kind")}:
                raise ValueError("Cached summary membership changed")
            continue
        if time.monotonic() - start >= seconds:
            break
        h = runtime.capture(u["input_ids"])[3]
        calls += 1
        mask = np.asarray(u["mask"], bool)
        save_unit(
            p,
            {"identity": {k: u[k] for k in ("index", "world_id", "kind")}},
            pq=pack_state(h[mask], h[-1]),
            summary=evidence_summaries(h, u["body"], mask),
        )
        if calls % 24 == 0:
            print(
                {
                    "phase": "SUMMARY",
                    "new_calls": calls,
                    "seconds": round(time.monotonic() - start, 2),
                },
                flush=True,
            )
    files = {
        f"private_units/unit_{u['index']:04d}.json": digest(
            out / "private_units" / f"unit_{u['index']:04d}.json"
        )
        for u in design
        if (out / "private_units" / f"unit_{u['index']:04d}.json").exists()
    }
    status = {
        "status": "COMPLETE" if len(files) == 168 else "PAUSED",
        "calls": prior["calls"] + calls,
        "seconds": prior["seconds"] + time.monotonic() - start,
        "units": len(files),
    }
    write_json(out / "run_status.json", status)
    if status["status"] == "COMPLETE":
        write_json(
            out / "collection_freeze.json",
            {"files": files, "manifest_sha256": digest(out / "manifest.json")},
        )
    print(status, flush=True)


def augmented_inputs(out):
    """Append fixed summaries to exactly the previous156 source observations."""
    functions, extension = (
        load_functions(OUTPUT / "synthetic_reference.npz"),
        load_extension(SPAN),
    )
    x, r, meta, edges, reader, bias = source_arrays(
        SOURCE, QUERY, extension[QUERY], functions[QUERY]
    )
    freeze = read_json(out / "collection_freeze.json")
    if any(digest(out / n) != sha for n, sha in freeze["files"].items()):
        raise ValueError("Collected summary changed")
    grouped, natural = {"synthetic": [], "reference": {}}, {}
    for d in read_json(out / "private_design.json"):
        _, a = checked_arrays(out / "private_units" / f"unit_{d['index']:04d}.json")
        if d["kind"] == "synthetic":
            grouped["synthetic"].append(a)
        elif d["kind"] == "reference":
            grouped["reference"].setdefault(d["author_id"], []).append(a)
        else:
            natural[d["source_index"]] = a
    summaries = [a["summary"].ravel() for a in grouped["synthetic"]]
    replay = [a["pq"] for a in grouped["synthetic"]]
    for row in meta[48:60]:
        a = grouped["reference"][row["author_id"]]
        if len(a) != 2:
            raise ValueError("Each reference author needs exactly two views")
        summaries.append(np.mean([v["summary"] for v in a], axis=0).ravel())
        replay.append(np.mean([v["pq"] for v in a], axis=0))
    replay_error = float(np.max(np.abs(np.asarray(replay) - x[:60])))
    if replay_error > 1e-5:
        raise ValueError(f"Old source PQ replay mismatch: {replay_error}")
    for d in read_json(SOURCE / "private_design.json"):
        if d["query"] == QUERY:
            _, a = checked_arrays(
                SOURCE / "private_packets" / f"unit_{d['index']:04d}.json"
            )
            summaries.append(a["evidence_summary"].ravel())
    summaries = np.asarray(summaries)
    if len(summaries) != len(x):
        raise ValueError("Summary/source rows diverged")
    return (
        np.c_[x, summaries],
        r,
        meta,
        edges,
        np.vstack([reader, np.zeros((summaries.shape[1], 3))]),
        bias,
        functions[QUERY].anchor,
        natural,
        replay_error,
    )


def fit(out):
    """Repeat the frozen four variants with augmented input, then freeze natural scores."""
    dest = out / "fit"
    if dest.exists() or read_json(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete collection and choose a fresh fit directory")
    x, r, meta, edges, reader, bias, anchor, natural, error = augmented_inputs(out)
    dest.mkdir()
    gram, base = x @ x.T, anchor + (x @ reader + bias).sum(1)
    authors = sorted({d["author_id"] for d in meta if d["kind"] == "conditional"})
    splits = [
        (
            "author_holdout",
            [i for i, d in enumerate(meta) if d["author_id"] != a],
            [
                i
                for i, d in enumerate(meta)
                if d["author_id"] == a and d["kind"] == "conditional"
            ],
        )
        for a in authors
    ]
    splits += [
        (
            "holdout_" + condition,
            [i for i, d in enumerate(meta) if d["condition"] != condition],
            [
                i
                for i, d in enumerate(meta)
                if d["condition"] in (condition, counterpart)
            ],
        )
        for condition, counterpart in (("denied", "endorsed"), ("past", "today"))
    ]
    rows, files = [], {}
    for split, train, test in splits:
        for method, (degree, secant) in METHODS.items():
            model = fit_response_kernel(
                gram, r, train, edges, degree=degree, secant=secant
            )
            pred = base[test] + predict_response_kernel(model, gram[test]).sum(1)
            for i, v in zip(test, pred, strict=True):
                rows.append(
                    {
                        "query": QUERY,
                        "split": split,
                        "method": method,
                        "author_id": meta[i]["author_id"],
                        "condition": meta[i]["condition"],
                        "score": float(v),
                        "fixed": meta[i]["fixed"],
                        "native": meta[i]["native"],
                    }
                )
    source = pd.DataFrame(rows)
    source.to_csv(dest / "private_source_predictions.csv", index=False)
    summarize_responses(response_table(source)).to_csv(
        dest / "source_response_metrics.csv", index=False
    )
    path = dest / f"private_inputs_{QUERY}.npz"
    np.savez_compressed(path, x=x, reader=reader, bias=bias, anchor=np.asarray(anchor))
    files[path.name] = digest(path)
    for method, (degree, secant) in METHODS.items():
        model = fit_response_kernel(
            gram, r, np.arange(len(x)), edges, degree=degree, secant=secant
        )
        path = dest / f"program_{QUERY}_{method}.npz"
        np.savez_compressed(path, **model)
        files[path.name] = digest(path)
    write_json(
        dest / "construction_freeze.json",
        {
            "files": files,
            "collection_sha256": digest(out / "collection_freeze.json"),
            "code_sha256": digest(Path(__file__)),
            "human_labels_used": False,
            "old_source_pq_max_error": error,
            "augmentation": "excerpt_mean+query_cosine_mean",
        },
    )
    _, old, _ = load_operators(OLD)
    worlds = {w["world_id"]: w for w in read_json(NATURAL / "private_worlds.json")}
    rows, errors = [], []
    for d in read_json(NATURAL / "private_design.json"):
        if d["query"] != QUERY:
            continue
        _, z = checked_arrays(EARLY / "private_units" / f"unit_{d['index']:04d}.json")
        cached = z["residual_mean"] - old[QUERY]["mean_r"]
        a = natural[d["index"]]
        errors.append(float(np.max(np.abs(cached - a["pq"]))))
        xx = np.r_[cached, a["summary"].ravel()]
        for method in METHODS:
            with np.load(
                dest / f"program_{QUERY}_{method}.npz", allow_pickle=False
            ) as z:
                score = (
                    anchor
                    + float((xx @ reader + bias).sum())
                    + float(predict_response_kernel(dict(z), (xx @ x.T)[None]).sum())
                )
            w = worlds[d["world_id"]]
            rows.append(
                {
                    "query": QUERY,
                    "world_id": d["world_id"],
                    "split": "natural_exposed",
                    "method": method,
                    "author_id": w["author_id"],
                    "condition": w["condition"],
                    "score": score,
                }
            )
    if max(errors) > 1e-5:
        raise ValueError("Natural PQ replay differs")
    predictions = pd.DataFrame(rows)
    p = dest / "private_natural_predictions.csv"
    predictions.to_csv(p, index=False)
    write_json(
        dest / "prediction_freeze.json",
        {
            "sha256": digest(p),
            "construction_sha256": digest(dest / "construction_freeze.json"),
            "human_labels_used": False,
            "targets_joined": False,
            "natural_pq_max_error": max(errors),
        },
    )
    targetpath = NATURAL / "predictions.csv"
    if digest(targetpath) != read_json(NATURAL / "prediction_freeze.json")["sha256"]:
        raise ValueError("Natural targets changed")
    targets = (
        pd.read_csv(targetpath)
        .pivot(index=["query", "world_id"], columns="readout", values="score")
        .reset_index()
        .rename(columns={"R_synth": "fixed"})
    )
    joined = predictions.merge(
        targets[["query", "world_id", "fixed", "native"]],
        on=["query", "world_id"],
        validate="many_to_one",
    )
    responses = response_table(joined)
    responses.to_csv(dest / "private_natural_responses.csv", index=False)
    summarize_responses(responses).to_csv(
        dest / "natural_response_metrics.csv", index=False
    )
    levels = [
        {
            "query": QUERY,
            "method": method,
            "target": t,
            **response_metrics(g[t], g.score),
        }
        for method, g in joined.groupby("method")
        for t in ("fixed", "native")
    ]
    pd.DataFrame(levels).to_csv(dest / "natural_level_metrics.csv", index=False)
    oldresponses = pd.read_csv(FIT / "private_natural_responses.csv")
    olddelta = pd.read_csv(FIT / "paired_error_intervals.csv")
    comparisons = []
    for (method, axis), g in responses.groupby(["method", "axis"]):
        oldg = (
            oldresponses.query("query==@QUERY and method==@method and axis==@axis")
            .set_index("author_id")
            .reindex(g.author_id)
        )
        incremental = (g.score - g.native).to_numpy() ** 2 - (
            oldg.score - oldg.native
        ).to_numpy() ** 2
        delta0 = (
            olddelta.query("query==@QUERY and method==@method and axis==@axis")
            .iloc[0]
            .delta_mse_vs_A_source
        )
        comparisons.append(
            {
                "query": QUERY,
                "method": method,
                "axis": axis,
                "delta_mse_vs_PQ": float(incremental.mean()),
                "delta_mse_vs_A_source": float(delta0 + incremental.mean()),
            }
        )
    pd.DataFrame(comparisons).to_csv(dest / "paired_error_intervals.csv", index=False)
    write_json(
        dest / "summary.json",
        {
            "status": "COMPLETE",
            "query": QUERY,
            "source_rows": len(x),
            "dimensions": x.shape[1],
            "calls": 168,
            "full_model_calls": 0,
            "human_labels_used": False,
            "old_source_pq_max_error": error,
            "natural_pq_max_error": max(errors),
            "extra_search_permitted": False,
        },
    )
    print(read_json(dest / "summary.json"), flush=True)


def main():
    """Execute one bounded collection/fit, preserving old frozen programs."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "collect", "fit"))
    p.add_argument("--output-dir", type=Path, default=DEFAULT)
    p.add_argument("--seconds", type=int, default=600)
    a = p.parse_args()
    out = a.output_dir.resolve()
    if not out.is_relative_to((BASE / "conditional_navigation_v1").resolve()):
        raise ValueError("Artifacts must stay in private ignored results")
    with threadpool_limits(limits=2):
        if a.action == "prepare":
            prepare(out)
        elif a.action == "collect":
            with collector_guard(out):
                collect(out, a.seconds)
        else:
            fit(out)


if __name__ == "__main__":
    main()
