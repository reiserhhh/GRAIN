#!/usr/bin/env python3
"""M29: nine frozen functions on existing source authors, using only the prefix."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.direct_traits import AXES, BIG5
from grain_research.vector_mechanisms.factor_catalog import association, bh_adjust
from grain_research.vector_mechanisms.functional_profiles import FunctionalQuery
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write

S9 = BASE / "source_remainder_s9_r1"
M10 = BASE / "prefix_source_connection_v1"
M8 = BASE / "complete_level_response_v1"
D1 = BASE / "d1_author_discovery_r1"


def prepare(out, functions, states):
    """Bind the old three-view source sample and newly constructed function bytes."""
    from transformers import AutoTokenizer

    from grain_research.vector_mechanisms.controls import material_hash
    from scripts.run_grain_level_response import question_mask
    from scripts.run_grain_real_author_path import MODEL

    if (
        sha(S9 / "private_material.json")
        != read(S9 / "manifest.json")["material_sha256"]
    ):
        raise ValueError("Source material changed")
    if read(functions / "summary.json")["status"] != "COMPLETE_M28":
        raise ValueError("Complete nine-query construction required")
    if (
        sha(functions / "nine_query_functions.npz")
        != read(functions / "summary.json")["functions_sha256"]
    ):
        raise ValueError("Function archive changed")
    source = read(S9 / "private_material.json")
    if len(source) != 277 or len({r["author_id"] for r in source}) != 277:
        raise ValueError("Use the exact 213 Big5 / 64 MBTI source panel")
    construction = {d["author_id"] for d in read(states / "private_design.json")}
    if construction & {r["author_id"] for r in source}:
        raise ValueError("Source/function author overlap")
    qtokens = (
        read(M8 / "manifest.json")["question_tokens"]
        | read(states / "manifest.json")["question_tokens"]
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    design, reused, hashes = [], 0, {}
    for row in source:
        if row["source_group"] not in {"big5_only", "mbti_only"}:
            raise ValueError("Non-source group is forbidden")
        for q in BIG5 if row["source_group"] == "big5_only" else AXES:
            views = []
            for view in row["views"]:
                _, _, encoded, mask = question_mask(tokenizer, view["texts"], q)
                ids = encoded["input_ids"]
                if len(ids) > 3056 or np.asarray(ids)[mask].tolist() != qtokens[q]:
                    raise ValueError("Source question or context budget changed")
                if q == row["query"] and material_hash(ids) != view["prompt_sha256"]:
                    raise ValueError("Old source prompt does not replay")
                views.append({"input_ids": ids, "question_mask": mask.tolist()})
            if len(views) != 3:
                raise ValueError("Exactly three source views required")
            reuse = None
            if q == row["query"]:
                p = M10 / "private_packets" / f"author_{row['index']:03d}.json"
                meta = read(p)
                if (
                    meta["author_id"] != row["author_id"]
                    or meta["query"] != q
                    or sha(p.with_suffix(".npz")) != meta["arrays_sha256"]
                ):
                    raise ValueError("Reused source arrays changed")
                reuse = str(p.relative_to(ROOT))
                hashes[reuse] = sha(p)
                hashes[str(p.with_suffix(".npz").relative_to(ROOT))] = meta[
                    "arrays_sha256"
                ]
                reused += 1
            design.append(
                {
                    "index": len(design),
                    "author_id": row["author_id"],
                    "query": q,
                    "source_group": row["source_group"],
                    "views": views,
                    "reuse": reuse,
                }
            )
    if len(design) != 1321 or reused != 277:
        raise ValueError("Unexpected source-query design")
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    write(out / "private_design.json", design)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_M29_ALL_FACTOR_SOURCE_V1",
            "units": len(design),
            "reused_units": reused,
            "new_prefix_calls": 3132,
            "per_run_seconds": 1200,
            "functions": str(functions.resolve()),
            "functions_sha256": sha(functions / "nine_query_functions.npz"),
            "input_sha256": sha(out / "private_design.json"),
            "source_sha256": sha(S9 / "private_material.json"),
            "source_outcome_sha256": sha(D1 / "private_predictions.csv"),
            "reuse_sha256": hashes,
            "model_sha256": sha(MODEL / "model.safetensors"),
            "code_sha256": sha(Path(__file__)),
            "human_labels_read": False,
        },
    )
    write(
        out / "run_status.json",
        {"status": "PREPARED", "calls": 0, "seconds": 0, "completed_units": 0},
    )
    print(
        {
            "status": "PREPARED",
            "source_authors": 277,
            "author_queries": len(design),
            "new_prefix_calls": 3132,
        },
        flush=True,
    )


def collect(out):
    """Recover or reuse exactly three P/Q states per author/query; no suffix."""
    import torch

    from grain_research.vector_mechanisms.prefix import capture_prefix
    from grain_research.vector_mechanisms.trace import DecoderObserver
    from scripts.run_grain_real_author_path import MODEL

    m, prior = read(out / "manifest.json"), read(out / "run_status.json")
    if prior["status"] == "COMPLETE":
        return
    if (
        sha(Path(__file__)) != m["code_sha256"]
        or sha(out / "private_design.json") != m["input_sha256"]
        or sha(MODEL / "model.safetensors") != m["model_sha256"]
    ):
        raise ValueError("Frozen input/code/model changed")
    for path, digest in m["reuse_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError("Existing prefix archive changed")
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    current, old = observer.manifest(), read(M8 / "observer.json")
    for key in (
        "torch",
        "transformers",
        "dtype",
        "device",
        "attention",
        "template_sha256",
        "pad_to_length",
    ):
        if current[key] != old[key]:
            raise ValueError(f"Frozen observer changed: {key}")
    write(out / "observer.json", current)
    start, complete = time.monotonic(), 0

    def status(state):
        """Record private-text-free progress at a complete three-view unit."""
        value = {
            "status": state,
            "completed_units": complete,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "human_labels_read": False,
        }
        write(out / "run_status.json", value)
        return value

    try:
        for d in read(out / "private_design.json"):
            p = out / "private_packets" / f"unit_{d['index']:04d}.json"
            if p.exists():
                if sha(p.with_suffix(".npz")) != read(p)["array_sha256"]:
                    raise ValueError("Completed source unit changed")
                complete += 1
                continue
            if time.monotonic() - start > m["per_run_seconds"]:
                print(status("PAUSED_RESOURCE_BOUNDARY"), flush=True)
                return
            if d["reuse"]:
                with np.load(
                    (ROOT / d["reuse"]).with_suffix(".npz"), allow_pickle=False
                ) as z:
                    Ps, Qs = z["P"].copy(), z["Q"].copy()
            else:
                Ps, Qs = [], []
                for view in d["views"]:
                    cache, trace = capture_prefix(observer, view["input_ids"])
                    if trace["forbidden_calls"]:
                        raise ValueError("Prefix-only collector invoked suffix")
                    mask = np.asarray(view["question_mask"], bool)
                    Ps.append(cache["boundary"][mask])
                    Qs.append(cache["boundary"][-1])
            np.savez_compressed(
                p.with_suffix(".npz"), P=np.asarray(Ps), Q=np.asarray(Qs)
            )
            write(
                p,
                {k: d[k] for k in ("index", "author_id", "query", "source_group")}
                | {
                    "array_sha256": sha(p.with_suffix(".npz")),
                    "reused": d["reuse"] is not None,
                },
            )
            complete += 1
            if complete % 24 == 0:
                print(status("COLLECTING"), flush=True)
        print(status("COMPLETE"), flush=True)
    except Exception:
        status("ERROR")
        raise


def analyze(out):
    """Freeze all function values, then join only the already exposed source Y."""
    m = read(out / "manifest.json")
    if read(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete planned source prefixes first")
    p = Path(m["functions"]) / "nine_query_functions.npz"
    if sha(p) != m["functions_sha256"]:
        raise ValueError("Fixed functions changed")
    with np.load(p, allow_pickle=False) as z:
        functions = {
            q: FunctionalQuery(*(z[f"{q}__{k}"] for k in ("Z", "h", "G", "g", "b")))
            for q in (*BIG5, *AXES)
        }
    rows = []
    for d in read(out / "private_design.json"):
        path = out / "private_packets" / f"unit_{d['index']:04d}.json"
        meta = read(path)
        if (
            any(
                meta[k] != d[k] for k in ("index", "author_id", "query", "source_group")
            )
            or sha(path.with_suffix(".npz")) != meta["array_sha256"]
        ):
            raise ValueError("Source join changed")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
            if len(z["P"]) != 3 or len(z["Q"]) != 3:
                raise ValueError("Incomplete author view coverage")
            parts = [
                functions[d["query"]].parts(P, Q)
                for P, Q in zip(z["P"], z["Q"], strict=True)
            ]
        rows.append(
            {k: d[k] for k in ("author_id", "query", "source_group")}
            | {k: float(np.mean([r[k] for r in parts])) for k in parts[0]}
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "private_predictions.csv", index=False)
    write(
        out / "prediction_freeze.json",
        {
            "sha256": sha(out / "private_predictions.csv"),
            "human_labels_read_before_freeze": False,
        },
    )
    if sha(D1 / "private_predictions.csv") != m["source_outcome_sha256"]:
        raise ValueError("Frozen source outcomes changed")
    labels = pd.read_csv(D1 / "private_predictions.csv", dtype={"author_id": str})
    labels = labels[
        (labels.observer == "native_query") & (labels.arm == "cue_mask")
    ].copy()
    labels["query"] = labels.trait.map(lambda t: t.lower() if t.lower() in BIG5 else t)
    joined = frame.merge(
        labels[["author_id", "query", "y_true", "prediction"]],
        on=["author_id", "query"],
        validate="one_to_one",
    )
    if len(joined) != len(frame):
        raise ValueError("Missing source criterion")
    joined.to_csv(out / "private_external_links.csv", index=False)
    metrics, covs = [], []
    for q, group in joined.groupby("query"):
        binary = q in AXES
        for arm in ("score", "mean", "configuration", "final", "prediction"):
            metrics.append(
                {
                    "query": q,
                    "arm": arm,
                    **association(group[arm], group.y_true, binary=binary),
                }
            )
        cov = lambda x, y: float(np.mean((x - x.mean()) * (y - y.mean())))
        contributions = {
            k: cov(group[k].to_numpy(), group.y_true.to_numpy())
            for k in ("mean", "configuration", "final")
        }
        total = cov(group.score.to_numpy(), group.y_true.to_numpy())
        covs.append(
            {
                "query": q,
                **contributions,
                "total": total,
                "accounting_error": abs(total - sum(contributions.values())),
            }
        )
    metrics = pd.DataFrame(metrics)
    metrics["q_bh_within_arm"] = metrics.groupby("arm").p.transform(bh_adjust)
    metrics.to_csv(out / "external_metrics.csv", index=False)
    pd.DataFrame(covs).to_csv(out / "source_component_covariance.csv", index=False)
    write(
        out / "external_summary.json",
        {
            "status": "COMPLETE_M29",
            "authors": frame.author_id.nunique(),
            "author_queries": len(frame),
            "new_fits": 0,
            "score_definition": "three-view mean of frozen affine function, no clipping/calibration",
            "native_control": "D1 cached native cue-mask; observer differences remain in original S9 ledger",
            "function_sha256": m["functions_sha256"],
        },
    )
    print(
        metrics[metrics.arm == "score"][["query", "n", "pearson_r", "auc"]]
        .round(4)
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--functions-dir", type=Path)
    parser.add_argument("--state-dir", type=Path)
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(BASE.resolve()):
        parser.error("Keep private outputs under ignored mechanism results")
    if args.mode == "prepare":
        if args.functions_dir is None or args.state_dir is None:
            parser.error("Specify completed M28 analysis and state folders")
        prepare(args.output_dir, args.functions_dir, args.state_dir)
    elif args.mode == "collect":
        collect(args.output_dir)
    else:
        analyze(args.output_dir)
