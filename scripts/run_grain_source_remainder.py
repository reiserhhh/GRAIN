#!/usr/bin/env python3
"""S9: source-author configuration/observer decomposition without student retuning."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.direct_traits import (
    candidates,
    tokenize_prompt,
)
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.analyze_grain_multiquery_sensitivity import corr
from scripts.run_grain_mean_null_path import mean_null_direction
from scripts.run_grain_multiquery_sensitivity import conditional_torch, token_groups
from scripts.run_grain_real_author_path import BASE, MODEL, key, read, sha, write
from scripts.run_grain_vector_mechanism_external_features import bundle_arms

CONNECTION = BASE / "response_student_source_connection_r2"
CURVE = BASE / "student_learning_curve_s4_r1"
VIEWS = BASE / "vm05_qualification_r2/common_views.json"


def covariance(a, b):
    """Population-normalized sample covariance for additive score accounting."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite([a, b]).all():
        raise ValueError("Require paired finite vectors")
    return float(np.mean((a - a.mean()) * (b - b.mean())))


def component_frame(frame):
    """Decompose old teacher minus fixed student without orthogonality assumptions."""
    out = frame.copy()
    out["observer_difference"] = out.old_teacher - out.new_teacher
    out["configuration_response"] = out.new_teacher - out.mean_score
    out["mean_remainder"] = out.mean_score - out.student
    out["total_remainder"] = out.old_teacher - out.student
    error = (
        out.total_remainder
        - out.observer_difference
        - out.configuration_response
        - out.mean_remainder
    )
    if not np.allclose(error, 0, atol=1e-12):
        raise ValueError("Remainder identity failed")
    return out


def prepare(args):
    """Freeze only previously authorized authors and exact old native prompt hashes."""
    out = args.output_dir
    if out.exists():
        raise ValueError("Preserve existing source-remainder preparation")
    source = CONNECTION / "private_predictions.csv"
    if (
        sha(CONNECTION / "manifest.json")
        != read(CONNECTION / "summary.json")["manifest_sha256"]
    ):
        raise ValueError("Changed connection provenance")
    if (
        sha(CURVE / "private_oof_before_outcomes.csv")
        != read(CURVE / "summary.json")["oof_sha256"]
    ):
        raise ValueError("Changed frozen S4 student")
    target = pd.read_csv(
        source, dtype={"author_id": str}, usecols=["author_id", "query", "native_query"]
    )
    student = pd.read_csv(
        CURVE / "private_oof_before_outcomes.csv", dtype={"author_id": str, "size": str}
    )
    student = student[(student.family == "mean") & (student["size"] == "all")]
    target = target.merge(
        student[["author_id", "query", "teacher", "prediction"]],
        on=["author_id", "query"],
        validate="one_to_one",
    )
    if not np.allclose(target.native_query, target.teacher):
        raise ValueError("S4 and source teacher disagree")
    samples = {
        q: sorted(
            target[target["query"] == q].author_id,
            key=lambda uid: key(uid, "s9_source_731"),
        )[:limit]
        for q, limit in (("neuroticism", 213), ("TF_cont", 64))
    }
    if [len(samples[q]) for q in ("neuroticism", "TF_cont")] != [213, 64]:
        raise ValueError("Missing planned source authors")
    ordered = []
    for i in range(213):
        ordered.append((samples["neuroticism"][i], "neuroticism"))
        if i < 64:
            ordered.append((samples["TF_cont"][i], "TF_cont"))
    wanted = {uid for uid, _ in ordered}
    snapshot = read(BASE / "d1_author_discovery_r1/membership_snapshot.json")
    if not wanted <= set(snapshot["private_author_ids"]):
        raise ValueError("Non-D1 author requested")
    source_views = read(VIEWS)
    if material_hash(source_views) != snapshot["common_material_sha256"]:
        raise ValueError("D1 common text changed")
    bundles = {b["author_id"]: b for b in source_views if b["author_id"] in wanted}
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    caches, hashes = {}, {}
    for path, digest in snapshot["native_inputs"].items():
        p = Path(path)
        if p.parent.name != "private_cache":
            continue
        packet = read(p)
        if packet["author_id"] in wanted:
            if sha(p) != digest:
                raise ValueError("Old native source changed")
            if packet["author_id"] in caches:
                raise ValueError("Duplicate source author cache")
            caches[packet["author_id"]] = packet
            hashes[str(p)] = digest
    if set(caches) != wanted or set(bundles) != wanted:
        raise ValueError("Incomplete paired source material")
    targets = target.set_index(["author_id", "query"])
    rows = []
    for index, (uid, query) in enumerate(ordered):
        expected_group = "big5_only" if query == "neuroticism" else "mbti_only"
        if (
            caches[uid]["source_group"] != expected_group
            or bundles[uid]["source_group"] != expected_group
        ):
            raise ValueError("Wrong source role")
        texts = bundle_arms(bundles[uid])["cue_mask"]
        cached = {
            r["view"]: r
            for r in caches[uid]["readouts"]
            if r["arm"] == "cue_mask" and r["task"] == query
        }
        if set(cached) != {0, 1, 2} or len(texts) != 36:
            raise ValueError("Missing complete three-view native rows")
        views = []
        for view in range(3):
            body = texts[12 * view : 12 * (view + 1)]
            ids, _, _ = tokenize_prompt(body, query, tokenizer)
            if (
                material_hash(ids) != cached[view]["prompt_sha256"]
                or len(ids) != cached[view]["prompt_tokens"]
            ):
                raise ValueError("Exact source prompt tokens changed")
            views.append(
                {
                    "texts": body,
                    "prompt_sha256": material_hash(ids),
                    "old_teacher": cached[view]["conditional_score"],
                }
            )
        native = targets.loc[(uid, query)]
        if not np.isclose(
            np.mean([v["old_teacher"] for v in views]), native.native_query, atol=1e-12
        ):
            raise ValueError("Source mean does not reproduce old teacher")
        rows.append(
            {
                "author_id": uid,
                "query": query,
                "index": index,
                "source_group": expected_group,
                "student": float(native.prediction),
                "views": views,
            }
        )
    old_observer = read(BASE / "d1_direct_snapshot_47/shard_000/actual_observer.json")
    if old_observer["revision"] != MODEL.name:
        raise ValueError("Observer checkpoint differs from source teacher")
    out.mkdir(parents=True)
    (out / "private_packets").mkdir()
    write(out / "private_material.json", rows)
    write(out / "private_source_hashes.json", hashes)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_S9_SOURCE_REMAINDER_V1",
            "authors": len(rows),
            "query_counts": {q: len(names) for q, names in samples.items()},
            "views": len(rows) * 3,
            "material_sha256": sha(out / "private_material.json"),
            "source_cache_hashes_sha256": sha(out / "private_source_hashes.json"),
            "source_predictions_sha256": sha(source),
            "student_oof_sha256": sha(CURVE / "private_oof_before_outcomes.csv"),
            "model_sha256": sha(MODEL / "model.safetensors"),
            "script_sha256": sha(__file__),
            "old_observer": {
                k: old_observer[k]
                for k in (
                    "model",
                    "revision",
                    "dtype",
                    "device",
                    "attention",
                    "pad_to_length",
                )
            },
            "new_observer": {
                "device": args.device,
                "dtype": "float32",
                "pad_to_length": None,
            },
            "max_forwards": len(rows) * 6,
            "max_seconds": 3600,
            "labels_read": False,
        },
    )
    print(
        {
            "status": "PREPARED",
            "authors": len(rows),
            "views": len(rows) * 3,
            "max_forwards": len(rows) * 6,
        },
        flush=True,
    )


def score_pair(observer, ids, alphabet, groups, collapse):
    """Score unchanged or all-comment mean-projected states under the same observer."""

    def boundary(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        state = (
            h + mean_null_direction(h[0], groups, list(range(len(groups))))[None]
            if collapse
            else h
        )
        return (state, *output[1:]) if isinstance(output, tuple) else state

    handle = observer.model.model.layers[16].register_forward_hook(boundary)
    try:
        ids = torch.tensor([ids], device=next(observer.model.parameters()).device)
        with torch.no_grad():
            logits = observer.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            ).logits[0, -1]
            value = float(conditional_torch(logits, alphabet).cpu())
        observer.calls += 1
        if not np.isfinite(value):
            raise ValueError("Nonfinite source score")
        return value
    finally:
        handle.remove()


def verify_prepared(out, device):
    """Bind a resumed run to exact material, source, student, model and code."""
    m = read(out / "manifest.json")
    checks = (
        (out / "private_material.json", "material_sha256"),
        (out / "private_source_hashes.json", "source_cache_hashes_sha256"),
        (CONNECTION / "private_predictions.csv", "source_predictions_sha256"),
        (CURVE / "private_oof_before_outcomes.csv", "student_oof_sha256"),
        (MODEL / "model.safetensors", "model_sha256"),
        (Path(__file__), "script_sha256"),
    )
    if (
        any(sha(path) != m[field] for path, field in checks)
        or m["new_observer"]["device"] != device
    ):
        raise ValueError("Prepared source experiment changed")
    return m, read(out / "private_material.json")


def run(args):
    """Collect complete authors; a pilot boundary is a resource checkpoint, not futility."""
    out = args.output_dir
    manifest, material = verify_prepared(out, args.device)
    state_path = out / "run_status.json"
    prior = read(state_path) if state_path.exists() else {"calls": 0, "seconds": 0}
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device=args.device, dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    start = time.monotonic()

    def status(state):
        record = {
            "status": state,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "complete_authors": len(
                list((out / "private_packets").glob("author_*.json"))
            ),
            "labels_read": False,
        }
        write(state_path, record)
        print(record, flush=True)

    try:
        for author in material:
            path = out / "private_packets" / f"author_{author['index']:03d}.json"
            if path.exists():
                if read(path)["author_id"] != author["author_id"]:
                    raise ValueError("Resumed author mismatch")
                continue
            if (
                author["index"] >= args.max_authors
                or prior["seconds"] + time.monotonic() - start
                >= manifest["max_seconds"]
            ):
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            rows = []
            alphabet = candidates(author["query"], observer.tokenizer)
            for view, payload in enumerate(author["views"]):
                ids, user, spans = tokenize_prompt(
                    payload["texts"], author["query"], observer.tokenizer
                )
                if material_hash(ids) != payload["prompt_sha256"]:
                    raise ValueError("Scoring prompt changed")
                groups = token_groups(observer, user, spans, ids)
                original = score_pair(observer, ids, alphabet, groups, False)
                collapsed = score_pair(observer, ids, alphabet, groups, True)
                rows.append(
                    {
                        "view": view,
                        "old_teacher": payload["old_teacher"],
                        "new_teacher": original,
                        "mean_score": collapsed,
                    }
                )
            write(
                path,
                {
                    "author_id": author["author_id"],
                    "query": author["query"],
                    "student": author["student"],
                    "views": rows,
                },
            )
            if (author["index"] + 1) % 8 == 0:
                status("RUNNING")
            if prior["calls"] + observer.calls > manifest["max_forwards"]:
                raise RuntimeError("Forward budget exceeded")
        status("COMPLETE")
    except BaseException as exc:
        status("FAILED")
        write(out / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def evaluate(args):
    """Freeze all author components before joining exposed source outcomes once."""
    out = args.output_dir
    manifest, material = verify_prepared(out, args.device)
    if (out / "summary.json").exists():
        raise ValueError("Preserve completed source analysis")
    packets = [read(p) for p in sorted((out / "private_packets").glob("author_*.json"))]
    if len(packets) != manifest["authors"] or {p["author_id"] for p in packets} != {
        m["author_id"] for m in material
    }:
        raise ValueError("Complete frozen cohort required before outcome connection")
    rows = []
    for p in packets:
        if len(p["views"]) != 3 or {v["view"] for v in p["views"]} != {0, 1, 2}:
            raise ValueError("Incomplete author views")
        rows.append(
            {
                "author_id": p["author_id"],
                "query": p["query"],
                "student": p["student"],
                **{
                    k: float(np.mean([v[k] for v in p["views"]]))
                    for k in ("old_teacher", "new_teacher", "mean_score")
                },
            }
        )
    frame = component_frame(pd.DataFrame(rows))
    path = out / "private_components_before_outcomes.csv"
    frame.to_csv(path, index=False)
    write(out / "components_frozen.json", {"sha256": sha(path), "outcomes_used": False})
    labels = pd.read_csv(
        CONNECTION / "private_predictions.csv",
        dtype={"author_id": str},
        usecols=["author_id", "query", "y_true"],
    )
    frame = frame.merge(labels, on=["author_id", "query"], validate="one_to_one")
    metrics, splits = [], []
    names = (
        "observer_difference",
        "configuration_response",
        "mean_remainder",
        "total_remainder",
    )
    for query, g in frame.groupby("query"):
        for measure in (
            "old_teacher",
            "new_teacher",
            "mean_score",
            "student",
            "configuration_response",
        ):
            metrics.append(
                {
                    "query": query,
                    "measure": measure,
                    **scalar_metrics(g.y_true, g[measure], query == "TF_cont"),
                }
            )
        rng = np.random.default_rng(731)
        indices = rng.integers(len(g), size=(1000, len(g)))
        for measure in names:
            estimates = [
                covariance(g.y_true.to_numpy()[ix], g[measure].to_numpy()[ix])
                for ix in indices
            ]
            splits.append(
                {
                    "query": query,
                    "measure": measure,
                    "covariance": covariance(g.y_true, g[measure]),
                    "covariance_lower": float(np.quantile(estimates, 0.025)),
                    "covariance_upper": float(np.quantile(estimates, 0.975)),
                    "label_r": corr(g.y_true, g[measure]),
                }
            )
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(splits).to_csv(out / "covariance_split.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "components_sha256": sha(path),
            "manifest_sha256": sha(out / "manifest.json"),
            "metrics": metrics,
            "covariance_split": splits,
            "student_refitted": False,
            "scope": "source exploratory covariance of defined model interventions, not causal personality fractions",
        },
    )
    print(pd.DataFrame(splits).to_string(index=False), flush=True)


def main():
    """Prepare, run, or evaluate a frozen bounded source-remainder experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "evaluate"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--max-authors", type=int, default=48)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(BASE):
        raise ValueError("Private outputs belong in ignored results")
    {"prepare": prepare, "run": run, "evaluate": evaluate}[args.mode](args)


if __name__ == "__main__":
    main()
