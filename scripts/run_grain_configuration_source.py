#!/usr/bin/env python3
"""S25: source N connection of no-fit actual configuration-head contrasts."""

from __future__ import annotations

import argparse
import inspect as pyinspect
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from transformers.models.qwen2 import modeling_qwen2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.configuration_mediation import projected_body
from grain_research.vector_mechanisms.configuration_source import (
    configuration_scores,
    full_head,
    projected_cache,
    replay_heads,
)
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.direct_traits import tokenize_prompt
from grain_research.vector_mechanisms.prefix import capture_prefix
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.analyze_grain_teacher_student_bridge import paired_intervals
from scripts.run_grain_configuration_mediation import verify as verify_s24
from scripts.run_grain_functional_transfer import verify as verify_s20
from scripts.run_grain_multiquery_sensitivity import token_groups
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write
from scripts.run_grain_source_remainder import CONNECTION, covariance


def sources(s24, device):
    """Resolve full source material and frozen endpoints without reading source Y."""
    m24, panel = verify_s24(s24, device)
    if read(s24 / "summary.json")["status"] != "COMPLETE":
        raise ValueError("Complete S24 actual-head validation required")
    s20, s9 = Path(m24["s20"]), Path(m24["s9"])
    _m20, _s14, _s12, material, old_scores = verify_s20(s20, device)
    by_id = {a["author_id"]: a for a in read(s9 / "private_material.json")}
    rows, hashes = [], {}
    for author in material:
        old = by_id[author["author_id"]]
        if old["query"] != "neuroticism" or old["views"] != author["views"]:
            raise ValueError("S9 source views or query differ")
        path = s9 / "private_packets" / f"author_{old['index']:03d}.json"
        packet = read(path)
        if packet["author_id"] != author["author_id"] or {
            v["view"] for v in packet["views"]
        } != {0, 1, 2}:
            raise ValueError("Source identity or complete views differ")
        native = float(np.mean([v["new_teacher"] for v in packet["views"]]))
        projected = float(np.mean([v["mean_score"] for v in packet["views"]]))
        rows.append(
            {
                "author_id": author["author_id"],
                "native_score": native,
                "projected_score": projected,
                "configuration_contrast": native - projected,
            }
        )
        hashes[str(path)] = sha(path)
    for i, author, _p, _e in panel:
        path = s24 / "private_packets" / f"author_{i:03d}.json"
        if read(path)["author_id"] != author["author_id"]:
            raise ValueError("S24 identity differs")
        for p in (path, path.with_suffix(".npz")):
            hashes[str(p)] = sha(p)
    comparison = old_scores.rename(columns={"mean_response": "pairing_score"}).merge(
        pd.DataFrame(rows), on="author_id", validate="one_to_one", sort=False
    )
    if len(comparison) != 213 or comparison.author_id.tolist() != [
        a["author_id"] for a in material
    ]:
        raise ValueError("Complete ordered source comparisons required")
    return s20, material, comparison, hashes


def prepare(args):
    """Freeze source membership, exact observer implementation and no-fit scorer."""
    s20, material, comparison, hashes = sources(args.source, args.device)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    comparison.to_csv(out / "private_comparisons.csv", index=False)
    with np.load(s20 / "private_reference.npz", allow_pickle=False) as z:
        np.savez_compressed(out / "private_reference.npz", reference=z["reference"])
    files = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/configuration_source.py",
        ROOT / "grain_research/vector_mechanisms/prefix.py",
        ROOT / "docs/GRAIN_CONFIGURATION_SOURCE_S25_PLAN.md",
    ]
    qwen_file = Path(pyinspect.getfile(modeling_qwen2))
    pilot = list(range(24, 32))
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_S25_CONFIGURATION_SOURCE_V1",
            "source": str(args.source),
            "source_manifest_sha256": sha(args.source / "manifest.json"),
            "source_summary_sha256": sha(args.source / "summary.json"),
            "packet_hashes": hashes,
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "qwen_implementation": str(qwen_file),
            "qwen_sha256": sha(qwen_file),
            "comparison_sha256": sha(out / "private_comparisons.csv"),
            "reference_sha256": sha(out / "private_reference.npz"),
            "source_y_sha256": sha(CONNECTION / "private_predictions.csv"),
            "membership_sha256": material_hash([a["author_id"] for a in material]),
            "order": pilot + [i for i in range(213) if i not in pilot],
            "pilot": pilot,
            "authors": 213,
            "views": 3,
            "gradient_reference_authors": 24,
            "method": args.method,
            "device": args.device,
            "max_prefix_calls": 639 if args.method == "local" else 1278,
            "max_seconds": 1200,
            "outcomes_used_in_score": False,
        },
    )
    print({"status": "PREPARED", "authors": 213, "method": args.method}, flush=True)


def verify(out, device):
    """Bind every resume/analysis to the original plan and complete source panel."""
    m = read(out / "manifest.json")
    source = Path(m["source"])
    checks = [
        (source / "manifest.json", m["source_manifest_sha256"]),
        (source / "summary.json", m["source_summary_sha256"]),
        (out / "private_comparisons.csv", m["comparison_sha256"]),
        (out / "private_reference.npz", m["reference_sha256"]),
        (Path(m["qwen_implementation"]), m["qwen_sha256"]),
        *[(Path(p), h) for p, h in m["packet_hashes"].items()],
        *[(ROOT / p, h) for p, h in m["code_hashes"].items()],
    ]
    if device != m["device"] or any(sha(p) != h for p, h in checks):
        raise ValueError("Frozen S25 source or implementation changed")
    _s20, material, _comparison, _hashes = sources(source, device)
    if material_hash([a["author_id"] for a in material]) != m["membership_sha256"]:
        raise ValueError("Author membership changed")
    return m, material


def validate_packet(path, author, reference):
    """Validate complete heads, view identities and frozen score reconstruction."""
    row = read(path)
    if (
        row["author_id"] != author["author_id"]
        or sha(path.with_suffix(".npz")) != row["arrays_sha256"]
    ):
        raise ValueError("Packet identity or arrays changed")
    if [v["prompt_sha256"] for v in row["views"]] != [
        v["prompt_sha256"] for v in author["views"]
    ]:
        raise ValueError("Three-view prompt order differs")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
        delta, scores, mean = configuration_scores(
            z["native_heads"], z["projected_heads"], reference
        )
        if delta.shape != (3, len(reference)) or not np.allclose(
            delta, z["delta_heads"], atol=1e-15, rtol=1e-12
        ):
            raise ValueError("Invalid complete head differences")
    if not np.allclose(
        scores, [v["prediction"] for v in row["views"]], atol=1e-15, rtol=1e-12
    ) or not np.isclose(mean, row["mean_prediction"], atol=1e-15, rtol=1e-12):
        raise ValueError("Frozen head score replay differs")
    return row


def run(args):
    """Collect a bounded pilot or full source head contrasts without source outcomes."""
    out = args.output_dir
    m, material = verify(out, args.device)
    with np.load(out / "private_reference.npz", allow_pickle=False) as z:
        reference = z["reference"]
    previous = (
        read(out / "run_status.json")
        if (out / "run_status.json").exists()
        else {"prefix_calls": 0, "local_reprojections": 0, "seconds": 0}
    )
    if args.max_authors > 8:
        checkpoint = read(out / "checkpoint_008.json")
        if not checkpoint["continue_worthwhile"] or checkpoint[
            "manifest_sha256"
        ] != sha(out / "manifest.json"):
            raise ValueError(
                "Passed numerical/resource pilot required before extension"
            )
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device=args.device, dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    start, reprojections = time.monotonic(), 0

    def status(name):
        state = {
            "status": name,
            "authors": len(list((out / "private_packets").glob("author_*.json"))),
            "prefix_calls": previous["prefix_calls"] + observer.calls,
            "local_reprojections": previous["local_reprojections"] + reprojections,
            "seconds": previous["seconds"] + time.monotonic() - start,
            "full_suffix_calls": 0,
            "backwards": 0,
            "source_outcomes_read": False,
        }
        write(out / "run_status.json", state)
        print(state, flush=True)

    try:
        with threadpool_limits(limits=2):
            for ordinal, i in enumerate(m["order"]):
                author = material[i]
                path = out / "private_packets" / f"author_{i:03d}.json"
                if path.exists():
                    validate_packet(path, author, reference)
                    continue
                if ordinal >= args.max_authors:
                    status("PAUSED_PILOT_BOUNDARY")
                    return
                cost = 3 if m["method"] == "local" else 6
                if (
                    previous["seconds"] + time.monotonic() - start >= m["max_seconds"]
                    or previous["prefix_calls"] + observer.calls + cost
                    > m["max_prefix_calls"]
                ):
                    status("PAUSED_RESOURCE_BOUNDARY")
                    return
                native_heads, projected_heads, details = [], [], []
                for view, payload in enumerate(author["views"]):
                    ids, user, spans = tokenize_prompt(
                        payload["texts"], "neuroticism", observer.tokenizer
                    )
                    if material_hash(ids) != payload["prompt_sha256"]:
                        raise ValueError("Native source prompt differs")
                    groups = token_groups(observer, user, spans, ids)
                    cache, trace = capture_prefix(observer, ids)
                    native_head = full_head(cache)
                    if m["method"] == "local":
                        projected, audit = projected_cache(observer, cache, groups)
                        reprojections += 1
                        forbidden = len(trace["forbidden_calls"])
                    else:
                        with projected_body(observer.model.model.layers[16], groups):
                            projected, projected_trace = capture_prefix(observer, ids)
                        audit = {
                            "query_boundary_error": float(
                                np.max(
                                    np.abs(
                                        projected["boundary"][-1]
                                        - cache["boundary"][-1]
                                    )
                                )
                            ),
                            "query_vector_error": float(
                                np.max(np.abs(projected["q"] - cache["q"]))
                            ),
                        }
                        forbidden = len(trace["forbidden_calls"]) + len(
                            projected_trace["forbidden_calls"]
                        )
                    projected_head = full_head(projected)
                    prediction = float(reference @ (native_head - projected_head))
                    replay = None
                    if 24 <= i < 48 and view == 0:
                        old = Path(m["source"]) / "private_packets" / path.name
                        previous_packet = read(old)
                        with np.load(old.with_suffix(".npz"), allow_pickle=False) as z:
                            replay = replay_heads(
                                native_head,
                                projected_head,
                                z["native_head"],
                                z["projected_head"],
                                reference,
                                previous_packet["linear_prediction"],
                            )
                    if forbidden or max(audit.values()) != 0:
                        raise ValueError("Prefix/query invariants differ")
                    native_heads.append(native_head)
                    projected_heads.append(projected_head)
                    details.append(
                        {
                            "view": view,
                            "prediction": prediction,
                            "prompt_sha256": payload["prompt_sha256"],
                            "forbidden_calls": forbidden,
                            **audit,
                            "s24_replay": replay,
                        }
                    )
                    del cache, projected
                delta, _scores, mean = configuration_scores(
                    native_heads, projected_heads, reference
                )
                np.savez_compressed(
                    path.with_suffix(".npz"),
                    native_heads=native_heads,
                    projected_heads=projected_heads,
                    delta_heads=delta,
                )
                write(
                    path,
                    {
                        "author_id": author["author_id"],
                        "source_index": i,
                        "views": details,
                        "mean_prediction": mean,
                        "arrays_sha256": sha(path.with_suffix(".npz")),
                    },
                )
                status("RUNNING")
        status("COMPLETE")
    except Exception as exc:
        status("FAILED")
        write(out / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def collect(out, m, material):
    """Read validated complete packets in canonical source order, not collection order."""
    with np.load(out / "private_reference.npz", allow_pickle=False) as z:
        reference = z["reference"]
    rows, packets = [], []
    for i, author in enumerate(material):
        path = out / "private_packets" / f"author_{i:03d}.json"
        if not path.exists():
            continue
        row = validate_packet(path, author, reference)
        rows.append(
            {
                "author_id": author["author_id"],
                "reference_author": i < m["gradient_reference_authors"],
                "configuration_score": row["mean_prediction"],
                **{
                    f"view{v}_score": x["prediction"]
                    for v, x in enumerate(row["views"])
                },
            }
        )
        packets.append(row)
    return pd.DataFrame(rows), packets


def inspect(args):
    """Assess pilot numerical correctness and decision value before full collection."""
    out = args.output_dir
    m, material = verify(out, args.device)
    frame, packets = collect(out, m, material)
    state = read(out / "run_status.json")
    replays = [
        v["s24_replay"]
        for p in packets
        for v in p["views"]
        if v["s24_replay"] is not None
    ]
    if (
        len(frame) != 8
        or len(replays) != 8
        or state["status"] != "PAUSED_PILOT_BOUNDARY"
    ):
        raise ValueError("Inspect the complete frozen eight-author pilot")
    estimated = state["seconds"] * 213 / len(frame)
    checkpoint = {
        **state,
        "manifest_sha256": sha(out / "manifest.json"),
        "head_replay_error": max(
            max(r["native_head_error"], r["projected_head_error"]) for r in replays
        ),
        "linear_replay_error": max(r["linear_score_error"] for r in replays),
        "predicted_total_seconds": estimated,
        "continue_worthwhile": estimated <= m["max_seconds"],
        "reason": "Useful actual-head shortcut; complete three-view source scores answer the mainline association question, without interim Y access.",
    }
    write(out / "checkpoint_008.json", checkpoint)
    print(checkpoint, flush=True)


def evaluate(args):
    """Freeze213 head scores, then compare authorized source N and existing routes."""
    out = args.output_dir
    m, material = verify(out, args.device)
    state = read(out / "run_status.json")
    if (
        state["status"] != "COMPLETE"
        or state["authors"] != 213
        or (out / "summary.json").exists()
    ):
        raise ValueError("Complete213 author collection and a new analysis required")
    frame, packets = collect(out, m, material)
    if len(frame) != 213 or not frame.author_id.is_unique:
        raise ValueError("Complete unique source scores required")
    frozen = out / "private_scores_before_outcomes.csv"
    frame.to_csv(frozen, index=False)
    write(
        out / "scores_frozen.json",
        {"scores_sha256": sha(frozen), "authors": 213, "human_outcome_fit": False},
    )
    if sha(CONNECTION / "private_predictions.csv") != m["source_y_sha256"]:
        raise ValueError("Authorized source outcomes changed")
    comparisons = pd.read_csv(out / "private_comparisons.csv", dtype={"author_id": str})
    frame = frame.merge(
        comparisons.drop(columns="reference_author"),
        on="author_id",
        validate="one_to_one",
        sort=False,
    )
    labels = pd.read_csv(
        CONNECTION / "private_predictions.csv",
        usecols=["author_id", "query", "y_true"],
        dtype={"author_id": str},
    )
    labels = labels[labels["query"] == "neuroticism"][["author_id", "y_true"]]
    frame = frame.merge(labels, on="author_id", validate="one_to_one", sort=False)
    if (
        len(frame) != 213
        or not np.isfinite(frame.select_dtypes(include="number")).all().all()
    ):
        raise ValueError("Complete finite source comparison and outcome join required")
    metrics, consistency = [], []
    methods = (
        "configuration_score",
        "pairing_score",
        "configuration_contrast",
        "native_score",
        "projected_score",
    )
    for name, group in (
        ("primary189", frame[~frame.reference_author]),
        ("discovery24", frame[frame.reference_author]),
        ("all213", frame),
    ):
        for method in methods:
            metrics.append(
                {
                    "cohort": name,
                    "method": method,
                    "n": len(group),
                    **scalar_metrics(group.y_true, group[method]),
                    **paired_intervals(
                        group.y_true, group[method], group.pairing_score, False
                    ),
                    "covariance_y": covariance(group.y_true, group[method]),
                    "covariance_configuration": covariance(
                        group.configuration_contrast, group[method]
                    ),
                    "configuration_r": scalar_metrics(
                        group.configuration_contrast, group[method]
                    )["pearson_r"],
                    "mean": float(group[method].mean()),
                    "std": float(group[method].std(ddof=0)),
                }
            )
        for method, suffix in (("configuration", "score"), ("pairing", "response")):
            for a, b in ((0, 1), (0, 2), (1, 2)):
                consistency.append(
                    {
                        "cohort": name,
                        "method": method,
                        "view_a": a,
                        "view_b": b,
                        **scalar_metrics(
                            group[f"view{a}_{suffix}"], group[f"view{b}_{suffix}"]
                        ),
                    }
                )
    frame.to_csv(out / "private_source_connections.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(consistency).to_csv(out / "view_consistency.csv", index=False)
    replays = [
        v["s24_replay"]
        for p in packets
        for v in p["views"]
        if v["s24_replay"] is not None
    ]
    summary = {
        "status": "COMPLETE",
        "runtime": state,
        "metrics": metrics,
        "view_consistency": consistency,
        "source_authors": 213,
        "human_outcome_fit": False,
        "s24_replayed_authors": len(replays),
        "scores_sha256": sha(frozen),
        "max_head_replay_error": max(
            max(r["native_head_error"], r["projected_head_error"]) for r in replays
        ),
        "max_linear_replay_error": max(r["linear_score_error"] for r in replays),
    }
    write(out / "summary.json", summary)
    print(
        pd.DataFrame(metrics)[
            [
                "cohort",
                "method",
                "pearson_r",
                "pearson_p",
                "student_lower",
                "student_upper",
                "delta",
                "delta_lower",
                "delta_upper",
                "configuration_r",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(
        {"runtime": state, "max_head_replay_error": summary["max_head_replay_error"]},
        flush=True,
    )


def main():
    """Prepare, pilot eight, inspect, complete213, then join the source outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run", "inspect", "evaluate"])
    parser.add_argument(
        "--source", type=Path, default=BASE / "configuration_mediation_s24_r1"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "configuration_source_s25_r1"
    )
    parser.add_argument("--method", choices=["local", "dual"], default="local")
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--max-authors", type=int, default=8)
    args = parser.parse_args()
    args.source, args.output_dir = args.source.resolve(), args.output_dir.resolve()
    if (
        not args.output_dir.is_relative_to(BASE.resolve())
        or not 1 <= args.max_authors <= 213
    ):
        raise ValueError("Ignored private output and bounded author count required")
    globals()[args.mode](args)


if __name__ == "__main__":
    main()
