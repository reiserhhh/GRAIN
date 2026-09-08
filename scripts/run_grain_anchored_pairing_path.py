#!/usr/bin/env python3
"""Test the known pairing mechanism within actual fixed-P/Q conditional anchors."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.conditional_baseline import project_boundary
from grain_research.vector_mechanisms.configuration_source import full_head
from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
from grain_research.vector_mechanisms.full_state_edit import replay_suffix
from grain_research.vector_mechanisms.level_response import correlation
from grain_research.vector_mechanisms.prefix import prefix_pairing
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_conditional_baseline import SOURCE
from scripts.run_grain_level_response import question_mask
from scripts.run_grain_multiquery_sensitivity import native_alphabet
from scripts.run_grain_prefix_source_connection import LEVEL
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write
from scripts.run_grain_source_remainder import covariance

PLAN = ROOT / "docs/GRAIN_ANCHORED_PAIRING_PATH_PLAN_V1.md"
BASELINE = BASE / "conditional_baseline_construction_v1"


def paired_suffix(observer, state, alphabet, head, direction):
    """Remove only the declared final head component and verify its incoming value."""
    errors = []

    def patch(_module, args):
        value = args[0].clone()
        actual = value[0, -1].detach().float().cpu().numpy()
        error = float(np.max(np.abs(actual - head)))
        if error > 1e-4:
            raise ValueError("Anchored mathematical head differs from actual suffix")
        errors.append(error)
        value[0, -1] -= torch.as_tensor(
            direction, device=value.device, dtype=value.dtype
        )
        return (value, *args[1:])

    handle = observer.model.model.layers[17].self_attn.o_proj.register_forward_pre_hook(
        patch
    )
    try:
        score = replay_suffix(
            observer, state, cached_suffix_kwargs(observer.model, state), alphabet
        )["conditional_score"]
    finally:
        handle.remove()
    if len(errors) != 1:
        raise ValueError("Exactly one final-head intervention required")
    return score, errors[0]


def prepare(out):
    """Freeze old author material and new intervention implementation before inference."""
    inherited = read(BASELINE / "manifest.json")
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    files = [
        Path(__file__),
        PLAN,
        ROOT / "grain_research/vector_mechanisms/conditional_baseline.py",
        ROOT / "grain_research/vector_mechanisms/prefix.py",
        ROOT / "grain_research/vector_mechanisms/full_state_edit.py",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_ANCHORED_PAIRING_V1",
            "parent_sha256": sha(BASELINE / "manifest.json"),
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "max_calls": 146,
            "max_seconds": 600,
            "model_sha256": inherited["model_sha256"],
            "reference_sha256": inherited["reference_sha256"],
            "outcomes_used_for_collection": False,
        },
    )
    write(
        out / "run_status.json",
        {"status": "PREPARED", "calls": 0, "seconds": 0, "authors": 0},
    )
    print({"status": "PREPARED", "authors": 48, "suffix_calls": 146})


def run(out):
    """Continue from immutable cached boundaries, preserving every completed author."""
    m = read(out / "manifest.json")
    previous = read(out / "run_status.json")
    if previous["status"] == "COMPLETE":
        print("Already complete")
        return
    for path, digest in (
        (MODEL / "model.safetensors", m["model_sha256"]),
        (LEVEL / "reference.npz", m["reference_sha256"]),
        (BASELINE / "manifest.json", m["parent_sha256"]),
        *[(ROOT / p, h) for p, h in m["code_hashes"].items()],
    ):
        if sha(path) != digest:
            raise ValueError("Frozen pairing input changed")
    inherited = read(BASELINE / "manifest.json")
    design_path = BASELINE / "private_design.json"
    if sha(design_path) != inherited["design_sha256"]:
        raise ValueError("Fixed source design changed")
    with np.load(LEVEL / "reference.npz", allow_pickle=False) as a:
        reference = dict(a)
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    current, old = observer.manifest(), read(SOURCE / "observer.json")
    for key in (
        "torch",
        "transformers",
        "dtype",
        "device",
        "attention",
        "template_sha256",
    ):
        if current[key] != old[key]:
            raise ValueError("Observer changed")
    write(out / "observer.json", current)
    start, completed, seen = time.monotonic(), 0, set()

    def status(value):
        """Record real finite resource use without text, IDs or questionnaire output."""
        result = {
            "status": value,
            "calls": previous["calls"] + observer.calls,
            "seconds": previous["seconds"] + time.monotonic() - start,
            "authors": completed,
            "prefix_calls": 0,
            "gradient_calls": 0,
        }
        write(out / "run_status.json", result)
        print(result, flush=True)

    try:
        for row in read(design_path):
            query = row["query"]
            dst = out / "private_packets" / f"author_{row['index']:03d}.json"
            if dst.exists():
                completed += 1
                seen.add(query)
                continue
            cost = 3 + int(query not in seen)
            if (
                previous["calls"] + observer.calls + cost > m["max_calls"]
                or previous["seconds"] + time.monotonic() - start > m["max_seconds"]
            ):
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            source_packet = SOURCE / "private_packets" / dst.name
            arrays = source_packet.with_suffix(".npz")
            for path in (source_packet, arrays):
                if sha(path) != inherited["sources"][str(path.relative_to(ROOT))]:
                    raise ValueError("Fixed anchor packet changed")
            source = read(source_packet)
            views, directions = [], []
            with np.load(arrays, allow_pickle=False) as a:
                for i, material in enumerate(row["views"]):
                    h = a[f"view_{i}"].copy()
                    _, spans, encoded, p = question_mask(
                        observer.tokenizer, material["texts"], query
                    )
                    groups = [
                        [
                            j
                            for j, (lo, hi) in enumerate(encoded["offset_mapping"])
                            if hi > lo >= left and hi <= right
                        ]
                        for left, right in spans
                    ]
                    used = np.concatenate(groups)
                    if (
                        len(h) != len(encoded["input_ids"])
                        or p[used].any()
                        or (len(h) - 1) in used
                    ):
                        raise ValueError("Comment groups differ from E-only layout")
                    h[p], h[-1] = reference[f"{query}__Z"], reference[f"{query}__h"]
                    cache = project_boundary(observer.model, h)
                    head = full_head(cache)
                    direction = prefix_pairing(cache, groups)
                    state = torch.as_tensor(
                        h[None], device=next(observer.model.parameters()).device
                    )
                    alphabet = native_alphabet(query, observer.tokenizer)
                    anchor = source["views"][i]["anchor"]
                    replay_error = None
                    if query not in seen:
                        replay = replay_suffix(
                            observer,
                            state,
                            cached_suffix_kwargs(observer.model, state),
                            alphabet,
                        )["conditional_score"]
                        replay_error = abs(replay - anchor)
                        if replay_error > 2e-6:
                            raise ValueError("Native conditional anchor replay failed")
                        seen.add(query)
                    patched, head_error = paired_suffix(
                        observer, state, alphabet, head, direction
                    )
                    views.append(
                        {
                            "view": i,
                            "anchor": anchor,
                            "no_pair": patched,
                            "pair_effect": anchor - patched,
                            "head_error": head_error,
                            "replay_error": replay_error,
                            "pair_norm": float(np.linalg.norm(direction)),
                        }
                    )
                    directions.append(direction)
            np.savez_compressed(dst.with_suffix(".npz"), pairing=directions)
            write(
                dst,
                {
                    "author_id": row["author_id"],
                    "query": query,
                    "views": views,
                    "arrays_sha256": sha(dst.with_suffix(".npz")),
                },
            )
            completed += 1
            if completed % 8 == 0:
                status("RUNNING")
        status("COMPLETE")
    except BaseException as exc:
        status("FAILED_RETAINED")
        write(out / "error_log.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def evaluate(out):
    """Freeze actual vector-intervention values, then account for exposed source links."""
    if read(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Completed intervention required")
    records, errors = [], []
    for row in read(BASELINE / "private_design.json"):
        path = out / "private_packets" / f"author_{row['index']:03d}.json"
        packet = read(path)
        if (
            packet["author_id"] != row["author_id"]
            or sha(path.with_suffix(".npz")) != packet["arrays_sha256"]
        ):
            raise ValueError("Completed pairing identity or array changed")
        records.append(
            {
                "author_id": row["author_id"],
                "query": row["query"],
                **{
                    k: float(np.mean([v[k] for v in packet["views"]]))
                    for k in ("anchor", "no_pair", "pair_effect")
                },
            }
        )
        errors.extend(v["head_error"] for v in packet["views"])
    frame = pd.DataFrame(records)
    frame.to_csv(out / "private_interventions_before_outcomes.csv", index=False)
    write(
        out / "interventions_frozen.json",
        {
            "sha256": sha(out / "private_interventions_before_outcomes.csv"),
            "labels_used": False,
        },
    )
    source_path = SOURCE / "private_anchor_source_connection.csv"
    if sha(source_path) != read(BASELINE / "manifest.json")["source_outcome_sha256"]:
        raise ValueError("Exposed source changed")
    source = pd.read_csv(source_path, dtype={"author_id": str})
    frame = frame.merge(
        source.drop(columns="anchor"), on=["author_id", "query"], validate="one_to_one"
    )
    if len(frame) != 48 or not np.allclose(
        frame.anchor, frame.no_pair + frame.pair_effect, atol=1e-12
    ):
        raise ValueError("Anchored pairing ledger failed")
    metrics = []
    for query, g in frame.groupby("query"):
        for name in (
            "native",
            "anchor",
            "no_pair",
            "pair_effect",
            "configuration_response",
            "transport_remainder",
        ):
            metrics.append(
                {
                    "query": query,
                    "component": name,
                    "authors": len(g),
                    "mean": float(g[name].mean()),
                    "sd": float(g[name].std(ddof=0)),
                    "r_y": correlation(g.y_true.to_numpy(), g[name].to_numpy()),
                    "cov_y": covariance(g.y_true.to_numpy(), g[name].to_numpy()),
                    "r_configuration": correlation(
                        g.configuration_response.to_numpy(), g[name].to_numpy()
                    ),
                }
            )
    frame.to_csv(out / "private_source_connection.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "metrics": metrics,
            "cost": read(out / "run_status.json"),
            "head_replay_max_error": max(errors),
            "scope": "48 source-exposed diagnostic authors; conditional actual intervention, no new label fitting",
        },
    )
    print(pd.DataFrame(metrics).to_string(index=False))


def main():
    """Run one bounded causal-mechanism packet with no model or label optimization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "evaluate"))
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "anchored_pairing_path_v1"
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError("Keep private output under ignored results")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    {"prepare": prepare, "run": run, "evaluate": evaluate}[args.phase](out)


if __name__ == "__main__":
    main()
