#!/usr/bin/env python3
"""Complete one missing cell linking historical configuration and query mechanisms."""

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

from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
from grain_research.vector_mechanisms.full_state_edit import replay_suffix
from grain_research.vector_mechanisms.level_response import correlation
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_conditional_baseline import SOURCE
from scripts.run_grain_level_response import question_mask
from scripts.run_grain_mean_null_path import mean_null_direction
from scripts.run_grain_multiquery_sensitivity import native_alphabet
from scripts.run_grain_prefix_source_connection import LEVEL
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write
from scripts.run_grain_source_remainder import covariance

BASELINE = BASE / "conditional_baseline_construction_v1"
S9 = BASE / "source_remainder_s9_r1"
PLAN = ROOT / "docs/GRAIN_CONFIGURATION_QUERY_FACTORIAL_PLAN_V1.md"


def factorial_cells(f11, f01, f10, f00):
    """Return exact anchored contrasts without assigning overlap twice."""
    a, b, c, d = (np.asarray(x, float) for x in (f11, f01, f10, f00))
    if len({x.shape for x in (a, b, c, d)}) != 1 or not all(
        np.isfinite(x).all() for x in (a, b, c, d)
    ):
        raise ValueError("Paired finite cell arrays required")
    return {
        "baseline": d,
        "E_main": c - d,
        "X_main": b - d,
        "interaction": a - c - b + d,
    }


def prepare(out):
    """Freeze the exact source panel, existing cells and new computation."""
    old = read(BASELINE / "manifest.json")
    source_path = S9 / "private_components_before_outcomes.csv"
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    files = [
        Path(__file__),
        PLAN,
        ROOT / "scripts/run_grain_mean_null_path.py",
        ROOT / "grain_research/vector_mechanisms/full_state_edit.py",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_CONFIGURATION_QUERY_FACTORIAL_V1",
            "parent_sha256": sha(BASELINE / "manifest.json"),
            "model_sha256": old["model_sha256"],
            "reference_sha256": old["reference_sha256"],
            "old_components_sha256": sha(source_path),
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "max_calls": 146,
            "max_seconds": 600,
            "old_s9_manifest_sha256": sha(S9 / "manifest.json"),
        },
    )
    write(
        out / "run_status.json",
        {"status": "PREPARED", "calls": 0, "seconds": 0, "authors": 0},
    )
    print({"status": "PREPARED", "authors": 48, "missing_cells": 144})


def find_s9_packets():
    """Index historical source metadata, without logging identifiers or texts."""
    lookup = {}
    for path in (S9 / "private_packets").glob("*.json"):
        if path.name.startswith("."):
            continue
        row = read(path)
        lookup[(row["author_id"], row["query"])] = row
    return lookup


def run(out):
    """Run only the old suffix from cached full states, with resumable finite cost."""
    manifest, prior = read(out / "manifest.json"), read(out / "run_status.json")
    if prior["status"] == "COMPLETE":
        print("Already complete")
        return
    for p, digest in (
        (BASELINE / "manifest.json", manifest["parent_sha256"]),
        (MODEL / "model.safetensors", manifest["model_sha256"]),
        (LEVEL / "reference.npz", manifest["reference_sha256"]),
        (S9 / "manifest.json", manifest["old_s9_manifest_sha256"]),
        *[(ROOT / p, h) for p, h in manifest["code_hashes"].items()],
    ):
        if sha(p) != digest:
            raise ValueError("Frozen factorial input changed")
    parent = read(BASELINE / "manifest.json")
    if sha(BASELINE / "private_design.json") != parent["design_sha256"]:
        raise ValueError("Source design changed")
    design = read(BASELINE / "private_design.json")
    old_packets = find_s9_packets()
    with np.load(LEVEL / "reference.npz", allow_pickle=False) as a:
        ref = dict(a)
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    current, old_observer = observer.manifest(), read(SOURCE / "observer.json")
    for key in (
        "torch",
        "transformers",
        "dtype",
        "device",
        "attention",
        "template_sha256",
    ):
        if current[key] != old_observer[key]:
            raise ValueError("Observer changed")
    write(out / "observer.json", current)
    start, count, seen = time.monotonic(), 0, set()

    def status(value):
        """Write bounded cost only, not per-author data."""
        state = {
            "status": value,
            "authors": count,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "prefix_calls": 0,
            "gradient_calls": 0,
        }
        write(out / "run_status.json", state)
        print(state, flush=True)

    try:
        for row in design:
            query = row["query"]
            dst = out / "private_packets" / f"author_{row['index']:03d}.json"
            if dst.exists():
                count += 1
                seen.add(query)
                continue
            if (
                prior["calls"] + observer.calls + 3 + int(query not in seen)
                > manifest["max_calls"]
                or prior["seconds"] + time.monotonic() - start > manifest["max_seconds"]
            ):
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            source = SOURCE / "private_packets" / dst.name
            for path in (source, source.with_suffix(".npz")):
                if sha(path) != parent["sources"][str(path.relative_to(ROOT))]:
                    raise ValueError("Frozen M11 state or score changed")
            anchor_packet = read(source)
            old = old_packets[(row["author_id"], query)]
            views = []
            with np.load(source.with_suffix(".npz"), allow_pickle=False) as a:
                for i, material in enumerate(row["views"]):
                    h = a[f"view_{i}"]
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
                    if (
                        len(h) != len(encoded["input_ids"])
                        or p[np.concatenate(groups)].any()
                    ):
                        raise ValueError("Fixed E-body groups required")
                    state = torch.as_tensor(
                        h[None], device=next(observer.model.parameters()).device
                    )
                    collapsed = (
                        state
                        + mean_null_direction(
                            state[0], groups, list(range(len(groups)))
                        )[None]
                    )
                    if not torch.equal(collapsed[:, p], state[:, p]) or not torch.equal(
                        collapsed[:, -1], state[:, -1]
                    ):
                        raise ValueError("Configuration projection changed P/Q")
                    mean_error = max(
                        float(
                            torch.max(
                                torch.abs(collapsed[0, g].mean(0) - state[0, g].mean(0))
                            ).cpu()
                        )
                        for g in groups
                    )
                    if mean_error > 2e-5:
                        raise ValueError("Comment means not preserved")
                    alphabet = native_alphabet(query, observer.tokenizer)
                    kwargs = cached_suffix_kwargs(observer.model, collapsed)
                    replay_error = None
                    if query not in seen:
                        replay = replay_suffix(observer, collapsed, kwargs, alphabet)[
                            "conditional_score"
                        ]
                        replay_error = abs(replay - old["views"][i]["mean_score"])
                        if replay_error > 2e-6:
                            raise ValueError("S9 projected cell replay failed")
                        seen.add(query)
                    collapsed[:, p] = torch.as_tensor(
                        ref[f"{query}__Z"], device=state.device, dtype=state.dtype
                    )
                    collapsed[:, -1] = torch.as_tensor(
                        ref[f"{query}__h"], device=state.device, dtype=state.dtype
                    )
                    f00 = replay_suffix(observer, collapsed, kwargs, alphabet)[
                        "conditional_score"
                    ]
                    views.append(
                        {
                            "view": i,
                            "f11": old["views"][i]["new_teacher"],
                            "f01": old["views"][i]["mean_score"],
                            "f10": anchor_packet["views"][i]["anchor"],
                            "f00": f00,
                            "mean_error": mean_error,
                            "projected_replay_error": replay_error,
                        }
                    )
            write(dst, {"author_id": row["author_id"], "query": query, "views": views})
            count += 1
            if count % 8 == 0:
                status("RUNNING")
        status("COMPLETE")
    except BaseException as exc:
        status("FAILED_RETAINED")
        write(out / "error_log.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def evaluate(out):
    """Close all identities before connecting source outcomes and earlier mechanisms."""
    if read(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete cells required")
    records = []
    for row in read(BASELINE / "private_design.json"):
        packet = read(out / "private_packets" / f"author_{row['index']:03d}.json")
        if packet["author_id"] != row["author_id"]:
            raise ValueError("Cell identity changed")
        record = {
            "author_id": row["author_id"],
            "query": row["query"],
            **{
                k: float(np.mean([v[k] for v in packet["views"]]))
                for k in ("f11", "f01", "f10", "f00")
            },
        }
        record.update(
            {
                k: float(v)
                for k, v in factorial_cells(
                    *(record[k] for k in ("f11", "f01", "f10", "f00"))
                ).items()
            }
        )
        records.append(record)
    frame = pd.DataFrame(records)
    frame.to_csv(out / "private_cells_before_outcomes.csv", index=False)
    write(
        out / "cells_frozen.json",
        {
            "sha256": sha(out / "private_cells_before_outcomes.csv"),
            "outcomes_used": False,
        },
    )
    source_path = SOURCE / "private_anchor_source_connection.csv"
    if sha(source_path) != read(BASELINE / "manifest.json")["source_outcome_sha256"]:
        raise ValueError("Existing source link changed")
    source = pd.read_csv(source_path, dtype={"author_id": str})
    frame = frame.merge(
        source[
            [
                "author_id",
                "query",
                "y_true",
                "configuration_response",
                "PQ",
                "native",
                "base_ref",
            ]
        ],
        on=["author_id", "query"],
        validate="one_to_one",
    )
    if (
        len(frame) != 48
        or not np.allclose(frame.f11, frame.native, atol=1e-12)
        or not np.allclose(
            frame.configuration_response, frame.E_main + frame.interaction, atol=1e-12
        )
        or not np.allclose(
            frame.f11,
            frame.baseline + frame.E_main + frame.X_main + frame.interaction,
            atol=1e-12,
        )
    ):
        raise ValueError("Old/new exact factorial linkage failed")
    metrics = []
    for query, group in frame.groupby("query"):
        for name in (
            "f11",
            "baseline",
            "E_main",
            "X_main",
            "interaction",
            "configuration_response",
        ):
            metrics.append(
                {
                    "query": query,
                    "component": name,
                    "authors": len(group),
                    "r_y": correlation(group.y_true.to_numpy(), group[name].to_numpy()),
                    "cov_y": covariance(
                        group.y_true.to_numpy(), group[name].to_numpy()
                    ),
                    "rms": float(np.sqrt(np.mean(group[name] ** 2))),
                    "sd": float(group[name].std(ddof=0)),
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
            "scope": "48 source-exposed diagnostic authors; anchored four-cell identity, not independent variance fractions",
        },
    )
    print(pd.DataFrame(metrics).to_string(index=False))


def main():
    """Run the single missing condition with no new predictive model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "evaluate"))
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "configuration_query_factorial_v1"
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError("Keep private results under ignored research output")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    {"prepare": prepare, "run": run, "evaluate": evaluate}[args.phase](out)


if __name__ == "__main__":
    main()
