#!/usr/bin/env python3
"""Split observed context effects into evidence-side and question-side carriers."""

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

from grain_research.vector_mechanisms.context_carrier import (
    cached_suffix_kwargs,
    query_region_mask,
    split_context_state,
)
from grain_research.vector_mechanisms.full_state_edit import (
    align_states,
    grid_components,
    replay_suffix,
)
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_full_state_edit_connection import SOURCE, token_view
from scripts.run_grain_multiquery_sensitivity import QUERIES, query_prompt
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write

PARENT = BASE / "full_state_edit_connection_v1"


def prepare(out):
    """Bind unchanged authors, state archives, and the template-defined query boundary."""
    if read(PARENT / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete factorial source required")
    parent = read(PARENT / "manifest.json")
    if (
        sha(PARENT / "private_inputs.json") != parent["input_sha256"]
        or sha(SOURCE / "private_material.json") != parent["source_material_sha256"]
    ):
        raise ValueError("Source data changed")
    source = {
        r["author_id"]: r
        for r in read(SOURCE / "private_material.json")
        if r["view"] == 0
    }
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    design = []
    for a in read(PARENT / "private_inputs.json"):
        original = source[a["author_id"]]
        bodies = [t for j, t in enumerate(original["texts"]) if j not in original["e1"]]
        for q in QUERIES:
            p = a["queries"][q]
            text, _spans, encoded = token_view(tokenizer, bodies, q)
            if list(encoded["input_ids"]) != p["edited_ids"]:
                raise ValueError("Edited tokens differ from parent")
            user, _ = query_prompt(bodies, q)
            start = text.index(user) + user.rindex("\nQuestion: ") + 1
            mask = query_region_mask(encoded["offset_mapping"], start)
            path = PARENT / "private_packets" / f"unit_{a['index']:02d}_{q}.json"
            packet = read(path)
            if (
                packet["author_id"] != a["author_id"]
                or sha(path.with_suffix(".npz")) != packet["arrays_sha256"]
            ):
                raise ValueError("State archive changed")
            design.append(
                {
                    "index": a["index"],
                    "author_id": a["author_id"],
                    "role": a["role"],
                    "query": q,
                    "question_mask": mask.tolist(),
                    "alphabet": p["alphabet"],
                    "parent_packet": str(path.relative_to(ROOT)),
                    "packet_sha256": sha(path),
                    "arrays_sha256": packet["arrays_sha256"],
                }
            )
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    write(out / "private_design.json", design)
    files = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/context_carrier.py",
        ROOT / "grain_research/vector_mechanisms/full_state_edit.py",
        ROOT / "docs/GRAIN_CONTEXT_CARRIER_SPLIT_PLAN_V1.md",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_CONTEXT_CARRIER_SPLIT_V1",
            "authors": 4,
            "units": 12,
            "queries": list(QUERIES),
            "max_calls": 36,
            "max_seconds": 300,
            "prefix_calls": 0,
            "gradient_calls": 0,
            "labels_read": False,
            "fixed_final_query": "aligned original Q0",
            "parent_manifest_sha256": sha(PARENT / "manifest.json"),
            "model_sha256": parent["model_sha256"],
            "input_sha256": sha(out / "private_design.json"),
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
        },
    )
    write(
        out / "run_status.json",
        {"status": "PREPARED", "calls": 0, "seconds": 0.0, "completed_units": 0},
    )
    print(
        {
            "status": "PREPARED",
            "units": 12,
            "question_region_token_range": [
                min(sum(d["question_mask"]) for d in design),
                max(sum(d["question_mask"]) for d in design),
            ],
        },
        flush=True,
    )


def summarize(out):
    """Report all role-conditioned E/P effects; no fitting, ranking or label joins."""
    rows = []
    for p in sorted((out / "private_packets").glob("unit_*.json")):
        r = read(p)
        g = r["components"]
        rows.append(
            {
                "index": r["index"],
                "role": r["role"],
                "query": r["query"],
                "context_total": g["natural"],
                "evidence_main": g["context_main"],
                "question_main": g["query_main"],
                "interaction": g["interaction"],
                "evidence_allocated": g["context_allocated"],
                "question_allocated": g["query_allocated"],
                "replay_error": r["replay_error"],
            }
        )
    if len(rows) != 12:
        raise ValueError("All twelve fixed cells required")
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "private_effects.csv", index=False)
    metrics = []
    for q, group in frame.groupby("query"):
        for col in (
            "context_total",
            "evidence_main",
            "question_main",
            "interaction",
            "evidence_allocated",
            "question_allocated",
        ):
            metrics.append(
                {
                    "query": q,
                    "component": col,
                    "authors": 4,
                    "mean": float(group[col].mean()),
                    "rms": float(np.sqrt(np.mean(group[col] ** 2))),
                }
            )
    pd.DataFrame(metrics).to_csv(out / "component_metrics.csv", index=False)
    result = {
        "status": "COMPLETE",
        "scope": "four-author finite mechanism pilot; not population inference",
        "query_condition": "Q0 fixed; not the Q-averaged context allocation",
        "max_replay_error": float(frame.replay_error.max()),
        "max_account_error": float(
            np.max(
                np.abs(
                    frame.context_total
                    - frame.evidence_allocated
                    - frame.question_allocated
                )
            )
        ),
        "opposed_evidence_question": int(
            (frame.evidence_allocated * frame.question_allocated < 0).sum()
        ),
        "cost": read(out / "run_status.json"),
        "metrics": metrics,
        "effects_sha256": sha(out / "private_effects.csv"),
    }
    write(out / "summary.json", result)
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


def run(out):
    """Replay cached edited states, then evaluate only two new suffix endpoints."""
    manifest = read(out / "manifest.json")
    for name, digest in manifest["code_hashes"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Frozen implementation changed")
    if (
        sha(out / "private_design.json") != manifest["input_sha256"]
        or sha(MODEL / "model.safetensors") != manifest["model_sha256"]
    ):
        raise ValueError("Frozen material or checkpoint changed")
    prior = read(out / "run_status.json")
    if prior["status"] == "COMPLETE":
        summarize(out)
        return
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    observer.sites = {"block_16": observer.model.model.layers[16]}
    current, old = observer.manifest(), read(PARENT / "observer.json")
    for name in (
        "torch",
        "transformers",
        "dtype",
        "device",
        "attention",
        "template_sha256",
        "pad_to_length",
    ):
        if current[name] != old[name]:
            raise ValueError(f"Observer changed: {name}")
    write(out / "observer.json", current)
    device = next(observer.model.parameters()).device
    start = time.monotonic()
    complete = 0

    def status(state):
        p = {
            "status": state,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "completed_units": complete,
            "prefix_calls": 0,
            "gradient_calls": 0,
            "labels_read": False,
        }
        write(out / "run_status.json", p)
        print(p, flush=True)

    try:
        for d in read(out / "private_design.json"):
            path = out / "private_packets" / f"unit_{d['index']:02d}_{d['query']}.json"
            if path.exists():
                if read(path)["parent_packet_sha256"] != d["packet_sha256"]:
                    raise ValueError("Existing output changed source")
                complete += 1
                continue
            if (
                prior["calls"] + observer.calls + 3 > manifest["max_calls"]
                or prior["seconds"] + time.monotonic() - start > manifest["max_seconds"]
            ):
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            src = ROOT / d["parent_packet"]
            if (
                sha(src) != d["packet_sha256"]
                or sha(src.with_suffix(".npz")) != d["arrays_sha256"]
            ):
                raise ValueError("Source endpoint or arrays changed")
            original = read(src)
            with np.load(src.with_suffix(".npz"), allow_pickle=False) as z:
                full = torch.tensor(z["full_state"][None], device=device)
                edited = torch.tensor(z["edited_state"][None], device=device)
                aligned = align_states(full, edited, z["mapping"])
            kw = cached_suffix_kwargs(observer.model, edited)
            native = replay_suffix(observer, edited, kw, d["alphabet"])
            error = abs(
                native["conditional_score"] - original["edited"]["conditional_score"]
            )
            if error > 2e-6:
                raise ValueError("Cached suffix replay failed")
            scores = {
                arm: replay_suffix(
                    observer,
                    split_context_state(aligned, edited, d["question_mask"], arm),
                    kw,
                    d["alphabet"],
                )
                for arm in ("E", "P")
            }
            baseline = original["cells"]["f00"]["conditional_score"]
            endpoint = original["cells"]["f10"]["conditional_score"]
            parts = grid_components(
                baseline,
                baseline,
                scores["E"]["conditional_score"],
                scores["P"]["conditional_score"],
                endpoint,
            )
            write(
                path,
                {
                    "author_id": d["author_id"],
                    "index": d["index"],
                    "role": d["role"],
                    "query": d["query"],
                    "parent_packet_sha256": d["packet_sha256"],
                    "replay_error": error,
                    "new_cells": scores,
                    "origin": baseline,
                    "joint_context_endpoint": endpoint,
                    "components": parts,
                },
            )
            complete += 1
            del full, edited, aligned, kw
        status("COMPLETE")
    except BaseException as exc:
        status("FAILED_RETAINED")
        write(out / "error_log.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    summarize(out)


def main():
    """Run only the explicit phase using local cached model and source data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "context_carrier_split_v1"
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError("Keep all private data inside ignored results")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    (prepare if args.phase == "prepare" else run)(out)


if __name__ == "__main__":
    main()
