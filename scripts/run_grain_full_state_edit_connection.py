#!/usr/bin/env python3
"""Bounded four-author context/query factorial experiment on frozen S1 natural edits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.direct_traits import SYSTEM
from grain_research.vector_mechanisms.full_state_edit import (
    align_excerpts,
    align_states,
    capture_boundary,
    grid_components,
    mix_states,
    replay_suffix,
)
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_multiquery_sensitivity import load_panel
from scripts.run_grain_multiquery_sensitivity import (
    QUERIES,
    native_alphabet,
    query_prompt,
)
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write

SOURCE = BASE / "multiquery_sensitivity_s1_r1"
BRIDGE = BASE / "functional_construction_bridge_v1"


def token_view(tokenizer, bodies, query):
    """Bind exact template offsets and body spans; retain no data in public output."""
    user, spans = query_prompt(bodies, query)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if text.count(user) != 1:
        raise ValueError("Ambiguous user body in template")
    offset = text.index(user)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if list(ids) != list(encoded["input_ids"]) or len(ids) + 16 > 3072:
        raise ValueError("Template token mismatch or over budget")
    return text, [(a + offset, b + offset) for a, b in spans], encoded


def prepare(out):
    """Freeze four hash-selected authors and all token alignments before neural work."""
    source_manifest, records = load_panel(SOURCE)
    split_path = BRIDGE / "split_manifest.json"
    bridge_manifest = read(BRIDGE / "manifest.json")
    if sha(split_path) != bridge_manifest["split_sha256"]:
        raise ValueError("Prior reference/evaluation split changed")
    split = read(split_path)
    groups = [[], []]
    for uid, ref in zip(split["authors"], split["reference_mask"], strict=True):
        groups[0 if ref else 1].append(uid)
    key = lambda uid: hashlib.sha256(f"{split['salt']}|{uid}".encode()).hexdigest()
    selected = [
        u
        for pair in zip(*(sorted(g, key=key)[:2] for g in groups), strict=True)
        for u in pair
    ]
    lookup = {r[0]["author_id"]: r for r in records if r[0]["view"] == 0}
    if len(lookup) != 24 or len(selected) != 4:
        raise ValueError("Complete original author panel required")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    packets, alignment = [], []
    for index, uid in enumerate(selected):
        row, cached, _ = lookup[uid]
        kept = [j for j in range(len(row["texts"])) if j not in row["e1"]]
        entry = {
            "author_id": uid,
            "panel_index": row["panel_index"],
            "index": index,
            "role": "reference" if uid in groups[0] else "evaluation",
            "queries": {},
        }
        for query in QUERIES:
            full, fs, fe = token_view(tokenizer, row["texts"], query)
            edit, es, ee = token_view(tokenizer, [row["texts"][j] for j in kept], query)
            mapping, audit = align_excerpts(full, edit, fs, es, kept, fe, ee)
            alphabet = native_alphabet(query, tokenizer)
            if alphabet != cached["queries"][query]["alphabet"]:
                raise ValueError("Source scoring alphabet changed")
            entry["queries"][query] = {
                "full_ids": list(fe["input_ids"]),
                "edited_ids": list(ee["input_ids"]),
                "mapping": mapping.tolist(),
                "alignment": audit,
                "alphabet": alphabet,
                "source_full_score": cached["queries"][query]["readouts"]["full"][
                    "conditional_score"
                ],
                "source_edit_score": cached["queries"][query]["readouts"]["e1"][
                    "conditional_score"
                ],
            }
            alignment.append(audit)
        packets.append(entry)
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    write(out / "private_inputs.json", packets)
    code = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/full_state_edit.py",
        ROOT / "grain_research/vector_mechanisms/direct_traits.py",
        ROOT / "docs/GRAIN_FULL_STATE_EDIT_CONNECTION_PLAN_V1.md",
    ]
    if sha(MODEL / "model.safetensors") != source_manifest["model_sha256"]:
        raise ValueError("Source checkpoint changed")
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_FULL_STATE_EDIT_CONNECTION_V1",
            "scope": "four-author finite mechanism pilot",
            "authors": 4,
            "queries": list(QUERIES),
            "view": 0,
            "edit": "E1",
            "max_calls": 84,
            "max_seconds": 1200,
            "full_calls_per_unit": 2,
            "suffix_calls_per_unit": 5,
            "gradients": 0,
            "labels_read": False,
            "input_sha256": sha(out / "private_inputs.json"),
            "source_manifest_sha256": sha(SOURCE / "manifest.json"),
            "source_material_sha256": source_manifest["material_sha256"],
            "source_split_sha256": sha(split_path),
            "model_sha256": source_manifest["model_sha256"],
            "device": "mps",
            "dtype": "float32",
            "block": 16,
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in code},
            "replay_tolerance": 2e-6,
            "alignment_policy": alignment[0]["policy"],
            "rebasing_includes": "deletion, compact layout, scaffold history reassignment and retokenized boundaries",
        },
    )
    aggregate = {
        "status": "PREPARED",
        "authors": 4,
        "queries": 3,
        "units": 12,
        "accounted_tokens": sum(a["tokens"] for a in alignment),
        "transported_tokens": sum(a["exact_token_matches"] for a in alignment),
        "fresh_boundary_tokens": sum(
            sum(a["fresh_tokens"].values()) for a in alignment
        ),
        "renumbered_header_instances": sum(a["renumbered_headers"] for a in alignment),
        "nonmonotone_maps": sum(not a["monotone"] for a in alignment),
        "duplicate_donors": sum(a["duplicate_donors"] for a in alignment),
        "new_neural_calls": 0,
    }
    write(out / "alignment_audit.json", aggregate)
    write(
        out / "run_status.json",
        {"status": "PREPARED", "calls": 0, "seconds": 0.0, "completed_units": 0},
    )
    print(json.dumps(aggregate), flush=True)


def run(out):
    """Collect complete query units with same-run identity replays and strict budgets."""
    manifest = read(out / "manifest.json")
    for name, expected in manifest["code_hashes"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen code changed: {name}")
    if (
        sha(out / "private_inputs.json") != manifest["input_sha256"]
        or sha(MODEL / "model.safetensors") != manifest["model_sha256"]
    ):
        raise ValueError("Frozen data/model changed")
    prior = read(out / "run_status.json")
    if prior["status"] == "COMPLETE":
        print("Already complete; no new calls")
        return
    material = read(out / "private_inputs.json")
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    observer.sites = {"block_16": observer.model.model.layers[16]}
    current, previous = observer.manifest(), read(SOURCE / "observer.json")
    for name in (
        "torch",
        "transformers",
        "dtype",
        "device",
        "attention",
        "template_sha256",
        "pad_to_length",
    ):
        if current[name] != previous[name]:
            raise ValueError(f"Observer changed: {name}")
    write(out / "observer.json", current)
    start = time.monotonic()
    completed = 0

    def status(state):
        payload = {
            "status": state,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "completed_units": completed,
            "planned_units": 12,
            "labels_read": False,
            "gradients": 0,
        }
        write(out / "run_status.json", payload)
        print(json.dumps(payload), flush=True)

    try:
        for author in material:
            for query in QUERIES:
                path = (
                    out / "private_packets" / f"unit_{author['index']:02d}_{query}.json"
                )
                if path.exists():
                    old = read(path)
                    if (
                        old["author_id"] != author["author_id"]
                        or sha(path.with_suffix(".npz")) != old["arrays_sha256"]
                    ):
                        raise ValueError("Saved unit changed")
                    completed += 1
                    continue
                if (
                    prior["calls"] + observer.calls + 7 > manifest["max_calls"]
                    or prior["seconds"] + time.monotonic() - start
                    > manifest["max_seconds"]
                ):
                    status("PAUSED_RESOURCE_BOUNDARY")
                    return
                p = author["queries"][query]
                h, kw, full = capture_boundary(observer, p["full_ids"], p["alphabet"])
                he, kwe, edited = capture_boundary(
                    observer, p["edited_ids"], p["alphabet"]
                )
                original_replay = replay_suffix(observer, h, kw, p["alphabet"])
                f11 = replay_suffix(observer, he, kwe, p["alphabet"])
                errors = {
                    "same_run_full": abs(
                        original_replay["conditional_score"] - full["conditional_score"]
                    ),
                    "same_run_edit": abs(
                        f11["conditional_score"] - edited["conditional_score"]
                    ),
                    "source_full": abs(
                        full["conditional_score"] - p["source_full_score"]
                    ),
                    "source_edit": abs(
                        edited["conditional_score"] - p["source_edit_score"]
                    ),
                }
                if max(errors.values()) > manifest["replay_tolerance"]:
                    write(
                        out / "replay_failure.json", {"query": query, "errors": errors}
                    )
                    raise ValueError("Scoring replay exceeded tolerance")
                aligned = align_states(h, he, p["mapping"])
                scores = {"f11": f11}
                for context, query_state in ((0, 0), (1, 0), (0, 1)):
                    state = mix_states(aligned, he, context, query_state)
                    scores[f"f{context}{query_state}"] = replay_suffix(
                        observer, state, kwe, p["alphabet"]
                    )
                vals = {k: v["conditional_score"] for k, v in scores.items()}
                parts = grid_components(full["conditional_score"], **vals)
                np.savez_compressed(
                    path.with_suffix(".npz"),
                    full_state=h[0].cpu().numpy(),
                    edited_state=he[0].cpu().numpy(),
                    mapping=np.asarray(p["mapping"]),
                )
                write(
                    path,
                    {
                        "author_id": author["author_id"],
                        "role": author["role"],
                        "index": author["index"],
                        "query": query,
                        "full": full,
                        "edited": edited,
                        "cells": scores,
                        "components": parts,
                        "replay_errors": errors,
                        "alignment": p["alignment"],
                        "calls": 7,
                        "arrays_sha256": sha(path.with_suffix(".npz")),
                    },
                )
                completed += 1
                del h, he, aligned, state, kw, kwe
            status("RUNNING")
        status("COMPLETE")
    except BaseException as exc:
        status("FAILED_RETAINED")
        write(out / "error_log.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def main():
    """Keep acquisition explicitly phased, offline, and confined to private results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "full_state_edit_connection_v1"
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError("Keep private materials under ignored results")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    (prepare if args.phase == "prepare" else run)(out)


if __name__ == "__main__":
    main()
