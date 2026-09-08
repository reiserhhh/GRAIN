#!/usr/bin/env python3
"""M25: recover fixed query P/Q states for the existing matched S1 author panel."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.prefix import capture_prefix
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_conditional_edits import LEVEL, compatible_observers
from scripts.analyze_grain_multiquery_sensitivity import load_panel
from scripts.run_grain_level_response import question_mask
from scripts.run_grain_multiquery_sensitivity import QUERIES
from scripts.run_grain_real_author_path import MODEL

SOURCE = BASE / "multiquery_sensitivity_s1_r1"
PLAN = ROOT / "docs/GRAIN_SHARED_AUTHOR_PROFILE_M25.md"


def prepare(out):
    """Bind existing authors, conditions, templates and provenance before inference."""
    _, records = load_panel(SOURCE)
    lm = read(LEVEL / "manifest.json")
    if (
        sha(LEVEL / "private_design.json") != lm["input_sha256"]
        or sha(LEVEL / "reference.npz") != lm["reference_sha256"]
    ):
        raise ValueError("M8 source changed")
    old = {(d["author_id"], d["query"]): d for d in read(LEVEL / "private_design.json")}
    evaluation = {a for a, _ in old}
    authors = {r[0]["author_id"] for r in records}
    if len(authors) != 24 or len(evaluation) != 12 or not evaluation <= authors:
        raise ValueError("Expected disjoint 12-reference/12-evaluation panel")
    source_files = {}
    for path in (SOURCE / "private_packets").glob("panel_*.json"):
        packet = read(path)
        source_files[(packet["author_id"], packet["view"])] = path
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    hashes = {
        str(p.relative_to(ROOT)): sha(p)
        for p in (
            SOURCE / "manifest.json",
            SOURCE / "private_material.json",
            LEVEL / "manifest.json",
            LEVEL / "private_design.json",
            LEVEL / "observer.json",
            LEVEL / "reference.npz",
        )
    }
    design, overlap = [], []
    view_texts = {}
    for row, packet, _ in records:
        uid, view = row["author_id"], row["view"]
        view_texts[(uid, view)] = set(row["texts"])
        source = source_files[(uid, view)]
        for p in (source, source.with_suffix(".npz")):
            hashes[str(p.relative_to(ROOT))] = sha(p)
        for query in QUERIES:
            for condition in ("full", "e1", "e2"):
                excluded = [] if condition == "full" else row[condition]
                texts = [t for j, t in enumerate(row["texts"]) if j not in excluded]
                _, _, encoded, mask = question_mask(tokenizer, texts, query)
                ids = np.asarray(encoded["input_ids"])
                if ids[mask].tolist() != lm["question_tokens"][query]:
                    raise ValueError("Fixed query token identities differ")
                reused = None
                if uid in evaluation and view == 0 and condition in ("full", "e1"):
                    d = old[(uid, query)]
                    stem = "full" if condition == "full" else "edited"
                    if (
                        ids.tolist() != d[stem + "_ids"]
                        or mask.tolist() != d[stem + "_mask"]
                    ):
                        raise ValueError("M8 reused material mismatch")
                    p = LEVEL / "private_packets" / f"unit_{d['index']:02d}_{query}.npz"
                    if sha(p) != read(p.with_suffix(".json"))["arrays_sha256"]:
                        raise ValueError("M8 reused arrays changed")
                    hashes[str(p.relative_to(ROOT))] = sha(p)
                    reused = {"path": str(p.relative_to(ROOT)), "stem": stem}
                design.append(
                    {
                        "index": len(design),
                        "author_id": uid,
                        "view": view,
                        "query": query,
                        "condition": condition,
                        "role": "evaluation" if uid in evaluation else "reference",
                        "input_ids": ids.tolist(),
                        "question_mask": mask.tolist(),
                        "source_packet": str(source.relative_to(ROOT)),
                        "reuse": reused,
                        "native_score": packet["queries"][query]["readouts"][condition][
                            "conditional_score"
                        ],
                    }
                )
    for uid in sorted(authors):
        a, b = view_texts[(uid, 0)], view_texts[(uid, 1)]
        overlap.append(
            {
                "author_id": uid,
                "common_exact_texts": len(a & b),
                "view_counts": [len(a), len(b)],
            }
        )
    if len(design) != 432 or sum(d["reuse"] is not None for d in design) != 72:
        raise ValueError("Unexpected design size")
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    write(out / "private_design.json", design)
    write(out / "private_overlap.json", overlap)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_SHARED_PROFILE_M25_V1",
            "units": 432,
            "new_prefix_calls": 360,
            "max_seconds": 900,
            "human_labels_read": False,
            "plan_sha256": sha(PLAN),
            "script_sha256": sha(__file__),
            "prefix_sha256": sha(ROOT / "grain_research/vector_mechanisms/prefix.py"),
            "input_sha256": sha(out / "private_design.json"),
            "overlap_sha256": sha(out / "private_overlap.json"),
            "model_sha256": lm["model_sha256"],
            "question_tokens": lm["question_tokens"],
            "reference_sha256": lm["reference_sha256"],
            "private_input_hashes": hashes,
            "reference_authors": 12,
            "evaluation_authors": 12,
        },
    )
    write(
        out / "run_status.json",
        {"status": "PREPARED", "calls": 0, "seconds": 0, "completed_units": 0},
    )
    print(
        {
            "units": len(design),
            "new_prefix_calls": 360,
            "exact_overlap_authors": sum(x["common_exact_texts"] > 0 for x in overlap),
        },
        flush=True,
    )


def collect(out):
    """Reuse old states, then bounded prefix-only collection with endpoint replay."""
    m, prior = read(out / "manifest.json"), read(out / "run_status.json")
    if prior["status"] == "COMPLETE":
        return
    checks = {
        **m["private_input_hashes"],
        str(PLAN.relative_to(ROOT)): m["plan_sha256"],
        str(Path(__file__).resolve().relative_to(ROOT)): m["script_sha256"],
        "grain_research/vector_mechanisms/prefix.py": m["prefix_sha256"],
    }
    for p, digest in checks.items():
        if sha(ROOT / p) != digest:
            raise ValueError("Frozen implementation/material changed: " + p)
    if (
        sha(out / "private_design.json") != m["input_sha256"]
        or sha(MODEL / "model.safetensors") != m["model_sha256"]
    ):
        raise ValueError("Design or fixed observer weights changed")
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    if not compatible_observers(observer.manifest(), read(LEVEL / "observer.json")):
        raise ValueError("Observer differs from old native experiment")
    write(out / "observer.json", observer.manifest())
    start, completed, max_error = time.monotonic(), 0, 0.0

    def status(state):
        """Persist actual cost and resumable completed units, never estimated calls."""
        value = {
            "status": state,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "completed_units": completed,
            "max_endpoint_replay_error": max_error,
            "gradients": 0,
            "suffix_calls": 0,
            "human_labels_read": False,
        }
        write(out / "run_status.json", value)
        print(value, flush=True)

    try:
        for d in read(out / "private_design.json"):
            path = out / "private_packets" / f"cell_{d['index']:03d}.npz"
            meta = path.with_suffix(".json")
            if meta.exists():
                old = read(meta)
                if sha(path) != old["array_sha256"]:
                    raise ValueError("Retained packet corrupted")
                max_error = max(max_error, old["endpoint_replay_error"])
                completed += 1
                continue
            reuse = d["reuse"]
            if reuse:
                with np.load(ROOT / reuse["path"], allow_pickle=False) as z:
                    p, q = z[reuse["stem"] + "_P"], z[reuse["stem"] + "_Q"]
                trace = {"reused": True}
            else:
                if (
                    prior["calls"] + observer.calls >= m["new_prefix_calls"]
                    or prior["seconds"] + time.monotonic() - start >= m["max_seconds"]
                ):
                    status("PAUSED_RESOURCE_BOUNDARY")
                    return
                values, trace = capture_prefix(observer, d["input_ids"])
                state = values["boundary"]
                p, q = state[np.asarray(d["question_mask"], bool)], state[-1]
            source = (ROOT / d["source_packet"]).with_suffix(".npz")
            with np.load(source, allow_pickle=False) as z:
                expected = z[f"{d['query']}__{d['condition']}"]
            error = float(np.max(np.abs(q - expected)))
            if error > 2e-5:
                raise ValueError(f"S1 endpoint replay mismatch: {error}")
            max_error = max(max_error, error)
            np.savez_compressed(path, P=p, Q=q)
            write(
                meta,
                {
                    **{
                        k: d[k]
                        for k in (
                            "index",
                            "author_id",
                            "view",
                            "query",
                            "condition",
                            "role",
                            "native_score",
                        )
                    },
                    "array_sha256": sha(path),
                    "endpoint_replay_error": error,
                    "trace": trace,
                },
            )
            completed += 1
            if completed % 18 == 0:
                status("RUNNING")
        status("COMPLETE")
    except Exception:
        status("ERROR_RETAINED")
        raise


def main():
    """Prepare or execute only the explicitly named private packet."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "collect"))
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Use ignored private result directory")
    (prepare if args.action == "prepare" else collect)(out)


if __name__ == "__main__":
    main()
