#!/usr/bin/env python3
"""Capture complete 12-comment evidence in Qwen without external label access."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.cohort import validate_multicomment_bundle
from grain_research.vector_mechanisms.context import QUESTIONS, SYSTEM
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_vector_mechanism_controls import write_json


def render_evidence(comments, task):
    """Quote complete raw records and make no hidden username/source features."""
    evidence, intervals = "", []
    intro = "The excerpts below were written by one anonymous writer.\n\n"
    for i, c in enumerate(comments):
        evidence += ("\n\n" if i else "") + f"Excerpt {i + 1}:\n"
        begin = len(intro) + len(evidence)
        evidence += c["body"]
        intervals.append((begin, begin + len(c["body"])))
    user = (
        intro
        + evidence
        + "\nEvidence ends here.\n\nQuestion: "
        + QUESTIONS[task]
        + "\nAnswer:"
    )
    return user, evidence, intervals


def main():
    """Cache every forward and verify repeatability and pre-question invariance."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--pad-to-length", type=int, default=3072)
    a = p.parse_args()
    bundles = json.loads(a.views.read_text(encoding="utf-8"))
    if not bundles or any(b["split"] != "discovery" for b in bundles):
        raise ValueError("This experiment only opens discovery authors")
    for b in bundles:
        validate_multicomment_bundle(b, units=12)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if any(a.output_dir.iterdir()):
        raise ValueError("Use a new empty trace directory")
    archive_source(ROOT, a.output_dir / "source_snapshot.zip")
    torch.manual_seed(731)
    torch.set_num_threads(4)
    observer = DecoderObserver.load(
        "Qwen/Qwen2.5-1.5B-Instruct",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        device="cuda",
        dtype="bfloat16",
        max_tokens=a.pad_to_length,
        pad_to_length=a.pad_to_length,
        local_files_only=True,
    )
    cases = []
    for b in bundles:
        for vi, comments in enumerate(b["views"]):
            for task in QUESTIONS:
                user, evidence, intervals = render_evidence(comments, task)
                ids = observer.prompt_ids(SYSTEM, user)
                # Pool only original comment bodies, not artificial excerpt numbering.
                indices = observer.content_intervals(SYSTEM, user, intervals)
                if len(set(indices)) != len(indices):
                    raise ValueError("Repeated content span in prompt")
                cases.append(
                    {
                        "author_id": b["author_id"],
                        "view": vi,
                        "task": task,
                        "ids": ids,
                        "positions": [
                            observer.content_position(SYSTEM, user, evidence),
                            len(ids) - 1,
                        ],
                        "indices": indices,
                    }
                )
    manifest = {
        "protocol": "GRAIN_VM02_MULTICOMMENT_QWEN_V1",
        "source_view_sha256": material_hash(bundles),
        "authors": [b["author_id"] for b in bundles],
        "reference_mask": [b["discovery_role"] == "reference" for b in bundles],
        "observer": observer.manifest(),
        "cases": [
            {k: v for k, v in c.items() if k not in ("ids", "positions", "indices")}
            | {"token_sha256": material_hash(c["ids"]), "prompt_tokens": len(c["ids"])}
            for c in cases
        ],
        "code_sha256": code_fingerprint(ROOT),
        "labels_used": False,
        "max_prompt_tokens": max(len(c["ids"]) for c in cases),
        "fixed_padding": a.pad_to_length,
        "stage": "qwen_joint_full",
        "all_comments_complete": True,
    }
    manifest["identity"] = material_hash(manifest)
    write_json(a.output_dir / "manifest.json", manifest)
    cache = a.output_dir / "private_cache"
    cache.mkdir()
    write_json(
        a.output_dir / "run_status.json",
        {"status": "RUNNING", "planned_cases": len(cases)},
    )
    yes, no = [
        observer.tokenizer.encode(s, add_special_tokens=False) for s in ("Yes", "No")
    ]
    if len(yes) != 1 or len(no) != 1:
        raise ValueError("Need single token readouts")
    start, records, features = time.monotonic(), [], {}
    prequestion_error, replay_error = 0.0, 0.0
    author_index = {b["author_id"]: i for i, b in enumerate(bundles)}
    previous = None
    for i, c in enumerate(cases):
        pools = [("mean", c["indices"]), ("std", c["indices"])]
        states, logits = observer.capture(
            c["ids"], positions=c["positions"], pools=pools
        )
        if not np.isfinite(logits).all() or not all(
            np.isfinite(v).all() for v in states.values()
        ):
            raise ValueError("Nonfinite trace")
        if i == 0:
            repeated, repeated_logits = observer.capture(
                c["ids"], positions=c["positions"], pools=pools
            )
            replay_error = max(
                float(np.max(np.abs(logits - repeated_logits))),
                max(float(np.max(np.abs(states[k] - repeated[k]))) for k in states),
            )
            if replay_error != 0:
                raise ValueError("Fixed-shape replay failed")
        if c["task"] == "disposition":
            prequestion_error = max(
                prequestion_error,
                max(
                    float(np.max(np.abs(states[k][[0, 2, 3]] - previous[k][[0, 2, 3]])))
                    for k in states
                ),
            )
            if prequestion_error != 0:
                raise ValueError("Causal pre-question invariance failed")
        else:
            previous = states
        for site, v in states.items():
            variants = {f"decision_{c['task']}": v[1]}
            if c["task"] == "behavior":
                variants.update(evidence_end=v[0], history_mean=v[2], history_std=v[3])
            for kind, row in variants.items():
                key = f"qwen_{site}_{kind}__chronological"
                if key not in features:
                    features[key] = np.empty(
                        (len(bundles), 3, len(row)), dtype=np.float32
                    )
                features[key][author_index[c["author_id"]], c["view"]] = row
        name = f"{c['author_id']}_{c['view']}_{c['task']}"
        np.savez_compressed(cache / f"{name}.npz", **states)
        record = {
            "author_id": c["author_id"],
            "view": c["view"],
            "task": c["task"],
            "margin": float(logits[yes[0]] - logits[no[0]]),
            "prompt_tokens": len(c["ids"]),
            "body_tokens": len(c["indices"]),
        }
        records.append(record)
        write_json(cache / f"{name}.json", record)
        if (i + 1) % 24 == 0:
            progress = {
                "status": "RUNNING",
                "completed": i + 1,
                "planned": len(cases),
                "seconds": time.monotonic() - start,
            }
            write_json(a.output_dir / "run_status.json", progress)
            print(json.dumps(progress), flush=True)
    np.savez_compressed(a.output_dir / "representations.npz", **features)
    pd.DataFrame(records).to_csv(a.output_dir / "readouts.csv", index=False)
    status = {
        "status": "COMPLETE",
        "cases": len(cases),
        "calls": observer.calls,
        "input_tokens": observer.input_tokens,
        "processed_tokens": observer.processed_tokens,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "seconds": time.monotonic() - start,
        "prequestion_max_error": prequestion_error,
        "replay_max_error": replay_error,
        "labels_used": False,
    }
    write_json(a.output_dir / "run_status.json", status)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
