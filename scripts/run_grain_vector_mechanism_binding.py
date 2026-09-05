#!/usr/bin/env python3
"""Capture balanced cross-record binding, natural negation, and order controls."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.binding import (
    SYSTEM,
    binding_user_message,
    build_binding_controls,
)
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_vector_mechanism_controls import write_json


def main():
    """Run an untrained pinned observer, with fixed padding and full provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/grain_vector_contexts_v1.json"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config.update(max_tokens=512, pad_to_length=512, maximum_calls=350)
    cases = build_binding_controls()
    if args.limit is not None:
        cases = cases[: args.limit]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_status.json").exists():
        raise ValueError("Use a new run directory")
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    observer = DecoderObserver.load(
        name=config["model"],
        **{
            k: config[k]
            for k in (
                "revision",
                "device",
                "dtype",
                "max_tokens",
                "local_files_only",
                "pad_to_length",
            )
        },
    )
    manifest = {
        "config": config,
        "cases": cases,
        "system": SYSTEM,
        "code_sha256": code_fingerprint(ROOT),
        "scope": "Relational synthetic instrument, not new natural-author evidence",
    }
    manifest["identity"] = material_hash(manifest)
    write_json(output / "manifest.json", manifest)
    write_json(output / "model_manifest.json", observer.manifest())
    archive_source(ROOT, output / "source_snapshot.zip")
    ids_yes, ids_no = [
        observer.tokenizer.encode(s, add_special_tokens=False) for s in ("Yes", "No")
    ]
    if len(ids_yes) != 1 or len(ids_no) != 1:
        raise ValueError("Expected single-token natural Yes/No")
    cache = output / "cache"
    cache.mkdir(exist_ok=True)
    started = time.monotonic()
    rows, token_counts = [], {}
    for row in cases:
        user = binding_user_message(row)
        ids = observer.prompt_ids(SYSTEM, user)
        endpoint = observer.content_position(SYSTEM, user, row["passage"])
        content_ids = observer.tokenizer(row["passage"], add_special_tokens=False)[
            "input_ids"
        ]
        token_counts[row["case_id"]] = dict(Counter(content_ids))
        indices = observer.content_indices(SYSTEM, user, row["passage"])
        trace, logits = observer.capture(
            ids, positions=[endpoint, len(ids) - 1], pools=[("mean", indices)]
        )
        features = {
            f"{site}__{name}": values[i]
            for site, values in trace.items()
            for i, name in enumerate(("evidence_end", "decision", "content_mean"))
        }
        np.savez_compressed(cache / f"{row['case_id']}.npz", **features)
        margin = float(logits[ids_yes[0]] - logits[ids_no[0]])
        if not np.isfinite(margin):
            raise ValueError("Nonfinite logits")
        rows.append(
            {
                k: row[k]
                for k in (
                    "case_id",
                    "group_id",
                    "split",
                    "a",
                    "b",
                    "order",
                    "task",
                    "expected_yes",
                )
            }
            | {
                "margin": margin,
                "correct_default": int((margin >= 0) == row["expected_yes"]),
                "prompt_tokens": len(ids),
                "evidence_position": endpoint,
                "prompt_token_sha256": material_hash(ids),
            }
        )
        if (
            observer.calls > config["maximum_calls"]
            or time.monotonic() - started > config["maximum_seconds"]
        ):
            raise RuntimeError("Bounded run budget exceeded")
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "predictions.csv", index=False)
    write_json(output / "token_multisets.json", token_counts)
    summary = (
        frame.groupby(["split", "order", "task"])
        .correct_default.agg(["mean", "count"])
        .reset_index()
    )
    summary.to_csv(output / "accuracy.csv", index=False)
    # Replaying identical tokens checks determinism; suffix wording must not change the evidence state.
    first = cases[0]
    user = binding_user_message(first)
    ids = observer.prompt_ids(SYSTEM, user)
    ep = observer.content_position(SYSTEM, user, first["passage"])
    repeated, _ = observer.capture(ids, positions=[ep, len(ids) - 1])
    with np.load(cache / f"{first['case_id']}.npz") as saved:
        replay = max(
            float(abs(repeated[k][0] - saved[f"{k}__evidence_end"]).max())
            for k in repeated
        )
    invariance = []
    for group in frame.groupby(["group_id", "a", "b", "order"]):
        ids_in_group = group[1].case_id.tolist()
        if len(ids_in_group) != 3:
            continue
        with (
            np.load(cache / f"{ids_in_group[0]}.npz") as one,
            np.load(cache / f"{ids_in_group[1]}.npz") as two,
        ):
            invariance.append(
                max(
                    float(abs(one[k] - two[k]).max())
                    for k in one.files
                    if k.endswith("__evidence_end")
                )
            )
    passed = replay == 0 and max(invariance, default=0) == 0
    status = {
        "status": "COMPLETE" if passed else "FAILED_INVARIANCE",
        "cases": len(rows),
        "calls": observer.calls,
        "seconds": time.monotonic() - started,
        "replay_max_abs": replay,
        "pre_question_invariance_max_abs": max(invariance, default=0),
        "labels_scope": "synthetic_only",
        "original_model_training": False,
    }
    write_json(output / "run_status.json", status)
    if not passed:
        raise RuntimeError(
            "Binding instrument invariance failed; do not accept COMPLETE"
        )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
