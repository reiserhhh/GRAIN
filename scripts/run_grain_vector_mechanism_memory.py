#!/usr/bin/env python3
"""Frozen natural memory contrasts and synthetic update controls, not label training."""

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

from grain_research.vector_mechanisms.context import QUESTIONS, SYSTEM
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from grain_research.vector_mechanisms.memory import (
    memory_arms,
    natural_message,
    update_controls,
    update_message,
    validate_summary_review,
)
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_vector_mechanism_controls import write_json


def make_material(cases, audits, *, question_style="interrogative"):
    """Keep natural evidence judgments distinct from controlled record truth."""
    rows = []
    for case in cases:
        common = {k: case[k] for k in ("case_id", "author_id", "current_index")}
        for task in QUESTIONS:
            for arm, history in memory_arms(
                case, audits[case["author_id"]]["audit"], task
            ).items():
                rows.append(
                    {
                        **common,
                        "kind": "natural",
                        "task": task,
                        "arm": arm,
                        "user": natural_message(history, case["current"]["body"], task),
                        "current": case["current"]["body"],
                        "expected_yes": None,
                        "placement": "history_before_current",
                        "negative_question": False,
                    }
                )
        for update in update_controls():
            for placement in ("before", "after"):
                for negative in (False, True):
                    rows.append(
                        {
                            **common,
                            "kind": "synthetic_on_natural",
                            "task": "record_event",
                            "arm": update["variant"],
                            "user": update_message(
                                case["current"]["body"],
                                update["text"],
                                placement=placement,
                                negative_question=negative,
                                question_style=question_style,
                            ),
                            "current": case["current"]["body"],
                            "placement": placement,
                            "negative_question": negative,
                            "expected_yes": update["expected_yes"] ^ int(negative),
                        }
                    )
    return rows


def main():
    """Cache logits and pre-answer states; no online selection or generation."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--materials", type=Path, required=True)
    p.add_argument("--generated", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--controlled-only", action="store_true")
    p.add_argument(
        "--question-style",
        choices=("interrogative", "statement"),
        default="interrogative",
    )
    a = p.parse_args()
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    cases = read(a.materials / "cases.json")
    parent = read(a.materials / "manifest.json")
    if material_hash(cases) != parent["case_sha256"] or parent["labels_used"]:
        raise ValueError("Wrong or label-informed case material")
    generated, review = read(a.generated), read(a.review)
    if not review.get("completed_before_readout") or review[
        "generated_sha256"
    ] != material_hash(generated):
        raise ValueError("Review not frozen against this generation before readout")
    audits = validate_summary_review(
        generated, review["reviews"], read(a.materials / "summarization_sources.json")
    )
    if len(cases) != 12 or len(audits) != 6:
        raise ValueError("This bounded discovery protocol has 12 currents, six authors")
    material = make_material(cases, audits, question_style=a.question_style)
    if a.controlled_only:
        material = [r for r in material if r["kind"] == "synthetic_on_natural"]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if any(a.output_dir.iterdir()):
        raise ValueError("Use a new empty output directory")
    archive_source(ROOT, a.output_dir / "source_snapshot.zip")
    torch.manual_seed(731)
    torch.set_num_threads(4)
    observer = DecoderObserver.load(
        "Qwen/Qwen2.5-1.5B-Instruct",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        device="cuda",
        dtype="bfloat16",
        max_tokens=3072,
        pad_to_length=3072,
        local_files_only=True,
    )
    jobs, current_token_ids = [], {}
    for i, row in enumerate(material):
        ids = observer.prompt_ids(SYSTEM, row["user"])
        # Explicit location avoids matching a current body inside quoted history.
        marker = (
            "Current excerpt:\n"
            if row["kind"] == "natural"
            else "Natural background:\n"
        )
        start = row["user"].index(marker) + len(marker)
        indices = observer.content_intervals(
            SYSTEM, row["user"], [(start, start + len(row["current"]))]
        )
        selected_ids = [ids[j] for j in indices]
        if current_token_ids.setdefault(row["case_id"], selected_ids) != selected_ids:
            raise ValueError("Current token boundary changes across conditions")
        end = observer.content_position(SYSTEM, row["user"], "\nEvidence ends here.")
        jobs.append(
            {
                **row,
                "job_id": f"j{i:04d}",
                "ids": ids,
                "indices": indices,
                "positions": [end, len(ids) - 1],
            }
        )
    manifest = {
        "protocol": "GRAIN_VM03_MEMORY_AUDIT_V1",
        "question_style": a.question_style,
        "controlled_only": a.controlled_only,
        "parent_identity": parent["identity"],
        "review_sha256": material_hash(review),
        "generated_sha256": material_hash(generated),
        "material_sha256": material_hash(material),
        "observer": observer.manifest(),
        "jobs": [
            {
                k: v
                for k, v in r.items()
                if k not in ("ids", "indices", "positions", "user", "current")
            }
            | {"prompt_tokens": len(r["ids"]), "token_sha256": material_hash(r["ids"])}
            for r in jobs
        ],
        "code_sha256": code_fingerprint(ROOT),
        "authors": 6,
        "currents": 12,
        "natural_truth_labels": False,
        "psychological_labels": False,
        "known_truth_scope": "synthetic experimental event only",
        "fixed_analysis_sites": ["block_10", "block_16"],
        "padding": 3072,
        "current_token_ids_equal": True,
    }
    manifest["identity"] = material_hash(manifest)
    write_json(a.output_dir / "manifest.json", manifest)
    cache = a.output_dir / "private_cache"
    cache.mkdir()
    yes, no = [
        observer.tokenizer.encode(s, add_special_tokens=False) for s in ("Yes", "No")
    ]
    if len(yes) != 1 or len(no) != 1:
        raise ValueError("This protocol requires single-token Yes/No")
    start_time, records, features = time.monotonic(), [], {}
    replay_error, prefix_error, input_error = 0.0, 0.0, 0.0
    prefixes, inputs = {}, {}
    for i, c in enumerate(jobs):
        pools = [("mean", c["indices"]), ("std", c["indices"])]
        states, logits = observer.capture(
            c["ids"], positions=c["positions"], pools=pools
        )
        if not np.isfinite(logits).all() or not all(
            np.isfinite(v).all() for v in states.values()
        ):
            raise ValueError("Nonfinite model observation")
        if i == 0:
            rs, rl = observer.capture(c["ids"], positions=c["positions"], pools=pools)
            replay_error = max(
                float(np.max(np.abs(logits - rl))),
                max(float(np.max(np.abs(states[k] - rs[k]))) for k in states),
            )
            if replay_error != 0:
                raise ValueError("Exact fixed-shape replay failed")
        # Causal prefix invariance is tested only when the entire input prefix matches.
        key = material_hash(c["ids"][: c["positions"][0] + 1])
        prefix_state = np.concatenate([v[[0, 2, 3]].ravel() for v in states.values()])
        if key in prefixes:
            prefix_error = max(
                prefix_error, float(np.max(np.abs(prefix_state - prefixes[key])))
            )
        else:
            prefixes[key] = prefix_state
        base = inputs.setdefault(c["case_id"], states["input_embedding"][2:].copy())
        input_error = max(
            input_error, float(np.max(np.abs(states["input_embedding"][2:] - base)))
        )
        if prefix_error != 0 or input_error != 0:
            write_json(
                a.output_dir / "run_status.json",
                {
                    "status": "FAILED_CONTROL",
                    "failed_job": c["job_id"],
                    "completed": i,
                    "prefix_error": prefix_error,
                    "input_error": input_error,
                },
            )
            raise ValueError("Input embedding or causal prefix control failed")
        margin = float(logits[yes[0]] - logits[no[0]])
        m = float(np.max(logits))
        logz = m + float(np.log(np.exp(logits.astype(np.float64) - m).sum()))
        candidate_mass = float(
            np.exp(float(logits[yes[0]]) - logz) + np.exp(float(logits[no[0]]) - logz)
        )
        record = {
            k: v
            for k, v in c.items()
            if k not in ("user", "current", "ids", "indices", "positions")
        }
        record.update(
            margin=margin,
            conditional_yes=float(1 / (1 + np.exp(-margin))),
            yes_no_probability_mass=candidate_mass,
            predicted_yes=int(margin > 0),
            prompt_tokens=len(c["ids"]),
            current_tokens=len(c["indices"]),
        )
        if c["expected_yes"] is not None:
            record["correct"] = int(record["predicted_yes"] == c["expected_yes"])
        records.append(record)
        np.savez_compressed(cache / (c["job_id"] + ".npz"), **states)
        write_json(cache / (c["job_id"] + ".json"), record)
        for site in ("block_10", "block_16"):
            for name, row_index in (
                ("evidence_end", 0),
                ("decision", 1),
                ("current_mean", 2),
                ("current_std", 3),
            ):
                features.setdefault(site + "_" + name, []).append(
                    states[site][row_index]
                )
        if (i + 1) % 48 == 0:
            status = {
                "status": "RUNNING",
                "completed": i + 1,
                "planned": len(jobs),
                "seconds": time.monotonic() - start_time,
            }
            write_json(a.output_dir / "run_status.json", status)
            print(json.dumps(status), flush=True)
    pd.DataFrame(records).to_csv(a.output_dir / "readouts.csv", index=False)
    np.savez_compressed(
        a.output_dir / "representations.npz",
        **{k: np.stack(v) for k, v in features.items()},
    )
    status = {
        "status": "COMPLETE",
        "cases": len(jobs),
        "calls": observer.calls,
        "input_tokens": observer.input_tokens,
        "processed_tokens": observer.processed_tokens,
        "seconds": time.monotonic() - start_time,
        "replay_max_error": replay_error,
        "causal_prefix_max_error": prefix_error,
        "input_embedding_max_error": input_error,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "natural_truth_labels": False,
        "psychological_labels": False,
    }
    write_json(a.output_dir / "run_status.json", status)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
