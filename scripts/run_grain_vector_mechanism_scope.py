#!/usr/bin/env python3
"""Capture scoped speech-act evidence with frozen prompts and pre-readout source coding."""

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

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from grain_research.vector_mechanisms.scope import DEFINITIONS, SYSTEM, scope_message
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_vector_mechanism_controls import write_json


def main():
    """Use only source-semantic labels, never psychological questionnaire values."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--materials", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    read = lambda name: json.loads((a.materials / name).read_text(encoding="utf-8"))
    parent, sources, ratings = (
        read("manifest.json"),
        read("calibration_sources.json"),
        read("source_ratings.json"),
    )
    if material_hash(sources) != parent["calibration_sha256"] or ratings[
        "source_sha256"
    ] != material_hash(sources):
        raise ValueError("Source coding identity mismatch")
    if not ratings["rated_before_model_readouts"]:
        raise ValueError("Need source coding frozen before logits")
    by_id = {r["case_id"]: r for r in ratings["ratings"]}
    if len(by_id) != 48 or set(by_id) != {s["case_id"] for s in sources}:
        raise ValueError("Coding must cover every calibration author")
    cases = read("synthetic_controls.json")
    for s in sources:
        for task in DEFINITIONS:
            truth = by_id[s["case_id"]][task]
            if truth is not None and (type(truth) is not int or truth not in (0, 1)):
                raise ValueError("Truth is binary or explicitly unresolved")
            for wording in ("question", "definition"):
                cases.append(
                    {
                        **s,
                        "source_case_id": s["case_id"],
                        "case_id": s["case_id"] + "_" + task + "_" + wording,
                        "group_id": s["author_id"],
                        "kind": "natural_scope",
                        "variant": "source_as_provided",
                        "task": task,
                        "wording": wording,
                        "expected_yes": truth,
                    }
                )
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if any(a.output_dir.iterdir()):
        raise ValueError("Use a new output directory")
    archive_source(ROOT, a.output_dir / "source_snapshot.zip")
    torch.manual_seed(731)
    torch.set_num_threads(4)
    observer = DecoderObserver.load(
        "Qwen/Qwen2.5-1.5B-Instruct",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        device="cuda",
        dtype="bfloat16",
        max_tokens=768,
        pad_to_length=768,
        local_files_only=True,
    )
    jobs = []
    for row in cases:
        user = scope_message(row["current"], row["task"], wording=row["wording"])
        ids = observer.prompt_ids(SYSTEM, user)
        start = user.index("Current excerpt:\n") + len("Current excerpt:\n")
        positions = observer.content_intervals(
            SYSTEM, user, [(start, start + len(row["current"]))]
        )
        end = observer.content_position(SYSTEM, user, "\nEvidence ends here.")
        jobs.append({**row, "ids": ids, "positions": positions, "end": end})
    manifest = {
        "protocol": "GRAIN_VM03_EVIDENCE_SCOPE_V1",
        "parent_identity": parent["identity"],
        "ratings_sha256": material_hash(ratings),
        "observer": observer.manifest(),
        "jobs": [
            {k: v for k, v in r.items() if k not in ("current", "ids", "positions")}
            | {"token_sha256": material_hash(r["ids"]), "prompt_tokens": len(r["ids"])}
            for r in jobs
        ],
        "code_sha256": code_fingerprint(ROOT),
        "external_labels_used": False,
        "readout": "Yes-minus-No last prefill logit; no threshold tuning during capture",
        "sites_fixed": ["block_10", "block_16"],
        "maximum_calls": 512,
    }
    manifest["identity"] = material_hash(manifest)
    write_json(a.output_dir / "manifest.json", manifest)
    cache = a.output_dir / "private_cache"
    cache.mkdir()
    yes, no = [
        observer.tokenizer.encode(s, add_special_tokens=False) for s in ("Yes", "No")
    ]
    if len(yes) != 1 or len(no) != 1:
        raise ValueError("Need one-token endpoints")
    start_time, rows, features = time.monotonic(), [], {}
    replay, prefix = 0.0, 0.0
    seen = {}
    for i, c in enumerate(jobs):
        pools = [("mean", c["positions"])]
        states, logits = observer.capture(
            c["ids"], positions=[c["end"], len(c["ids"]) - 1], pools=pools
        )
        if not np.isfinite(logits).all() or not all(
            np.isfinite(v).all() for v in states.values()
        ):
            raise ValueError("Nonfinite capture")
        if i == 0:
            again, other = observer.capture(
                c["ids"], positions=[c["end"], len(c["ids"]) - 1], pools=pools
            )
            replay = max(
                float(abs(logits - other).max()),
                max(float(abs(states[k] - again[k]).max()) for k in states),
            )
        key = material_hash(c["ids"][: c["end"] + 1])
        vector = np.concatenate([v[[0, 2]].ravel() for v in states.values()])
        if key in seen:
            prefix = max(prefix, float(abs(vector - seen[key]).max()))
        else:
            seen[key] = vector
        if replay != 0 or prefix != 0:
            write_json(
                a.output_dir / "run_status.json",
                {
                    "status": "FAILED_CONTROL",
                    "completed": i,
                    "replay": replay,
                    "prefix": prefix,
                },
            )
            raise ValueError("Causal prefix or replay control failed")
        margin = float(logits[yes[0]] - logits[no[0]])
        row = {
            k: v
            for k, v in c.items()
            if k not in ("current", "ids", "positions", "leakage_flags")
        }
        row.update(
            margin=margin,
            predicted_yes=int(margin > 0),
            prompt_tokens=len(c["ids"]),
            correct=None
            if c["expected_yes"] is None
            else int((margin > 0) == c["expected_yes"]),
        )
        rows.append(row)
        np.savez_compressed(cache / (c["case_id"] + ".npz"), **states)
        for site in ("block_10", "block_16"):
            for j, name in enumerate(("evidence_end", "decision", "current_mean")):
                features.setdefault(site + "_" + name, []).append(states[site][j])
        if (i + 1) % 48 == 0:
            status = {
                "status": "RUNNING",
                "completed": i + 1,
                "planned": len(jobs),
                "seconds": time.monotonic() - start_time,
            }
            write_json(a.output_dir / "run_status.json", status)
            print(json.dumps(status), flush=True)
        if observer.calls > 512 or time.monotonic() - start_time > 1800:
            raise RuntimeError("Bounded scope run exceeded budget")
    pd.DataFrame(rows).to_csv(a.output_dir / "readouts.csv", index=False)
    np.savez_compressed(
        a.output_dir / "representations.npz",
        **{k: np.stack(v) for k, v in features.items()},
    )
    status = {
        "status": "COMPLETE",
        "cases": len(jobs),
        "calls": observer.calls,
        "seconds": time.monotonic() - start_time,
        "input_tokens": observer.input_tokens,
        "processed_tokens": observer.processed_tokens,
        "replay_max_error": replay,
        "prefix_max_error": prefix,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "external_labels_used": False,
    }
    write_json(a.output_dir / "run_status.json", status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
