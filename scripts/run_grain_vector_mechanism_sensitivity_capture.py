#!/usr/bin/env python3
"""Capture local gradients on the frozen VM04 natural discovery materials."""

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

from grain_research.vector_mechanisms.context import SYSTEM
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from grain_research.vector_mechanisms.memory import render_records
from grain_research.vector_mechanisms.natural_patch import patch_message
from grain_research.vector_mechanisms.sensitivity import (
    local_margin_gradient,
    region_derivative_stats,
)
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_vector_mechanism_controls import write_json


def main():
    """Record states, derivatives and disjoint source regions without using labels."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--materials", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    parent, status = (
        read(a.parent_dir / "manifest.json"),
        read(a.parent_dir / "run_status.json"),
    )
    cases = read(a.materials / "cases.json")
    if (
        status["status"] != "COMPLETE"
        or material_hash(cases) != parent["material_sha256"]
    ):
        raise ValueError("Need unchanged completed natural pilot")
    by_case = {c["case_id"]: c for c in cases}
    raw_jobs, seen = [], set()
    for i, job in enumerate(parent["jobs"]):
        key = (job["case_id"], job["task"])
        if key not in seen:
            raw_jobs.append(job | {"parent_index": i})
            seen.add(key)
    if len(raw_jobs) != 24:
        raise ValueError("Expected 12 currents times two questions")
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a new output directory")
    archive_source(ROOT, out / "source_snapshot.zip")
    torch.manual_seed(731)
    torch.set_num_threads(4)
    obs = parent["observer"]
    observer = DecoderObserver.load(
        obs["model"],
        revision=obs["revision"],
        device="cuda",
        dtype="bfloat16",
        max_tokens=obs["max_tokens"],
        pad_to_length=obs["pad_to_length"],
        local_files_only=True,
    )
    jobs = []
    for job in raw_jobs:
        case = by_case[job["case_id"]]
        history = render_records(case["history"])
        user, current = patch_message(history, case["current"]["body"], job["task"])
        ids = observer.prompt_ids(SYSTEM, user)
        if material_hash(ids) != job["token_hashes"][0]:
            raise ValueError("Natural prompt identity changed")
        history_start = len(
            "The earlier and current excerpts are presented as writing by one anonymous writer.\n\n"
        )
        history_end = current[0] - len("\n\nCurrent excerpt:\n")
        if user[history_start:history_end] != history:
            raise ValueError("Historical region construction failed")
        question_start = user.index("Question: ", current[1])
        regions = {
            "history": observer.content_intervals(
                SYSTEM, user, [(history_start, history_end)]
            ),
            "current": observer.content_intervals(SYSTEM, user, [current]),
            "question": observer.content_intervals(
                SYSTEM, user, [(question_start, len(user))]
            ),
        }
        used = [i for region in regions.values() for i in region]
        if len(used) != len(set(used)):
            raise ValueError("Semantic regions overlap")
        regions["other"] = sorted(set(range(len(ids))) - set(used))
        jobs.append(job | {"ids": ids, "regions": regions})
    manifest = {
        "protocol": "GRAIN_VM04_LOCAL_SENSITIVITY_CAPTURE_V1",
        "parent_identity": parent["identity"],
        "code_sha256": code_fingerprint(ROOT),
        "observer": observer.manifest(),
        "cpu_saved_tensor_offload": True,
        "model_weight_training": False,
        "psychological_labels": False,
        "sites": ["block_10", "block_16"],
        "jobs": [{k: v for k, v in j.items() if k != "ids"} for j in jobs],
        "max_calls": 128,
        "max_seconds": 900,
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    cache = out / "private_cache"
    cache.mkdir()
    choices = [
        observer.tokenizer.encode(x, add_special_tokens=False) for x in ("Yes", "No")
    ]
    if any(len(c) != 1 for c in choices):
        raise ValueError("Need the frozen single-token readout")
    yes, no = choices[0][0], choices[1][0]
    rows, checks = [], []
    start = time.monotonic()
    stage = "RUNNING"
    write_json(out / "run_status.json", {"status": stage})
    try:
        for ji, job in enumerate(jobs):
            if (
                time.monotonic() - start > manifest["max_seconds"]
                or observer.calls + 4 > 128
            ):
                stage = "PARTIAL_BUDGET"
                break
            _, baseline = observer.capture(job["ids"])
            base = float(baseline[yes] - baseline[no])
            old = np.load(
                a.parent_dir / "private_cache" / f"j{job['parent_index']:02d}_a0.npz"
            )
            old_positions = (
                old["current_positions"].tolist() + old["decision_position"].tolist()
            )
            for site in manifest["sites"]:
                h, g, score = local_margin_gradient(
                    observer, job["ids"], site, yes, no, offload=True
                )
                state_error = float(np.max(np.abs(h[old_positions] - old[site])))
                if score != base or state_error != 0:
                    raise RuntimeError(
                        "Gradient observer differs from frozen no-gradient instrument"
                    )
                np.savez_compressed(
                    cache / f"j{ji:02d}_{site}.npz", state=h, gradient=g
                )
                for values in region_derivative_stats(h, g, job["regions"]):
                    rows.append(
                        {
                            "case_id": job["case_id"],
                            "author_id": job["author_id"],
                            "task": job["task"],
                            "site": site,
                            **values,
                        }
                    )
                checks.append(
                    {
                        "job": ji,
                        "site": site,
                        "margin": score,
                        "gradient_forward_error": abs(score - base),
                        "state_replay_error": state_error,
                    }
                )
            _, restored = observer.capture(job["ids"])
            if float(restored[yes] - restored[no]) != base:
                raise RuntimeError("Restoration failed")
            pd.DataFrame(rows).to_csv(out / "gradient_regions.csv", index=False)
            pd.DataFrame(checks).to_csv(out / "checks.csv", index=False)
            write_json(
                out / "run_status.json",
                {"status": "RUNNING", "jobs": ji + 1, "calls": observer.calls},
            )
            print(f"SENSITIVITY {ji + 1}/24 calls={observer.calls}", flush=True)
        else:
            stage = "COMPLETE"
    except Exception as exc:
        write_json(
            out / "run_status.json",
            {
                "status": "FAILED_CONTROL_OR_RUNTIME",
                "calls": observer.calls,
                "error": type(exc).__name__ + ": " + str(exc),
            },
        )
        raise
    write_json(
        out / "run_status.json",
        {
            "status": stage,
            "jobs": len(checks) // 2,
            "gradient_forwards": len(checks),
            "calls": observer.calls,
            "seconds": time.monotonic() - start,
            "input_tokens": observer.input_tokens,
            "processed_tokens": observer.processed_tokens,
            "forward_error_max": max(
                (r["gradient_forward_error"] for r in checks), default=0
            ),
            "state_error_max": max(
                (r["state_replay_error"] for r in checks), default=0
            ),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "psychological_labels_used": False,
        },
    )
    print(json.dumps(read(out / "run_status.json")), flush=True)


if __name__ == "__main__":
    main()
