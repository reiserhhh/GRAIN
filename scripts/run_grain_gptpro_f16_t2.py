#!/usr/bin/env python3
"""Construct and cold-replay T2 using existing source fits, never true states."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from grain_research.gptpro_foundation_v3.f16_state_instrument import (
    READER_HASH,
    cold_score,
    digest,
    finalize_reference,
    source_template,
    state_moments,
)


def save(path, value):
    """Atomically checkpoint strict JSON without replacing historical revisions."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def group(path, role):
    """Group declared records and preserve their given observation ordering."""
    groups = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["role"] != role:
            raise ValueError("wrong role")
        groups.setdefault(row["author_id"], []).append(row)
    return groups


def run_cell(arguments):
    """Fit source norms, then run a separate cold process on application prefixes."""
    source_root, boot_root, output_root, seed, smoke, resume = arguments
    started = time.monotonic()
    root = source_root / f"core_{seed}"
    output = output_root / str(seed)
    output.mkdir(exist_ok=resume)
    template = source_template(
        {"target_id": "T2", "world": "core", "target": "current_state"},
        root / "source.jsonl",
        boot_root / f"core_{seed}",
        seed,
        count=3 if smoke else 200,
        draws=8 if smoke else 200,
    )
    groups = group(root / "source.jsonl", "source")
    ids = sorted(groups)[:3] if smoke else sorted(groups)
    template["source_reference_authors"] = len(ids)
    protocol = {
        "template": template,
        "source_author_order": ids,
        "smoke": smoke,
        "runner_hash": digest(__file__),
        "brief_hash": digest(output_root / "briefs.json"),
    }
    protocol_path = output / "protocol_before_source_scoring.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("resume source/code/brief mismatch")
    else:
        save(protocol_path, protocol)
    checkpoint_root = output / "source_scores"
    checkpoint_root.mkdir(exist_ok=True)
    reference = []
    for i, author in enumerate(ids):
        path = checkpoint_root / f"{i:03d}.json"
        if path.exists():
            entry = json.loads(path.read_text())
            if entry["author_id"] != author:
                raise ValueError("reference author mismatch")
        else:
            entry = {
                "author_id": author,
                "result": state_moments(groups[author][:48], template),
            }
            save(path, entry)
        reference.append(entry["result"])
        if entry["result"]["status"] != "MOMENTS_COMPUTED":
            raise ValueError(f"unsupported source reference retained at {path}")
        if (i + 1) % 10 == 0 or i + 1 == len(ids):
            save(
                output / "checkpoint.json",
                {
                    "phase": "source_reference",
                    "completed": i + 1,
                    "total": len(ids),
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
            print(f"T2 seed={seed} source={i + 1}/{len(ids)}", flush=True)
    bundle = finalize_reference(template, reference)
    bundle_path = output / "bundle.json"
    save(bundle_path, bundle)
    # Application observations are first opened only after source norms freeze.
    app = group(root / "application.jsonl", "application")
    if set(app) & set(groups):
        raise ValueError("source/application author overlap")
    app_ids = sorted(app)[:2] if smoke else sorted(app)
    if not smoke and (len(ids), len(app_ids)) != (300, 200):
        raise ValueError("full source/application counts incomplete")
    inputs = [
        {
            "world_contract": bundle["world_contract"],
            "reader_hash": READER_HASH,
            "records": app[author][:48],
        }
        for author in app_ids
    ]
    warm = []
    warm_dir = output / "warm"
    warm_dir.mkdir(exist_ok=True)
    application_manifest = {
        "input_hash": digest(root / "application.jsonl"),
        "author_order": app_ids,
        "bundle_sha256": digest(bundle_path),
    }
    manifest_path = output / "application_manifest.json"
    if (
        manifest_path.exists()
        and json.loads(manifest_path.read_text()) != application_manifest
    ):
        raise ValueError("application resume mismatch")
    save(manifest_path, application_manifest)
    for i, raw in enumerate(inputs):
        path = warm_dir / f"{i:03d}.json"
        if path.exists():
            result = json.loads(path.read_text())
        else:
            result = cold_score(raw, bundle)
            save(path, result)
        warm.append(result)
        if result["status"] != "MOMENTS_COMPUTED":
            raise ValueError(f"unsupported application retained at {path}")
        if (i + 1) % 10 == 0 or i + 1 == len(inputs):
            save(
                output / "checkpoint.json",
                {
                    "phase": "warm",
                    "completed": i + 1,
                    "total": len(inputs),
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
    repo = Path(__file__).resolve().parents[1]
    cold_output = output / "cold_scores.jsonl"
    if not cold_output.exists():
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            (tmp / "bundle.json").write_bytes(bundle_path.read_bytes())
            (tmp / "input.jsonl").write_text(
                "".join(json.dumps(raw) + "\n" for raw in inputs)
            )
            save(
                output / "checkpoint.json",
                {"phase": "cold_subprocess", "authors": len(inputs)},
            )
            command = [
                sys.executable,
                "-m",
                "grain_research.gptpro_foundation_v3.f16_state_instrument",
                "--bundle",
                "bundle.json",
                "--sha256",
                digest(bundle_path),
                "--input",
                "input.jsonl",
                "--output",
                "scores.jsonl",
            ]
            env = dict(
                os.environ,
                PYTHONPATH=str(repo),
                OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS="1",
            )
            with (output / "cold_process.log").open("w") as log:
                subprocess.run(
                    command,
                    cwd=tmp,
                    env=env,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            cold_output.write_bytes((tmp / "scores.jsonl").read_bytes())
    cold = [json.loads(line) for line in cold_output.read_text().splitlines()]
    if len(cold) != len(warm):
        raise AssertionError("incomplete cold panel")
    max_error = max(
        abs(a[k] - b[k])
        for a, b in zip(warm, cold, strict=True)
        for k in ("score", "variance", "reference_score", "reference_variance")
    )
    if cold != warm or max_error > 1e-10:
        raise AssertionError("cold/warm mismatch")
    summary = {
        "target": "T2",
        "seed": seed,
        "status": "COLD_REPLAY_COMPLETE",
        "source_authors": len(ids),
        "application_authors": len(app_ids),
        "source_fits": len(bundle["source_models"]),
        "personal_draws": bundle["personal_draws"],
        "cold_warm_max_error": max_error,
        "elapsed_seconds": time.monotonic() - started,
        "calibration": "NOT_EVALUATED",
        "independent_builder": "not this same-agent cold replay",
        "source_parameter_refits": 0,
        "human_validity": "NOT_TESTED",
    }
    save(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    """Run three independent seed cells, at most three active CPU workers."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--bootstrap-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    brief = (
        Path(__file__).resolve().parents[1] / "configs/grain_f16_dynamic_briefs_v1.json"
    )
    frozen = args.output / "briefs.json"
    if frozen.exists() and frozen.read_bytes() != brief.read_bytes():
        raise ValueError("brief changed on resume")
    if not frozen.exists():
        frozen.write_bytes(brief.read_bytes())
    seeds = [20260915] if args.smoke else [20260915, 20260916, 20260917]
    tasks = [
        (
            args.source_root.resolve(),
            args.bootstrap_root.resolve(),
            args.output.resolve(),
            seed,
            args.smoke,
            args.resume,
        )
        for seed in seeds
    ]
    if args.smoke:
        summaries = [run_cell(tasks[0])]
    else:
        with ProcessPoolExecutor(max_workers=3) as pool:
            summaries = list(pool.map(run_cell, tasks))
    save(
        args.output / "summary.json",
        {
            "status": "T2_COLD_REPLAY_COMPLETE",
            "smoke": args.smoke,
            "cells": summaries,
            "brief_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
            "open": [
                "calibration evaluation",
                "independent numeric audit",
                "evidence cards",
                "T3 independent construction",
                "F16 parent acceptance",
            ],
        },
    )


if __name__ == "__main__":
    main()
