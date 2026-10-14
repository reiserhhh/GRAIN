#!/usr/bin/env python3
"""Resumable source-author bootstrap of the original two-round F05 estimator."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from grain_research.gptpro_foundation_v3.f05 import grouped
from grain_research.gptpro_foundation_v3.f05_recovery import fit_source_dynamics


def dump(path, value):
    """Commit each finite checkpoint atomically without erasing previous fits."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False) + "\n")
    temporary.replace(path)


def resample_source(people, indices):
    """Give duplicate draws independent bootstrap-group identities, preserve time."""
    ids = sorted(people)
    rows = []
    for copy, index in enumerate(indices):
        original = people[ids[int(index)]]
        if any(r["role"] != "source" for r in original):
            raise ValueError("only source data permitted")
        if len({r["event_id"] for r in original}) != len(original):
            raise ValueError("duplicate event in original author")
        for j, r in enumerate(original):
            rows.append(
                r
                | {"author_id": f"copy_{copy:04d}", "event_id": f"copy_{copy:04d}:{j}"}
            )
    return rows


def main():
    """Fit 200 actual source resamples per world/seed, bounded by wall time."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=5400)
    args = parser.parse_args()
    if args.max_seconds <= 0:
        raise ValueError("positive time budget required")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    worlds = ["core"] if args.smoke else ["core", "near_root", "shared_shock"]
    seeds = [20260915] if args.smoke else [20260915, 20260916, 20260917]
    outer = 2 if args.smoke else 200
    paths = [args.source / f"{w}_{s}" / "source.jsonl" for w in worlds for s in seeds]
    code = [Path(__file__), Path(fit_source_dynamics.__code__.co_filename)]
    manifest = [
        {"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in paths + code
    ]
    protocol = {
        "worlds": worlds,
        "seeds": seeds,
        "outer": outer,
        "smoke": args.smoke,
        "source_prefix": 8 if args.smoke else 48,
        "source_count": 12 if args.smoke else 300,
        "source_files_and_code": manifest,
        "scope": "actual two-round ML/MAP source refit; no application or truth reads",
        "shared_shock_scope": "author bootstrap conditional on one shared episode; not independent episode uncertainty",
    }
    protocol_path = args.output / "protocol_before_run.json"
    if args.resume:
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("resume protocol/code/input hash changed")
    else:
        dump(protocol_path, protocol)
    lock = args.output / "running.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    started = time.monotonic()
    completed = 0
    failures = 0
    try:
        for wi, world in enumerate(worlds):
            for seed in seeds:
                source_path = args.source / f"{world}_{seed}" / "source.jsonl"
                people = grouped(
                    [json.loads(s) for s in source_path.read_text().splitlines()]
                )
                if args.smoke:
                    people = {k: people[k][:8] for k in sorted(people)[:12]}
                ids = sorted(people)
                directory = args.output / f"{world}_{seed}"
                directory.mkdir(exist_ok=True)
                for b in range(outer):
                    path = directory / f"fit_{b:03d}.json"
                    if path.exists():
                        saved = json.loads(path.read_text())
                        if (
                            saved["bootstrap"] != b
                            or saved["seed"] != seed
                            or saved["world"] != world
                        ):
                            raise ValueError("resume result identity mismatch")
                        completed += 1
                        failures += int(saved["status"] != "FITTED")
                        continue
                    if time.monotonic() - started >= args.max_seconds:
                        dump(
                            args.output / "summary.json",
                            {
                                "status": "TIME_BUDGET_CHECKPOINT",
                                "completed": completed,
                                "planned": len(worlds) * len(seeds) * outer,
                                "failed_tasks": failures,
                            },
                        )
                        return
                    indices = np.random.default_rng(
                        np.random.SeedSequence([seed, 13, 5, wi, b, 0])
                    ).integers(0, len(ids), len(ids))
                    before = time.monotonic()
                    record = {
                        "seed": seed,
                        "world": world,
                        "bootstrap": b,
                        "source_ids": ids,
                        "indices": indices.tolist(),
                    }
                    try:
                        fit = fit_source_dynamics(resample_source(people, indices))
                        record.update(
                            status="FITTED",
                            fit=fit,
                            successful_authors=[
                                r["successful_authors"] for r in fit["rounds"]
                            ],
                            noise_success=[r["noise_success"] for r in fit["rounds"]],
                        )
                    except (
                        ValueError,
                        np.linalg.LinAlgError,
                        FloatingPointError,
                    ) as error:
                        record.update(status="FIT_ERROR", error=repr(error))
                        failures += 1
                    record["elapsed_seconds"] = time.monotonic() - before
                    dump(path, record)
                    completed += 1
                    # Every fit is durable; the coarse checkpoint is also emitted every ten.
                    if (b + 1) % 10 == 0 or b + 1 == outer:
                        checkpoint = {
                            "status": "SOURCE_BOOTSTRAP_RUNNING",
                            "pid": os.getpid(),
                            "world": world,
                            "seed": seed,
                            "bootstrap_completed": b + 1,
                            "completed": completed,
                            "failed_tasks": failures,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                        dump(args.output / "checkpoint.json", checkpoint)
                        print(json.dumps(checkpoint), flush=True)
        dump(
            args.output / "summary.json",
            {
                "status": "SOURCE_BOOTSTRAP_FITS_COMPLETE_APPLICATION_MIXTURE_PENDING",
                "completed": completed,
                "planned": len(worlds) * len(seeds) * outer,
                "failed_tasks": failures,
            },
        )
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
