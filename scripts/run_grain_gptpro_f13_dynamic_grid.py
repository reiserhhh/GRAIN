#!/usr/bin/env python3
"""Consume durable source fits in protocol order without restarting the producer."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.run_grain_gptpro_f13_dynamic_source_bootstrap import dump


def main():
    """Run all 200x200 personal-mixture tasks, retaining unsupported source fits."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=5400)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    protocol_path = args.bootstrap / "protocol_before_run.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["smoke"]:
        raise ValueError("grid requires full source protocol")
    root = Path(__file__).resolve().parents[1]
    worker = root / "scripts/run_grain_gptpro_f13_dynamic_personal.py"
    files = [
        protocol_path,
        worker,
        root / "grain_research/gptpro_foundation_v3/f13_dynamic_parameter.py",
    ]
    contract = {
        "bootstrap_root": str(args.bootstrap.resolve()),
        "source": str(args.source.resolve()),
        "files": [
            {
                "path": str(p.resolve()),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in files
        ],
        "scope": "200 parameter draws per prefix per source fit; source covariance aggregation still separate",
    }
    if args.resume:
        if (
            json.loads((args.output / "protocol_before_run.json").read_text())
            != contract
        ):
            raise ValueError("grid resume input/code mismatch")
    else:
        dump(args.output / "protocol_before_run.json", contract)
    lock = args.output / "running.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    started = time.monotonic()
    completed = 0
    try:
        for world in protocol["worlds"]:
            for seed in protocol["seeds"]:
                for b in range(protocol["outer"]):
                    output = args.output / f"{world}_{seed}" / f"b{b:03d}"
                    if (output / "summary.json").exists():
                        completed += 1
                        continue
                    source_fit = (
                        args.bootstrap / f"{world}_{seed}" / f"fit_{b:03d}.json"
                    )
                    while not source_fit.exists():
                        if time.monotonic() - started >= args.max_seconds:
                            dump(
                                args.output / "summary.json",
                                {
                                    "status": "TIME_BUDGET_CHECKPOINT",
                                    "completed": completed,
                                },
                            )
                            return
                        producer_lock = args.bootstrap / "running.lock"
                        live = False
                        if producer_lock.exists():
                            try:
                                os.kill(int(producer_lock.read_text()), 0)
                                live = True
                            except (ProcessLookupError, ValueError):
                                pass
                        if not live:
                            dump(
                                args.output / "summary.json",
                                {
                                    "status": "AWAITING_SOURCE_FIT",
                                    "completed": completed,
                                    "missing": str(source_fit),
                                },
                            )
                            return
                        time.sleep(2)
                    if time.monotonic() - started >= args.max_seconds:
                        dump(
                            args.output / "summary.json",
                            {
                                "status": "TIME_BUDGET_CHECKPOINT",
                                "completed": completed,
                            },
                        )
                        return
                    if json.loads(source_fit.read_text())["status"] != "FITTED":
                        output.mkdir(parents=True, exist_ok=False)
                        dump(
                            output / "summary.json",
                            {
                                "status": "SOURCE_FIT_UNSUPPORTED",
                                "source_fit": str(source_fit),
                            },
                        )
                    else:
                        command = [
                            sys.executable,
                            str(worker),
                            "--source-fit",
                            str(source_fit),
                            "--application",
                            str(args.source / f"{world}_{seed}" / "application.jsonl"),
                            "--output",
                            str(output),
                        ]
                        log = args.output / f"{world}_{seed}_b{b:03d}.log"
                        with log.open("w") as stream:
                            result = subprocess.run(
                                command,
                                cwd=root,
                                stdout=stream,
                                stderr=subprocess.STDOUT,
                                check=False,
                            )
                        if result.returncode:
                            dump(
                                args.output / "summary.json",
                                {
                                    "status": "WORKER_ERROR_RETAINED",
                                    "completed": completed,
                                    "log": str(log),
                                },
                            )
                            return
                    completed += 1
                    checkpoint = {
                        "status": "PERSONAL_GRID_RUNNING",
                        "pid": os.getpid(),
                        "completed": completed,
                        "world": world,
                        "seed": seed,
                        "bootstrap": b,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                    dump(args.output / "checkpoint.json", checkpoint)
                    if completed % 10 == 0:
                        print(json.dumps(checkpoint), flush=True)
        dump(
            args.output / "summary.json",
            {
                "status": "PERSONAL_GRID_COMPLETE_SOURCE_AGGREGATION_PENDING",
                "completed": completed,
            },
        )
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
