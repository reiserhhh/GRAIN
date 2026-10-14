#!/usr/bin/env python3
"""Apply independent T3 bundles, then perform a separate cold-process replay."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "grain_research/gptpro_foundation_v3/f16_independent_t3.py"
PROTOCOL = ROOT / "configs/grain_f16_t3_application_v1.json"


def digest(path):
    """Hash exact evidence bytes, excluding no part of an input or bundle."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    """Atomic resumable checkpoint; no new run overwrites another output root."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def load_scorer(path):
    """Load only the independent implementation, not its build runner."""
    spec = importlib.util.spec_from_file_location("independent_t3_application", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_prefixes(path, bundle, count, prefix):
    """Keep application identity/order separate from the numerical model input."""
    groups = {}
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["world"] != "core" or row["role"] != "application":
                raise ValueError("application must belong to frozen core world")
            groups.setdefault(row["author_id"], []).append(row)
    if len(groups) != 200 or set(groups) & set(bundle["source_ids"]):
        raise ValueError("application count or source overlap invalid")
    ids = sorted(groups)[:count]
    inputs = []
    for author in ids:
        rows = groups[author][:prefix]
        if len(rows) != prefix or [row["t"] for row in rows] != list(range(prefix)):
            raise ValueError("application prefix coverage/order invalid")
        if any(row["z"] not in (-1, 1) for row in rows):
            raise ValueError("condition outside declared core input schedule")
        inputs.append(
            {"world": "core", "reader_hash": bundle["reader_hash"], "prefix": rows}
        )
    return ids, inputs


def cold(args):
    """Standalone process sees a copied scorer, trusted bundle and observed inputs."""
    module = load_scorer(args.scorer)
    bundle = module.load_bundle(args.bundle, args.sha256, args.source_hash)
    if args.output.exists():
        raise ValueError("cold output must be new")
    with args.input.open() as stream, args.output.open("x") as output:
        for line in stream:
            value = json.loads(line)
            result = module.cold_score(value, bundle)
            output.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
            output.flush()
    return 0


def run_cell(task):
    """Reuse a frozen source reference; no source refit, truth or validation read."""
    build_root, source_root, output_root, seed, smoke, resume = task
    started = time.monotonic()
    module = load_scorer(SCORER)
    bundle_path = build_root / f"core_{seed}" / "bundle.json"
    expected = bundle_path.with_suffix(".json.sha256").read_text().strip()
    bundle = module.load_bundle(bundle_path, expected)
    if (
        bundle["source_authors"] != 300
        or bundle["source_bootstraps"] != 200
        or bundle["personal_draws"] != 200
        or bundle["reference_prefix"] != 48
    ):
        raise ValueError("incomplete independent source construction")
    output = output_root / str(seed)
    output.mkdir(exist_ok=resume)
    protocol = {
        "seed": seed,
        "smoke": smoke,
        "bundle_sha256": expected,
        "source_hash": bundle["source_hash"],
        "scorer_hash": digest(SCORER),
        "runner_hash": digest(__file__),
        "evaluation_brief_hash": digest(output_root / "protocol.json"),
    }
    manifest = output / "protocol_before_application.json"
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError("resume identity changed")
    save(manifest, protocol)
    if not (output / "bundle.json").exists():
        shutil.copyfile(bundle_path, output / "bundle.json")
    elif digest(output / "bundle.json") != expected:
        raise ValueError("copied bundle changed")
    application = source_root / f"core_{seed}" / "application.jsonl"
    ids, inputs = check_prefixes(application, bundle, 2 if smoke else 200, 48)
    application_manifest = {
        "file": str(application.resolve()),
        "sha256": digest(application),
        "author_order": ids,
        "input_hashes": [module.object_hash(value) for value in inputs],
        "prefix": 48,
    }
    path = output / "application_manifest.json"
    if path.exists() and json.loads(path.read_text()) != application_manifest:
        raise ValueError("application input changed")
    save(path, application_manifest)
    warm_dir = output / "warm"
    warm_dir.mkdir(exist_ok=True)
    warm = []
    for i, value in enumerate(inputs):
        path = warm_dir / f"{i:03d}.json"
        if path.exists():
            result = json.loads(path.read_text())
        else:
            result = module.cold_score(value, bundle)
            save(path, result)
        warm.append(result)
        if not result["supported"]:
            save(output / "unsupported.json", {"author": ids[i], "score": result})
            raise ValueError("unsupported application retained, not excluded")
        if (i + 1) % 10 == 0 or i + 1 == len(inputs):
            save(
                output / "checkpoint.json",
                {"phase": "warm", "completed": i + 1, "total": len(inputs)},
            )
            print(f"T3 seed={seed} warm={i + 1}/{len(inputs)}", flush=True)
    cold_path = output / "cold_scores.jsonl"
    if not cold_path.exists():
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            shutil.copyfile(SCORER, tmp / "scorer.py")
            shutil.copyfile(bundle_path, tmp / "bundle.json")
            (tmp / "input.jsonl").write_text(
                "".join(json.dumps(value) + "\n" for value in inputs)
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "cold",
                "--scorer",
                "scorer.py",
                "--bundle",
                "bundle.json",
                "--sha256",
                expected,
                "--source-hash",
                bundle["source_hash"],
                "--input",
                "input.jsonl",
                "--output",
                "scores.jsonl",
            ]
            save(output / "checkpoint.json", {"phase": "cold", "authors": len(inputs)})
            with (output / "cold_process.log").open("w") as log:
                subprocess.run(
                    command,
                    cwd=tmp,
                    env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            shutil.copyfile(tmp / "scores.jsonl", cold_path)
            save(
                output / "cold_inputs.json",
                {
                    "scorer_sha256": digest(tmp / "scorer.py"),
                    "bundle_sha256": digest(tmp / "bundle.json"),
                    "input_sha256": digest(tmp / "input.jsonl"),
                    "interface": "only declared file inputs; not an operating-system sandbox",
                },
            )
    cold_results = [json.loads(line) for line in cold_path.read_text().splitlines()]
    if cold_results != warm:
        raise AssertionError("independent scorer cold/warm objects differ")
    if digest(SCORER) != protocol["scorer_hash"] or digest(bundle_path) != expected:
        raise ValueError("implementation/bundle changed during scoring")
    result = {
        "seed": seed,
        "status": "T3_COLD_REPLAY_COMPLETE",
        "application_authors": len(ids),
        "source_authors": 300,
        "cold_warm_max_error": 0.0,
        "all_objects_equal": True,
        "cold_scores_sha256": digest(cold_path),
        "elapsed_seconds": time.monotonic() - started,
        "calibration": "NOT_EVALUATED",
        "source_refits": 0,
    }
    save(output / "summary.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    """Run the full three-seed application separately from the independent build."""
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    child = subs.add_parser("cold")
    for name in ("scorer", "bundle", "input", "output"):
        child.add_argument(f"--{name}", type=Path, required=True)
    child.add_argument("--sha256", required=True)
    child.add_argument("--source-hash", required=True)
    run = subs.add_parser("run")
    for name in ("build-root", "source-root", "output"):
        run.add_argument(f"--{name}", type=Path, required=True)
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "cold":
        return cold(args)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    frozen = args.output / "protocol.json"
    if frozen.exists() and frozen.read_bytes() != PROTOCOL.read_bytes():
        raise ValueError("application/evaluation protocol changed")
    if not frozen.exists():
        shutil.copyfile(PROTOCOL, frozen)
    seeds = [20260915] if args.smoke else [20260915, 20260916, 20260917]
    jobs = [
        (
            args.build_root.resolve(),
            args.source_root.resolve(),
            args.output.resolve(),
            seed,
            args.smoke,
            args.resume,
        )
        for seed in seeds
    ]
    with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
        cells = list(executor.map(run_cell, jobs))
    summary = {
        "status": "T3_COLD_REPLAY_COMPLETE",
        "smoke": args.smoke,
        "cells": cells,
        "parent_F16_complete": False,
    }
    save(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
