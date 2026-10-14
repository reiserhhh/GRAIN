#!/usr/bin/env python3
"""Run explicit-clock masks with frozen source fits and no future/truth scoring."""

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

import numpy as np

from grain_research.gptpro_foundation_v3 import f16_cutoff as adapter
from grain_research.gptpro_foundation_v3.f16_cutoff_audit import dense_moments

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/grain_gptpro_foundation_v3/F16"
CONFIG = ROOT / "configs/grain_f16_cutoff_v1.json"


def digest(path):
    """Record bytes, not a path or a previous completion flag."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    """Write an atomic checkpoint in an explicitly selected new/resumed run."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def masked(rows, case, config):
    """Use the frozen missingness masks while retaining original event clocks."""
    if case == "all_missing":
        return []
    if case == "only_first":
        return rows[:1]
    excluded = set()
    if case in ("internal_missing", "combined_missing"):
        excluded.update(config["internal_excluded_times"])
    if case in ("trailing_missing", "combined_missing"):
        excluded.update(config["trailing_excluded_times"])
    if case not in ("full", "internal_missing", "trailing_missing", "combined_missing"):
        raise ValueError("unknown fixed case")
    return [r for r in rows if r["t"] not in excluded]


def run_cell(task):
    """Apply fixed masks; source bundles remain unchanged and truth is never read."""
    target, seed, output_root, config, resume = task
    started = time.monotonic()
    oldroot = (
        RESULTS
        / ("t2_cold_full_r1" if target == "T2" else "t3_application_full_r1")
        / str(seed)
    )
    oldbundle = oldroot / "bundle.json"
    base = json.loads(oldbundle.read_text())
    count = len(base["source_models"]) if target == "T2" else base["source_bootstraps"]
    if (count, base["personal_draws"]) != (
        config["source_bootstraps"],
        config["personal_draws"],
    ):
        raise ValueError("source integration budget mismatch")
    output = output_root / f"{target}_{seed}"
    output.mkdir(exist_ok=resume)
    bundle = adapter.build_adapter(base)
    protocol = {
        "target": target,
        "seed": seed,
        "config": config,
        "base_bundle_sha256": digest(oldbundle),
        "adapter_hash": adapter.code_hash(),
        "runner_hash": digest(__file__),
        "audit_hash": digest(
            ROOT / "grain_research/gptpro_foundation_v3/f16_cutoff_audit.py"
        ),
    }
    path = output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("resume protocol/source/code mismatch")
    save(path, protocol)
    save(output / "bundle.json", bundle)
    appmanifest = json.loads((oldroot / "application_manifest.json").read_text())
    source_manifest = json.loads(
        (
            RESULTS / "t3_application_full_r1" / str(seed) / "application_manifest.json"
        ).read_text()
    )
    appfile = Path(source_manifest["file"])
    if digest(appfile) != source_manifest["sha256"]:
        raise ValueError("archived application changed")
    if target == "T2" and appmanifest["input_hash"] != source_manifest["sha256"]:
        raise ValueError("T2/T3 material mismatch")
    groups = {}
    for line in appfile.read_text().splitlines():
        row = json.loads(line)
        if row["t"] <= config["target_time"]:
            groups.setdefault(row["author_id"], []).append(row)
    if sorted(groups) != appmanifest["author_order"]:
        raise ValueError("application author identity/order mismatch")
    save(
        output / "input_manifest.json",
        {
            "application": str(appfile),
            "sha256": digest(appfile),
            "bundle_sha256": digest(oldbundle),
            "config_sha256": digest(CONFIG),
        },
    )
    all_inputs, results = [], []
    scores_dir = output / "scores"
    scores_dir.mkdir(exist_ok=True)
    for author_index in config["author_indices"]:
        author = appmanifest["author_order"][author_index]
        rows = groups[author]
        if [r["t"] for r in rows] != list(range(config["target_time"] + 1)):
            raise ValueError("full input must have the declared ordered prefix")
        for case in config["cases"]:
            selected = masked(rows, case, config)
            raw = {
                "world": "core",
                "reader_hash": base["reader_hash"],
                "records": selected,
                "target_time": config["target_time"],
            }
            all_inputs.append(raw)
            key = f"{author_index:03d}_{case}"
            path = scores_dir / f"{key}.json"
            if path.exists():
                entry = json.loads(path.read_text())
                if entry["input_hash"] != adapter.object_hash(raw):
                    raise ValueError("resume mask changed")
            else:
                audit_rows = []

                def check(
                    bootstrap,
                    parameters,
                    noise,
                    mean,
                    variance,
                    selected=selected,
                    audit_rows=audit_rows,
                ):
                    if bootstrap not in config["dense_audit_source_indices"]:
                        return
                    dm, dv = dense_moments(
                        parameters, selected, noise, config["target_time"], target
                    )
                    np.testing.assert_allclose(
                        mean, dm, atol=config["atol"], rtol=config["rtol"]
                    )
                    np.testing.assert_allclose(
                        variance, dv, atol=config["atol"], rtol=config["rtol"]
                    )
                    audit_rows.append(
                        {
                            "bootstrap": bootstrap,
                            "parameter_draws": len(parameters),
                            "max_mean_error": float(np.max(np.abs(mean - dm))),
                            "max_variance_error": float(np.max(np.abs(variance - dv))),
                        }
                    )

                score = adapter.cold_score(raw, bundle, audit_callback=check)
                entry = {
                    "author_id": author,
                    "author_index": author_index,
                    "case": case,
                    "input_hash": adapter.object_hash(raw),
                    "input": raw,
                    "score": score,
                    "dense_audit": audit_rows,
                }
                if case == "full" and score["supported"]:
                    archived = json.loads(
                        (oldroot / "warm" / f"{author_index:03d}.json").read_text()
                    )
                    expected = (
                        [
                            archived["score"],
                            archived["variance"],
                            archived["reference_score"],
                            archived["reference_variance"],
                        ]
                        if target == "T2"
                        else [
                            archived["raw_score"],
                            archived["variance_components"]["total"],
                            archived["source_standardized_score"],
                            archived["source_standardized_variance"],
                        ]
                    )
                    actual = [
                        score[k]
                        for k in (
                            "score",
                            "variance",
                            "reference_score",
                            "reference_variance",
                        )
                    ]
                    np.testing.assert_allclose(
                        actual, expected, atol=config["atol"], rtol=config["rtol"]
                    )
                    entry["legacy_max_error"] = float(
                        np.max(np.abs(np.array(actual) - expected))
                    )
                    entry["legacy_score_hash"] = digest(
                        oldroot / "warm" / f"{author_index:03d}.json"
                    )
                save(path, entry)
            results.append(entry)
            save(
                output / "checkpoint.json",
                {
                    "phase": "warm",
                    "completed": len(results),
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
            print(
                f"clock {target} {seed} {key} supported={entry['score']['supported']}",
                flush=True,
            )
    cold_path = output / "cold_scores.jsonl"
    if not cold_path.exists():
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            (tmp / "bundle.json").write_bytes((output / "bundle.json").read_bytes())
            (tmp / "inputs.jsonl").write_text(
                "".join(json.dumps(x) + "\n" for x in all_inputs)
            )
            command = [
                sys.executable,
                "-m",
                adapter.__name__,
                "--bundle",
                "bundle.json",
                "--sha256",
                digest(tmp / "bundle.json"),
                "--input",
                "inputs.jsonl",
                "--output",
                "scores.jsonl",
            ]
            subprocess.run(
                command,
                cwd=tmp,
                check=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(ROOT),
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                },
            )
            cold_path.write_bytes((tmp / "scores.jsonl").read_bytes())
    cold = [json.loads(s) for s in cold_path.read_text().splitlines()]
    if cold != [e["score"] for e in results]:
        raise ValueError("cold/warm object mismatch")
    summary = {
        "target": target,
        "seed": seed,
        "objects": len(results),
        "unique_authors": len(config["author_indices"]),
        "unsupported": sum(not e["score"]["supported"] for e in results),
        "cold_exact": True,
        "dense_checks": sum(
            a["parameter_draws"] for e in results for a in e["dense_audit"]
        ),
        "max_dense_error": max(
            max(a["max_mean_error"], a["max_variance_error"])
            for e in results
            for a in e["dense_audit"]
        ),
        "max_legacy_error": max(e.get("legacy_max_error", 0.0) for e in results),
        "elapsed_seconds": time.monotonic() - started,
    }
    save(output / "summary.json", summary)
    return summary


def main():
    """Freeze the bounded mask protocol before opening application observations."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    config = json.loads(CONFIG.read_text())
    if args.smoke:
        config.update(
            seeds=config["seeds"][:1],
            author_indices=[0],
            cases=["full", "trailing_missing", "all_missing"],
        )
    args.output.mkdir(parents=True, exist_ok=args.resume)
    frozen = args.output / "protocol.json"
    protocol = {
        "config": config,
        "config_file_hash": digest(CONFIG),
        "smoke": args.smoke,
        "adapter_hash": adapter.code_hash(),
        "runner_hash": digest(__file__),
    }
    if frozen.exists() and json.loads(frozen.read_text()) != protocol:
        raise ValueError("resume protocol mismatch")
    save(frozen, protocol)
    tasks = [
        (target, seed, args.output, config, args.resume)
        for target in config["targets"]
        for seed in config["seeds"]
    ]
    with ProcessPoolExecutor(max_workers=min(3, len(tasks))) as pool:
        rows = list(pool.map(run_cell, tasks))
    summary = {
        "cells": rows,
        "objects": sum(r["objects"] for r in rows),
        "unique_authors": len(config["seeds"]) * len(config["author_indices"]),
        "unsupported": sum(r["unsupported"] for r in rows),
        "dense_checks": sum(r["dense_checks"] for r in rows),
        "max_dense_error": max(r["max_dense_error"] for r in rows),
        "max_legacy_error": max(r["max_legacy_error"] for r in rows),
        "cold_exact": all(r["cold_exact"] for r in rows),
        "scope": "Numeric explicit-clock interface only; no eligibility classifier, new calibration, source fitting or human validity",
        "parent_F16_complete": False,
    }
    save(args.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
