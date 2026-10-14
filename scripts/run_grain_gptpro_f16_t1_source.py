#!/usr/bin/env python3
"""Build actual source-bootstrap T1 instruments, then cold-score application data."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from grain_research.gptpro_foundation_v3.f16_t1_source import (
    READER_HASH,
    code_hash,
    cold_score,
    fit_source,
    raw_score,
    sufficient,
)


def digest(path):
    """Hash exact inputs so previous T1 and new-source outputs remain comparable."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    """Write strict atomic JSON checkpoints."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def group(path, role):
    """Read synthetic F04 records with declared role and world."""
    groups = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["role"] != role or row["world"] != "L":
            raise ValueError("wrong source role/world")
        groups.setdefault(row["author_id"], []).append(row)
    return groups


def main():
    """Four-cell source refitting is not the previously executed linear bootstrap."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = [20260915] if args.smoke else [20260915, 20260916, 20260917]
    bootstraps, draws = (4, 10) if args.smoke else (200, 200)
    protocol = {
        "target": "T1=m(+1)-m(-1)",
        "world": "L",
        "seeds": seeds,
        "source_bootstraps": bootstraps,
        "personal_draws": draws,
        "primary": "analytic Gaussian source mixture; separate joint4D conditional Monte Carlo comparator",
        "bootstrap_rng": "SeedSequence([seed,16,1,0,b]) author copies",
        "integration_rng": "SeedSequence([seed,16,1,b]) normals shared across authors, never ID keyed",
        "source_estimator": "original F04 four_cell noise-corrected moments, spectral floor1e-6 with raw eigenvalues retained",
        "source_reference": "same analytic mixture scorer on source authors, ddof1 SD; no application norm",
        "scope": "historical synthetic F04 L source/app, not prospective human data",
        "runner_hash": digest(Path(__file__)),
        "scorer_hash": code_hash(),
        "smoke": args.smoke,
    }
    save(args.output / "protocol.json", protocol)
    panels = []
    for seed in seeds:
        root = args.source_root / f"L_{seed}"
        output = args.output / str(seed)
        output.mkdir()
        people = group(root / "source.jsonl", "source")
        if len(people) != 300:
            raise ValueError("original source panel incomplete")
        ids = sorted(people)[:12] if args.smoke else sorted(people)
        rows = [people[author] for author in ids]
        stats = tuple(
            np.stack(values)
            for values in zip(*(sufficient(value) for value in rows), strict=True)
        )
        plugin = fit_source(stats, np.arange(len(ids)))
        save(output / "plugin_model.json", plugin)
        fits, indices = [], []
        for b in range(bootstraps):
            selected = np.random.default_rng(
                np.random.SeedSequence([seed, 16, 1, 0, b])
            ).integers(0, len(ids), len(ids))
            model = fit_source(stats, selected)
            fits.append(model)
            indices.append(selected.tolist())
            if (b + 1) % 10 == 0 or b + 1 == bootstraps:
                save(
                    output / "source_fits.json",
                    {"source_order": ids, "indices": indices, "models": fits},
                )
                print(f"T1 seed={seed} sourcefits={b + 1}/{bootstraps}", flush=True)
        bundle = {
            "target_id": "T1",
            "object_class": "conditional_response_function",
            "invalidating_result": "source dependence, missing-condition model misspecification or failed interval calibration invalidates corresponding reliability claim",
            "formula_version": "T1_four_cell_source_mixture_v1",
            "observed_role": "synthetic author response",
            "unit": "endpoint contrast",
            "reference_mu": None,
            "reference_sigma": None,
            "source_hash": digest(root / "source.jsonl"),
            "source_fits_hash": digest(output / "source_fits.json"),
            "reader_hash": READER_HASH,
            "world_contract": "F04_L_four_condition",
            "scorer_code_hash": code_hash(),
            "seed": seed,
            "support_rule": "both endpoints observed or explicitly model-borrowed; empty prefix prior-only",
            "error_model": "conditional Gaussian source bootstrap mixture; source fit spectral floor1e-6 retained",
            "explanation_rule": "separate observed conditional contrast, prior borrowing and source uncertainty; unnamed simulated construct",
            "source_bootstraps": bootstraps,
            "personal_draws": draws,
            "source_models": fits,
            "source_ids": ids,
            "source_reference_authors": len(ids),
            "calibration": "NOT_EVALUATED",
        }
        scores = [raw_score(row, bundle) for row in rows]
        values = np.asarray([score["score"] for score in scores])
        bundle["reference_mu"] = float(values.mean())
        bundle["reference_sigma"] = float(values.std(ddof=1))
        save(output / "bundle.json", bundle)
        save(output / "source_scores.json", scores)
        panels.append((seed, root, output, bundle, set(people)))
    # All sources and references are frozen before the first application read.
    summaries = []
    repo = Path(__file__).resolve().parents[1]
    for seed, root, output, bundle, source_ids in panels:
        people = group(root / "application.jsonl", "application")
        if len(people) != 200 or set(people) & source_ids:
            raise ValueError("application panel incomplete or source overlap")
        ids = sorted(people)[:6] if args.smoke else sorted(people)
        inputs = [
            {
                "world_contract": bundle["world_contract"],
                "reader_hash": READER_HASH,
                "records": people[author],
            }
            for author in ids
        ]
        manifest = {
            "application_sha256": digest(root / "application.jsonl"),
            "author_order": ids,
            "bundle_sha256": digest(output / "bundle.json"),
        }
        save(output / "application_manifest.json", manifest)
        warm = [cold_score(raw, bundle) for raw in inputs]
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            (tmp / "bundle.json").write_bytes((output / "bundle.json").read_bytes())
            (tmp / "input.jsonl").write_text(
                "".join(json.dumps(raw) + "\n" for raw in inputs)
            )
            command = [
                sys.executable,
                "-m",
                "grain_research.gptpro_foundation_v3.f16_t1_source",
                "--bundle",
                "bundle.json",
                "--sha256",
                manifest["bundle_sha256"],
                "--input",
                "input.jsonl",
                "--output",
                "scores.jsonl",
            ]
            with (output / "cold_process.log").open("w") as log:
                subprocess.run(
                    command,
                    cwd=tmp,
                    env=dict(
                        os.environ,
                        PYTHONPATH=str(repo),
                        OPENBLAS_NUM_THREADS="1",
                        OMP_NUM_THREADS="1",
                    ),
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            (output / "cold_scores.jsonl").write_bytes(
                (tmp / "scores.jsonl").read_bytes()
            )
        results = [
            json.loads(line)
            for line in (output / "cold_scores.jsonl").read_text().splitlines()
        ]
        if warm != results:
            raise AssertionError("T1 source-uncertainty cold/warm mismatch")
        result = {
            "seed": seed,
            "source_authors": len(bundle["source_ids"]),
            "application_authors": len(ids),
            "source_bootstraps": bootstraps,
            "personal_draws": draws,
            "all_objects_equal": True,
            "cold_warm_max_error": 0.0,
            "scores_sha256": digest(output / "cold_scores.jsonl"),
            "calibration": "NOT_EVALUATED",
        }
        summaries.append(result)
        save(output / "summary.json", result)
    save(
        args.output / "summary.json",
        {
            "status": "T1_SOURCE_COLD_COMPLETE",
            "smoke": args.smoke,
            "cells": summaries,
            "parent_F16_complete": False,
        },
    )
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
