#!/usr/bin/env python3
"""Build source-frozen T1 bundles and verify actual cold process scoring."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from grain_research.gptpro_foundation_v3.f16_instrument import (
    READER_HASH,
    build_instrument,
    cold_score,
)


def save(path, value):
    """Write reproducible strict JSON artifacts."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    """Freeze briefs/bundles before opening application records, then compare."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    brief = {"target_id": "T1", "world": "L", "target": "m(+1)-m(-1)"}
    save(args.output / "brief.json", brief)
    repo = Path(__file__).resolve().parents[1]
    bundles = []
    for seed in (20260915, 20260916, 20260917):
        root = args.source_root / f"L_{seed}"
        manifest = {}
        for key, filename in (
            ("source", "source.jsonl"),
            ("models", "source_models.json"),
        ):
            path = (root / filename).resolve()
            manifest[key] = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        bundle = build_instrument(brief, manifest)
        folder = args.output / str(seed)
        folder.mkdir()
        save(folder / "source_manifest.json", manifest)
        save(folder / "bundle.json", bundle)
        bundles.append((root, folder, bundle))
    checks = []
    for root, folder, bundle in bundles:
        app = root / "application.jsonl"
        groups = {}
        for line in app.read_text().splitlines():
            row = json.loads(line)
            if row["role"] != "application":
                raise ValueError("wrong application role")
            groups.setdefault(row["author_id"], []).append(row)
        inputs = [
            {
                "world_contract": bundle["world_contract"],
                "reader_hash": READER_HASH,
                "records": rows,
            }
            for rows in groups.values()
        ]
        warm = [cold_score(raw, bundle) for raw in inputs]
        bundle_bytes = (folder / "bundle.json").read_bytes()
        digest = hashlib.sha256(bundle_bytes).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            cold = Path(tmp)
            (cold / "bundle.json").write_bytes(bundle_bytes)
            (cold / "input.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in inputs)
            )
            env = dict(
                os.environ,
                PYTHONPATH=str(repo),
                OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS="1",
            )
            command = [
                sys.executable,
                "-m",
                "grain_research.gptpro_foundation_v3.f16_instrument",
                "--bundle",
                "bundle.json",
                "--sha256",
                digest,
                "--input",
                "input.jsonl",
                "--output",
                "scores.jsonl",
            ]
            subprocess.run(
                command, cwd=cold, env=env, check=True, capture_output=True, text=True
            )
            results = [
                json.loads(line)
                for line in (cold / "scores.jsonl").read_text().splitlines()
            ]
            (folder / "scores.jsonl").write_bytes((cold / "scores.jsonl").read_bytes())
        if len(results) != len(warm):
            raise AssertionError("cold result count mismatch")
        error = max(
            abs(a[k] - b[k])
            for a, b in zip(warm, results, strict=True)
            for k in ("score", "variance", "reference_score", "reference_variance")
        )
        # Independently solve the observation-space Gaussian conditioning equations.
        direct_error = 0.0
        conditions = (-1.0, -1 / 3, 1 / 3, 1.0)
        m = np.asarray(bundle["model"]["mu"])
        v = np.asarray(bundle["model"]["Sigma"])
        c = np.array([-1.0, 0.0, 0.0, 1.0])
        for raw, result in zip(inputs, results, strict=True):
            rows = raw["records"]
            x = np.eye(4)[[conditions.index(r["z"]) for r in rows]]
            y = np.array([r["y"] for r in rows])
            k = x @ v @ x.T + bundle["model"]["sigma2"] * np.eye(len(rows))
            mean = m + v @ x.T @ np.linalg.solve(k, y - x @ m)
            covariance = v - v @ x.T @ np.linalg.solve(k, x @ v)
            direct_error = max(
                direct_error,
                abs(float(c @ mean) - result["score"]),
                abs(float(c @ covariance @ c) - result["variance"]),
            )
        if error > 1e-10 or direct_error > 1e-10:
            raise AssertionError("replay mismatch")
        save(
            folder / "application_manifest.json",
            {
                "sha256": hashlib.sha256(app.read_bytes()).hexdigest(),
                "author_order": list(groups),
                "bundle_sha256": digest,
            },
        )
        checks.append(
            {
                "seed": folder.name,
                "authors": len(results),
                "cold_warm_max_error": error,
                "direct_solve_max_error": direct_error,
                "source_authors": bundle["source_reference_authors"],
            }
        )
    save(
        args.output / "summary.json",
        {
            "target": "T1",
            "checks": checks,
            "completed": "source-frozen four-cell scoring and separate-process replay",
            "calibration": "NOT_EVALUATED",
            "independent_builder": "PENDING",
            "open": [
                "source uncertainty",
                "T2/T3",
                "evidence cards",
                "independent agent construction",
            ],
            "data": "SIM only; no human validity claim",
        },
    )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
