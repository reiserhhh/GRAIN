#!/usr/bin/env python3
"""Generate F05 source bundles and fit noise/prior in isolated child processes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.gptpro_foundation_v3.f00 import sha256
from grain_research.gptpro_foundation_v3.f05_recovery import (
    fit_source_dynamics,
    generate_dynamics,
)


def dump(path, value):
    """Write finite JSON checkpoints, never synthetic success flags."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    """Source worker never reads validation, application or truth files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fit-source", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.fit_source:
        source_file = args.fit_source / "source.jsonl"
        rows = [json.loads(line) for line in source_file.read_text().splitlines()]
        started = time.monotonic()
        fit = fit_source_dynamics(rows)
        dump(args.fit_source / "source_model.json", fit)
        dump(
            args.fit_source / "fit_manifest.json",
            {
                "source_sha256": sha256(source_file),
                "model_sha256": sha256(args.fit_source / "source_model.json"),
                "elapsed_seconds": time.monotonic() - started,
                "successful_authors_by_round": [
                    r["successful_authors"] for r in fit["rounds"]
                ],
                "noise_success_by_round": [r["noise_success"] for r in fit["rounds"]],
            },
        )
        return
    if not args.output or args.output.exists():
        raise ValueError("new output directory required")
    out = args.output.resolve()
    out.mkdir(parents=True)
    archive = out / "source_archive"
    archive.mkdir()
    files = [
        Path(__file__),
        ROOT / "grain_research/gptpro_foundation_v3/f05_recovery.py",
        ROOT / "grain_research/gptpro_foundation_v3/f05.py",
    ]
    for path in files:
        (archive / (sha256(path) + "_" + path.name)).write_bytes(path.read_bytes())
    dump(
        out / "input_manifest.json",
        {"files": [{"path": str(p), "sha256": sha256(p)} for p in files]},
    )
    counts = (
        {"source": 12, "validation": 6, "application": 6}
        if args.smoke
        else {"source": 300, "validation": 100, "application": 200}
    )
    events = 8 if args.smoke else 64
    outcomes = []
    for seed in [20260915] if args.smoke else [20260915, 20260916, 20260917]:
        for world in ("core", "near_root", "shared_shock"):
            root = out / f"{world}_{seed}"
            root.mkdir()
            rows, truth = generate_dynamics(world, seed, counts, events)
            for role in counts:
                with (root / f"{role}.jsonl").open("w") as handle:
                    for row in rows:
                        if row["role"] == role:
                            handle.write(json.dumps(row, allow_nan=False) + "\n")
            dump(root / "truth.json", truth)
            del rows, truth
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--fit-source",
                str(root),
            ]
            with (root / "fit.log").open("w") as log:
                completed = subprocess.run(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
                )
            if completed.returncode:
                dump(
                    out / "checkpoint.json",
                    {
                        "status": "SOURCE_PROCESS_FAILED",
                        "world": world,
                        "seed": seed,
                        "completed": outcomes,
                    },
                )
                raise RuntimeError(f"source worker failed: {root / 'fit.log'}")
            fit = json.loads((root / "fit_manifest.json").read_text())
            outcomes.append({"world": world, "seed": seed, **fit})
            dump(
                out / "checkpoint.json",
                {"status": "SOURCE_FITTING", "completed": outcomes},
            )
            print(json.dumps(outcomes[-1]), flush=True)
    dump(
        out / "summary.json",
        {
            "status": "SOURCE_FITS_COMPLETE_APPLICATION_PENDING",
            "smoke": args.smoke,
            "counts": counts,
            "events": events,
            "outcomes": outcomes,
            "remaining": [
                "cold application comparison",
                "shared covariance filtering",
                "profile Hessian",
                "missing-time contrast",
                "two-state invariance",
                "scientific audit",
            ],
        },
    )
    dump(
        out / "output_manifest.json",
        {
            "files": [
                {"path": str(p.relative_to(out)), "sha256": sha256(p)}
                for p in sorted(out.rglob("*"))
                if p.is_file() and not p.name.startswith("._")
            ]
        },
    )


if __name__ == "__main__":
    main()
