#!/usr/bin/env python3
"""Fit all frozen F04 source comparators; application scoring remains separate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "4")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.gptpro_foundation_v3.f00 import sha256
from grain_research.gptpro_foundation_v3.f04_recovery import (
    fit_source_halves,
    fit_source_lmm,
    fit_source_moments,
    generate_response_world,
    observed_arrays,
)


def dump(path, value):
    """Serialize generated numeric models, never private raw materials."""
    path.write_text(
        json.dumps(value, indent=2, default=lambda v: v.tolist(), allow_nan=False)
        + "\n"
    )


def main():
    """Generate separated role files and fit using source observations only."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("preserve previous results")
    out.mkdir(parents=True)
    (out / "source_archive").mkdir()
    started = datetime.now(timezone.utc).isoformat()
    paths = [
        Path(__file__),
        ROOT / "grain_research/gptpro_foundation_v3/f04_recovery.py",
        ROOT / "grain_research/response_sharing.py",
        ROOT.parent / "project persona/docs/GRAIN_GPTPRO_ATOMIC_PROTOCOLS_V3.md",
    ]
    for path in paths:
        (out / "source_archive" / (sha256(path) + "_" + path.name)).write_bytes(
            path.read_bytes()
        )
    dump(
        out / "input_manifest.json",
        {"files": [{"path": str(p), "sha256": sha256(p)} for p in paths]},
    )
    seeds = [20260915] if args.smoke else [20260915, 20260916, 20260917]
    counts = (
        {"source": 12, "validation": 6, "application": 6}
        if args.smoke
        else {"source": 300, "validation": 100, "application": 200}
    )
    events = 8 if args.smoke else 32
    outcomes = []
    for seed in seeds:
        for world in ("L", "T", "Q", "V", "S", "0", "H"):
            root = out / f"{world}_{seed}"
            root.mkdir()
            rows, truth = generate_response_world(
                world, seed, counts=counts, events=events
            )
            for role in counts:
                (root / f"{role}.jsonl").write_text(
                    "\n".join(json.dumps(r) for r in rows if r["role"] == role) + "\n"
                )
            dump(root / "truth.json", truth)
            # Re-read the source-only file. No estimator argument receives truth.
            source = [
                json.loads(line)
                for line in (root / "source.jsonl").read_text().splitlines()
            ]
            x, y = observed_arrays(source)
            models = {
                "pooled_OLS": {"beta": np.linalg.lstsq(x, y, rcond=None)[0]},
                "REML": fit_source_lmm(source),
                "four_cell": fit_source_moments(source, "four_cell"),
                "legacy_halves": fit_source_halves(source),
            }
            dump(root / "source_models.json", models)
            dump(
                root / "fit_manifest.json",
                {
                    "source_sha256": sha256(root / "source.jsonl"),
                    "models_sha256": sha256(root / "source_models.json"),
                    "input_roles": ["source"],
                    "source_authors": counts["source"],
                    "events_per_author": events,
                },
            )
            result = {
                "world": world,
                "seed": seed,
                "fallback": models["REML"]["fallback"],
                "attempts": models["REML"]["attempts"],
                "sigma2": models["REML"]["sigma2"],
            }
            outcomes.append(result)
            dump(
                out / "checkpoint.json",
                {"completed_source_fits": outcomes, "complete": False},
            )
            print(
                json.dumps({k: v for k, v in result.items() if k != "attempts"}),
                flush=True,
            )
    dump(
        out / "summary.json",
        {
            "status": "SOURCE_FITS_COMPLETE_APPLICATION_SCORING_PENDING",
            "smoke": args.smoke,
            "outcomes": outcomes,
            "counts": counts,
            "events": events,
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "parent_cards_promoted": [],
        },
    )
    dump(
        out / "output_manifest.json",
        {
            "files": [
                {"path": str(p.relative_to(out)), "sha256": sha256(p)}
                for p in sorted(out.rglob("*"))
                if p.is_file()
            ]
        },
    )


if __name__ == "__main__":
    main()
