#!/usr/bin/env python3
"""Frozen F08 source-author bootstrap with actual B1/B2/M refits and audit."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from grain_research.gptpro_foundation_v3.f02 import generate_world
from grain_research.gptpro_foundation_v3.f13_binary_source import (
    author_statistics,
    refit_statistics,
)
from grain_research.measurement_discovery.estimators import ordered_rows
from scripts.audit_grain_gptpro_f08_states import source_replay


def serial(value):
    """Serialize fitted numeric arrays without pickled executable payloads."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def main():
    """Fit source only and replay original row-level definitions for every fit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = json.loads((args.input / "protocol.json").read_text())
    seeds = original["seeds"][:1] if args.smoke else original["seeds"]
    counts = (
        {"source": 12, "validation": 6, "application": 6}
        if args.smoke
        else original["counts"]
    )
    outer = 2 if args.smoke else 200
    files = [
        Path(__file__),
        Path("grain_research/gptpro_foundation_v3/f13_binary_source.py"),
        Path("grain_research/gptpro_foundation_v3/f02.py"),
        Path("scripts/audit_grain_gptpro_f08_states.py"),
        args.input / "protocol.json",
    ]
    protocol = {
        "counts": counts,
        "seeds": seeds,
        "outer": outer,
        "unit": "source author, full 48-event history",
        "rng": "SeedSequence([seed,13,8,world_index,bootstrap])",
        "scope": "source fits only; reader uncertainty and personal uncertainty remain separate",
        "files": [
            {
                "path": str(p.resolve()),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in files
        ],
    }
    (args.output / "protocol_before_run.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    started, done = time.monotonic(), 0
    for wi, world in enumerate(("c2_xor", "c2_iid")):
        for seed in seeds:
            records, _ = generate_world(world, seed, counts=counts)
            groups = ordered_rows([r for r in records if r["role"] == "source"])
            ids = sorted(groups)
            stats = author_statistics([groups[a] for a in ids])
            for b in range(outer):
                recipe = [seed, 13, 8, wi, b]
                indices = np.random.default_rng(
                    np.random.SeedSequence(recipe)
                ).integers(0, len(ids), len(ids))
                fit = refit_statistics(stats, indices)
                copies = []
                for j, i in enumerate(indices):
                    copies.extend(
                        r | {"author_id": f"copy_{j:04d}"} for r in groups[ids[i]]
                    )
                initial, mean = source_replay(copies, fit)
                np.testing.assert_allclose(initial, fit["initial_pair"], atol=1e-12)
                np.testing.assert_allclose(mean, fit["source_mean"], atol=1e-12)
                payload = {
                    "world": world,
                    "seed": seed,
                    "bootstrap": b,
                    "rng_recipe": recipe,
                    "source_author_ids": ids,
                    "sampled_indices": indices,
                    "fit": fit,
                    "status": "SOURCE_FIT_ROW_REPLAY_PASS",
                }
                (args.output / f"{world}_{seed}_b{b:03d}.json").write_text(
                    json.dumps(payload, default=serial) + "\n"
                )
                done += 1
                if done % 10 == 0:
                    (args.output / "checkpoint.json").write_text(
                        json.dumps(
                            {"completed": done, "elapsed": time.monotonic() - started}
                        )
                        + "\n"
                    )
    summary = {
        "status": "SOURCE_BOOTSTRAP_FITS_AND_ROW_REPLAY_COMPLETE",
        "fits": done,
        "application_scoring": "NOT_RUN",
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
