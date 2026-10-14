#!/usr/bin/env python3
"""Run resumable F07 categorical MAP experiments with frozen channel designs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f07_channels import (
    COSTS,
    OPPORTUNITIES,
    binary_law,
    ordinal_law,
    world_parameters,
)
from grain_research.gptpro_foundation_v3.f07_estimation import (
    fit_channel,
)


def write_json(path, data):
    """Atomically replace a generated JSON checkpoint without corrupting resume."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def designs():
    """Yield every frozen binary design plus a separate ordinal opportunity reference."""
    for world in ("U", "D", "P", "flat"):
        presence, flips = world_parameters(world)
        for variant in ("base", "offtarget", "mnar"):
            for level in range(5):
                for budget in ("tokens400", "opportunities32"):
                    yield {
                        "world": world,
                        "variant": variant,
                        "level": level + 1,
                        "budget": budget,
                        "n": int(400 // COSTS[level] * OPPORTUNITIES[level])
                        if budget == "tokens400"
                        else 32,
                        "presence": float(presence[level]),
                        "flip": float(flips[level]),
                        "ordinal": False,
                    }
    # No token cost for ordinal text was specified: do not invent one.
    yield {
        "world": "ordinal",
        "variant": "ordinal",
        "level": None,
        "budget": "opportunities32",
        "n": 32,
        "presence": 1.0,
        "flip": 0.0,
        "ordinal": True,
    }


def simulate_author(seed, role_index, author_index):
    """Use independent author RNGs and shared uniform opportunities across arms."""
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, role_index, author_index, 707])
    )
    truth = rng.normal([0.0, 0.5], [1.0, 0.5])
    uniforms = rng.uniform(size=100)
    rows = []
    for design in designs():
        n = design["n"]
        conditions = np.where(np.arange(n) % 2 == 0, -1.0, 1.0)
        counts = []
        for z in (-1.0, 1.0):
            probabilities, _ = (
                ordinal_law(truth, z)
                if design["ordinal"]
                else binary_law(
                    truth, z, design["presence"], design["flip"], design["variant"]
                )
            )
            outcomes = np.searchsorted(
                np.cumsum(probabilities), uniforms[:n][conditions == z]
            )
            counts.append(np.bincount(outcomes, minlength=len(probabilities)))
        counts = np.asarray(counts)
        fit = fit_channel(
            counts,
            design["presence"],
            design["flip"],
            "base" if design["ordinal"] else design["variant"],
            design["ordinal"],
        )
        estimate, covariance = fit.pop("estimate"), fit.pop("covariance")
        error = estimate - truth
        rows.append(
            {
                **design,
                "counts": counts.tolist(),
                **fit,
                "estimate": estimate.tolist(),
                "covariance": covariance.tolist(),
                "error": error.tolist(),
                "squared_error": (error**2).tolist(),
                "coverage95": (
                    np.abs(error) <= 1.96 * np.sqrt(np.diag(covariance))
                ).tolist(),
            }
        )
    return {"truth": truth.tolist(), "uniforms": uniforms.tolist(), "rows": rows}


def main():
    """Checkpoint whole authors; resume refuses code/design drift and corrupt packets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=1800)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    roles = (
        {"source": 2, "validation": 1, "application": 2}
        if args.smoke
        else {"source": 300, "validation": 100, "application": 200}
    )
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__),
            ROOT / "grain_research/gptpro_foundation_v3/f07_channels.py",
            ROOT / "grain_research/gptpro_foundation_v3/f07_estimation.py",
        )
    }
    specification = {
        "roles_per_seed": roles,
        "seeds": [20260915, 20260916, 20260917],
        "code_hashes": hashes,
        "prior": {"mean": [0, 0.5], "variance": [1, 0.25]},
        "designs": list(designs()),
        "calibration": "fixed theoretical prior; no application selection",
    }
    manifest = args.output / "manifest.json"
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text()) != specification:
            raise RuntimeError(
                "existing packet requires explicit resume with identical specification"
            )
    else:
        write_json(manifest, specification)
    started = time.monotonic()
    completed, fits, failures = 0, 0, 0
    all_errors = {}
    expected = 3 * sum(roles.values())
    for seed in specification["seeds"]:
        for role_index, (role, size) in enumerate(roles.items()):
            for index in range(size):
                path = args.output / f"author_{seed}_{role}_{index:04d}.json.gz"
                if path.exists():
                    with gzip.open(path, "rt") as stream:
                        packet = json.load(stream)
                else:
                    if time.monotonic() - started >= args.max_seconds:
                        write_json(
                            args.output / "checkpoint.json",
                            {
                                "status": "TIME_LIMIT_RESUMABLE",
                                "completed_authors": completed,
                                "expected_authors": expected,
                                "fits": fits,
                                "optimizer_non_success": failures,
                            },
                        )
                        print("TIME_LIMIT_RESUMABLE", completed, flush=True)
                        return
                    packet = {
                        "seed": seed,
                        "role": role,
                        "index": index,
                        **simulate_author(seed, role_index, index),
                    }
                    temporary = path.with_suffix(".tmp")
                    with gzip.open(temporary, "wt") as stream:
                        json.dump(packet, stream, allow_nan=False)
                    temporary.replace(path)
                if (packet["seed"], packet["role"], packet["index"]) != (
                    seed,
                    role,
                    index,
                ) or len(packet["rows"]) != 121:
                    raise RuntimeError("packet identity/design mismatch")
                completed += 1
                fits += len(packet["rows"])
                failures += sum(not row["success"] for row in packet["rows"])
                if role == "application":
                    for row in packet["rows"]:
                        key = (
                            row["world"],
                            row["variant"],
                            row["level"],
                            row["budget"],
                        )
                        all_errors.setdefault(key, []).append(row["squared_error"])
                if completed % 50 == 0:
                    print(
                        f"authors={completed}/{expected} fits={fits} non_success={failures}",
                        flush=True,
                    )
    summaries = [
        {
            "world": key[0],
            "variant": key[1],
            "level": key[2],
            "budget": key[3],
            "n_application": len(values),
            "mse_a_b": np.mean(values, axis=0).tolist(),
        }
        for key, values in all_errors.items()
    ]
    write_json(args.output / "application_summary.json", summaries)
    write_json(
        args.output / "checkpoint.json",
        {
            "status": "COMPUTED_AUDIT_PENDING",
            "completed_authors": completed,
            "expected_authors": expected,
            "fits": fits,
            "optimizer_non_success": failures,
            "elapsed_this_invocation": time.monotonic() - started,
        },
    )
    print(
        json.dumps(json.loads((args.output / "checkpoint.json").read_text())),
        flush=True,
    )


if __name__ == "__main__":
    main()
