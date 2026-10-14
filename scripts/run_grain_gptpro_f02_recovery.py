#!/usr/bin/env python3
"""Recover the complete frozen F02 numerical battery with process-separated truth.

This runner restores omitted experiments. It does not promote downstream DAG
cards or replace source-level scientific review with a successful exit code.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent / "project persona"
sys.path.insert(0, str(ROOT))

from grain_research.gptpro_foundation_v3.f00 import sha256, validate_record
from grain_research.gptpro_foundation_v3.f02 import (
    c2_next_probability,
    future_distribution,
    generate_world,
)
from grain_research.gptpro_foundation_v3.f02_recovery import (
    fit_observed_source,
    paired_bootstrap,
    selection_views,
    sequential_predictions,
)


def dump(path, value):
    """Write generated JSON, converting numeric model arrays explicitly."""
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.tolist()) + "\n")


def table(path, rows):
    """Write a nonempty generated table with a union schema."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def jsonl(path):
    """Read an explicitly selected observations file."""
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def source_phase(root, world):
    """Separate process: only source observations are read, never truth or eval."""
    path = root / "source.jsonl"
    model = fit_observed_source(jsonl(path), world)
    dump(root / "model.json", model)
    dump(
        root / "fit_manifest.json",
        {
            "source_sha256": sha256(path),
            "model_sha256": sha256(root / "model.json"),
            "input_files": ["source.jsonl"],
            "truth_access": False,
        },
    )


def predict_phase(root, world, prefix):
    """Separate process: score observed prefixes, never read the truth ledger."""
    fit = json.loads((root / "fit_manifest.json").read_text())
    if sha256(root / "model.json") != fit["model_sha256"]:
        raise ValueError("frozen model hash mismatch")
    model = json.loads((root / "model.json").read_text())
    for role in ("validation", "application"):
        records = jsonl(root / f"{role}.jsonl")
        if world == "c3":
            predictions = selection_views(
                records, (prefix, max(row["t"] for row in records) + 1)
            )
        else:
            predictions = sequential_predictions(records, model, prefix)
        (root / f"{role}_predictions.jsonl").write_text(
            "\n".join(json.dumps(row) for row in predictions) + "\n"
        )


def score_phase(root, world, seed):
    """Only this terminal scorer reads hidden targets for error calculation."""
    truth = {
        row["author_id"]: row for row in json.loads((root / "truth.json").read_text())
    }
    author_metrics = []
    for role in ("validation", "application"):
        predictions = jsonl(root / f"{role}_predictions.jsonl")
        groups = defaultdict(list)
        if world == "c3":
            for row in predictions:
                latent = truth[row["author_id"]]
                for target in ("pi", "q", "publication"):
                    estimate = row[f"{target}_hat"]
                    expected = (
                        latent["published_rate"]
                        if target == "publication"
                        else latent[target]
                    )
                    author_metrics.append(
                        {
                            "seed": seed,
                            "world": world,
                            "role": role,
                            "author_id": row["author_id"],
                            "candidate": row["view"],
                            "prefix": row["prefix"],
                            "metric": f"{target}_squared_error",
                            "value": (estimate - expected) ** 2
                            if estimate is not None
                            else None,
                            "missing": estimate is None,
                        }
                    )
                if row["view"] == "opportunity_masked":
                    author_metrics.append(
                        {
                            "seed": seed,
                            "world": world,
                            "role": role,
                            "author_id": row["author_id"],
                            "candidate": row["view"],
                            "prefix": row["prefix"],
                            "metric": "q_plugin_bound_contains_truth",
                            "value": float(
                                row["q_lower"] <= latent["q"] <= row["q_upper"]
                            ),
                            "scope": "sample_plugin_bound_not_a_confidence_interval",
                        }
                    )
        else:
            observations = jsonl(root / f"{role}.jsonl")
            history = defaultdict(list)
            for observation in observations:
                history[observation["author_id"]].append(observation)
            # Oracle is added only by the terminal scorer; never serialized into
            # the source fit or observation-only inference output.
            seen_events = set()
            oracle_rows = []
            for row in predictions:
                if row["event_id"] in seen_events:
                    continue
                seen_events.add(row["event_id"])
                author = row["author_id"]
                t = row["t"]
                if world == "c1":
                    z = history[author][t]["condition"]["z"]
                    p = 0.5 + 0.25 * truth[author]["kind"] * z
                else:
                    a, b = [r["observation"] for r in history[author][t - 2 : t]]
                    p = c2_next_probability(
                        a, b, 1, "xor" if world == "c2_xor" else "iid"
                    )
                oracle_rows.append({**row, "candidate": "oracle_scoring_only", "p": p})
            predictions.extend(oracle_rows)
            for row in predictions:
                groups[(row["author_id"], row["candidate"])].append(row)
            for (author, candidate), rows in groups.items():
                p = np.array([row["p"] for row in rows])
                y = np.array([row["y"] for row in rows])
                clipped = np.clip(p, 1e-6, 1 - 1e-6)
                for metric, value in [
                    ("brier", np.mean((p - y) ** 2)),
                    (
                        "log_loss",
                        -np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)),
                    ),
                    ("clip_count", np.sum(clipped != p)),
                    (
                        "boundary_prediction_count",
                        np.sum((p <= 1e-6) | (p >= 1 - 1e-6)),
                    ),
                ]:
                    author_metrics.append(
                        {
                            "seed": seed,
                            "world": world,
                            "role": role,
                            "author_id": author,
                            "candidate": candidate,
                            "metric": metric,
                            "value": float(value),
                            "events": len(rows),
                        }
                    )
    table(root / "author_metrics.csv", author_metrics)


def exact_certificates(out):
    """Enumerate horizon1..6 and verify next-state conditioning independently."""
    rows = []
    for world in ("xor", "iid"):
        for history in itertools.product((0, 1), repeat=2):
            for horizon in range(1, 7):
                law = future_distribution(history, horizon, world)
                assert abs(sum(law.values()) - 1) < 1e-12
                for first in (0, 1):
                    mass = sum(p for key, p in law.items() if key[0] == str(first))
                    conditional = {
                        key[1:]: p / mass
                        for key, p in law.items()
                        if key[0] == str(first)
                    }
                    updated = future_distribution(
                        (history[1], first), horizon - 1, world
                    )
                    error = max(abs(conditional[key] - updated[key]) for key in updated)
                    rows.append(
                        {
                            "world": world,
                            "history": str(history),
                            "horizon": horizon,
                            "first": first,
                            "closure_max_error": error,
                        }
                    )
                    assert error < 1e-12
    # 00 and 11 share next law in XOR, but not two-step futures.
    differences = []
    for horizon in range(1, 7):
        a = future_distribution((0, 0), horizon)
        b = future_distribution((1, 1), horizon)
        differences.append(
            {
                "horizon": horizon,
                "total_variation_00_11": 0.5 * sum(abs(a[k] - b[k]) for k in a),
            }
        )
    table(out / "state_update_closure.csv", rows)
    table(out / "future_equivalence.csv", differences)
    dump(
        out / "exact_identification.json",
        {
            "c3": [
                {"pi": 0.8, "q": 0.5, "published_rate": 0.4},
                {"pi": 0.4, "q": 1.0, "published_rate": 0.4},
            ],
            "q_population_bounds_given_published_rate": [0.4, 1.0],
            "epsilon_nontransitivity": {"points": [0, 0.75, 1.5], "epsilon": 1.0},
            "scope": "finite laws and conditioning, not psychological equivalence",
        },
    )


def full_run(args):
    """Run all worlds/seeds with source, prediction and truth scoring separated."""
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("old result revisions must be preserved")
    out.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    seeds = [20260915] if args.smoke else [20260915, 20260916, 20260917]
    counts = (
        {"source": 12, "validation": 6, "application": 6}
        if args.smoke
        else {"source": 300, "validation": 100, "application": 200}
    )
    events, prefix = (8, 4) if args.smoke else (48, 12)
    sources = [
        Path(__file__),
        ROOT / "grain_research/gptpro_foundation_v3/f02.py",
        ROOT / "grain_research/gptpro_foundation_v3/f02_recovery.py",
        ROOT / "grain_research/measurement_discovery/estimators.py",
        PARENT / "docs/GRAIN_GPTPRO_ATOMIC_PROTOCOLS_V3.md",
    ]
    dump(
        out / "input_manifest.json",
        {
            "files": [{"path": str(path), "sha256": sha256(path)} for path in sources],
            "purpose": "restore original frozen F02 battery; task acceptance requires separate review",
        },
    )
    archive = out / "source_archive"
    archive.mkdir()
    for path in sources:
        (archive / (sha256(path) + "_" + path.name)).write_bytes(path.read_bytes())
    dump(
        out / "protocol.json",
        {
            "seeds": seeds,
            "author_counts_per_world_seed": counts,
            "events": events,
            "prefix": prefix,
            "smoke": args.smoke,
            "worlds": ["c1", "c2_xor", "c2_iid", "c3"],
            "truth_process": "score only",
        },
    )
    exact_certificates(out)
    metric_rows = []
    for seed in seeds:
        for world in ("c1", "c2_xor", "c2_iid", "c3"):
            root = out / f"{world}_{seed}"
            root.mkdir()
            records, truth = generate_world(world, seed, events=events, counts=counts)
            splitmap = {row["author_id"]: row["role"] for row in truth}
            seen = set()
            for row in records:
                validate_record(row, splitmap, seen)
            for role in counts:
                (root / f"{role}.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in records if row["role"] == role)
                    + "\n"
                )
            dump(root / "truth.json", truth)
            for phase in ("source", "predict", "score"):
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--phase",
                    phase,
                    "--output",
                    str(root),
                    "--world",
                    world,
                    "--seed",
                    str(seed),
                    "--prefix",
                    str(prefix),
                ]
                result = subprocess.run(
                    command, cwd=ROOT, capture_output=True, text=True, check=False
                )
                (root / f"{phase}.log").write_text(result.stdout + result.stderr)
                if result.returncode:
                    raise RuntimeError(f"{world}/{seed}/{phase} failed; see saved log")
            with (root / "author_metrics.csv").open(newline="") as handle:
                metric_rows.extend(csv.DictReader(handle))
            dump(
                out / "checkpoint.json",
                {"last_completed": root.name, "complete": False},
            )
            print(
                json.dumps({"world": world, "seed": seed, "completed": True}),
                flush=True,
            )
    table(out / "author_metrics.csv", metric_rows)
    summaries = defaultdict(list)
    for row in metric_rows:
        summaries[
            (
                row["world"],
                row["seed"],
                row["role"],
                row["candidate"],
                row["metric"],
                row.get("prefix", ""),
            )
        ].append(row)
    table(
        out / "metrics.csv",
        [
            dict(
                zip(("world", "seed", "role", "candidate", "metric", "prefix"), key),
                mean=float(
                    np.mean([float(row["value"]) for row in rows if row["value"] != ""])
                )
                if any(row["value"] != "" for row in rows)
                else None,
                authors=len(rows),
                observed=sum(row["value"] != "" for row in rows),
            )
            for key, rows in summaries.items()
        ],
    )
    comparisons = []
    for world, reference, candidates in [
        ("c1", "source_condition", ["personal_condition", "personal_mean"]),
        ("c2_xor", "b1", ["b2", "m"]),
        ("c2_iid", "b1", ["b2", "m"]),
    ]:
        for metric in ("brier", "log_loss"):
            for candidate in candidates:
                seed_differences = {}
                for seed in seeds:
                    selected = [
                        r
                        for r in metric_rows
                        if r["world"] == world
                        and r["role"] == "application"
                        and int(r["seed"]) == seed
                        and r["metric"] == metric
                    ]
                    a = {
                        r["author_id"]: float(r["value"])
                        for r in selected
                        if r["candidate"] == candidate
                    }
                    b = {
                        r["author_id"]: float(r["value"])
                        for r in selected
                        if r["candidate"] == reference
                    }
                    if set(a) != set(b):
                        raise ValueError("paired author mismatch")
                    seed_differences[seed] = np.array([a[u] - b[u] for u in sorted(a)])
                comparisons.append(
                    {
                        "world": world,
                        "candidate": candidate,
                        "reference": reference,
                        "metric": metric,
                        **paired_bootstrap(seed_differences),
                    }
                )
    table(out / "paired_comparisons.csv", comparisons)
    dump(
        out / "summary.json",
        {
            "status": "NUMERICAL_BATTERY_EXECUTED_REVIEW_PENDING",
            "smoke": args.smoke,
            "world_seed_runs": len(seeds) * 4,
            "simulated_authors": len(seeds) * 4 * sum(counts.values()),
            "records": len(seeds) * 4 * sum(counts.values()) * events,
            "parent_card_promotion": False,
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    dump(
        out / "output_manifest.json",
        {
            "files": [
                {"path": str(path.relative_to(out)), "sha256": sha256(path)}
                for path in sorted(out.rglob("*"))
                if path.is_file()
            ]
        },
    )


def main():
    """Dispatch isolated phases or the finite recovery orchestrator."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--phase", choices=["source", "predict", "score"])
    parser.add_argument("--world")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prefix", type=int, default=12)
    args = parser.parse_args()
    if args.phase == "source":
        source_phase(args.output, args.world)
    elif args.phase == "predict":
        predict_phase(args.output, args.world, args.prefix)
    elif args.phase == "score":
        score_phase(args.output, args.world, args.seed)
    else:
        full_run(args)


if __name__ == "__main__":
    main()
