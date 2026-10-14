#!/usr/bin/env python3
"""Three-level total covariance from complete source-bootstrap cells only."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from grain_research.gptpro_foundation_v3.f13_uncertainty import validate_covariance
from scripts.audit_grain_gptpro_f13_nested import wilson


def combine(means, events, parameters):
    """Preserve every covariance and separate source, parameter and event levels."""
    means = np.asarray(means, float)
    events = np.asarray(events, float)
    parameters = np.asarray(parameters, float)
    if (
        means.ndim != 2
        or events.shape != (len(means), means.shape[1], means.shape[1])
        or parameters.shape != events.shape
        or not len(means)
        or not np.isfinite(means).all()
    ):
        raise ValueError("aligned complete finite mixture required")
    for c in np.concatenate([events, parameters]):
        validate_covariance(c)
    center = means.mean(axis=0)
    source = (means - center).T @ (means - center) / len(means)
    event, parameter = events.mean(axis=0), parameters.mean(axis=0)
    return (
        center,
        event,
        parameter,
        source,
        validate_covariance(event + parameter + source),
    )


def table(path, rows, fields=None):
    """Write scalar tables, preserving empty tables with their schema."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """No partial bootstrap renormalization, no outcomes used for model selection."""
    p = argparse.ArgumentParser()
    for name in ("input", "source", "output"):
        p.add_argument(f"--{name}", required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        "worlds": ["core", "near_root", "shared_shock"],
        "seeds": [20260915, 20260916, 20260917],
        "outer": 200,
        "rule": "skip incomplete cells; any unsupported source contribution makes that author/budget unscored; no conditional renormalization",
    }
    (args.output / "protocol_before_run.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    predictions, metrics, manifests, statuses, unsupported = [], [], [], [], []
    for world in config["worlds"]:
        for seed in config["seeds"]:
            root = args.input / f"{world}_{seed}"
            paths = [root / f"b{b:03d}" / "personal_mixtures.json" for b in range(200)]
            if not all((path.parent / "summary.json").exists() for path in paths):
                statuses.append(
                    {
                        "world": world,
                        "seed": seed,
                        "status": "INCOMPLETE",
                        "completed": sum(
                            (path.parent / "summary.json").exists() for path in paths
                        ),
                    }
                )
                continue
            bundles = []
            for b, path in enumerate(paths):
                summary = json.loads((path.parent / "summary.json").read_text())
                if (
                    summary["status"]
                    != "PERSONAL_PARAMETER_MIXTURES_COMPLETE_SOURCE_AGGREGATION_PENDING"
                ):
                    raise ValueError(
                        f"unsupported source task requires explicit cell-level report: {path}"
                    )
                if (
                    summary["source_bootstrap"] != b
                    or summary["seed"] != seed
                    or summary["world"] != world
                ):
                    raise ValueError("source bootstrap identity mismatch")
                raw = path.read_bytes()
                manifests.append(
                    {
                        "path": str(path.resolve()),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
                rows = json.loads(raw)
                indexed = {(r["author_id"], r["budget"]): r for r in rows}
                if len(indexed) != 600 or len(rows) != 600:
                    raise ValueError("incomplete/duplicate personal grid")
                if bundles and set(indexed) != set(bundles[0]):
                    raise ValueError("author/budget identity changed")
                bundles.append(indexed)
            data_root = args.source / f"{world}_{seed}"
            truth = {
                r["author_id"]: r
                for r in json.loads((data_root / "truth.json").read_text())
            }
            observed = {}
            for line in (data_root / "application.jsonl").read_text().splitlines():
                row = json.loads(line)
                observed.setdefault(row["author_id"], {})[row["t"]] = row["y"]
            covariance_records = []
            for author, budget in sorted(bundles[0]):
                selected = [bundle[(author, budget)] for bundle in bundles]
                failed = [b for b, r in enumerate(selected) if r["mixture"] is None]
                if failed:
                    unsupported.append(
                        {
                            "world": world,
                            "seed": seed,
                            "author_id": author,
                            "budget": budget,
                            "failed_bootstraps": json.dumps(failed),
                        }
                    )
                    continue
                pieces = [r["mixture"] for r in selected]
                m, e, pvar, s, total = combine(
                    [x["mean"] for x in pieces],
                    [x["event_state"] for x in pieces],
                    [x["parameter"] for x in pieces],
                )
                direct = np.mean(
                    [
                        np.array(x["total"]) + np.outer(x["mean"], x["mean"])
                        for x in pieces
                    ],
                    axis=0,
                ) - np.outer(m, m)
                np.testing.assert_allclose(total, direct, atol=1e-10, rtol=1e-8)
                covariance_records.append(
                    {
                        "author_id": author,
                        "budget": budget,
                        "mean": m.tolist(),
                        "event_state": e.tolist(),
                        "personal_parameter": pvar.tolist(),
                        "source": s.tolist(),
                        "total": total.tolist(),
                    }
                )
                truths = [
                    truth[author]["states"][budget - 1],
                    observed[author][budget + 3],
                    np.mean([observed[author][t] for t in range(budget, budget + 4)]),
                ]
                for i, target in enumerate(
                    (
                        "current_state",
                        "future_observation_k4",
                        "future_observation_average4",
                    )
                ):
                    predictions.append(
                        {
                            "world": world,
                            "seed": seed,
                            "author_id": author,
                            "budget": budget,
                            "target": target,
                            "mean": m[i],
                            "variance": total[i, i],
                            "event_state_variance": e[i, i],
                            "parameter_variance": pvar[i, i],
                            "source_variance": s[i, i],
                            "truth": truths[i],
                        }
                    )
            (args.output / f"covariances_{world}_{seed}.json").write_text(
                json.dumps(covariance_records) + "\n"
            )
            statuses.append(
                {
                    "world": world,
                    "seed": seed,
                    "status": "AGGREGATED",
                    "completed": 200,
                    "scored_prefixes": len(covariance_records),
                }
            )
    groups = {}
    for row in predictions:
        key = tuple(row[k] for k in ("world", "seed", "budget", "target"))
        groups.setdefault(key, []).append(row)
    for key, rows in groups.items():
        m, v, y = (
            np.array([r[k] for r in rows]) for k in ("mean", "variance", "truth")
        )
        ok = abs(m - y) <= 1.96 * np.sqrt(v)
        low, high = wilson(int(ok.sum()), len(rows))
        decision = (
            "DESCRIPTIVE_DEPENDENT_AUTHORS"
            if key[0] == "shared_shock"
            else "INCOMPLETE_SUPPORT"
            if len(rows) != 200
            else "PASS"
            if low <= 0.95 <= high
            else "FAIL"
        )
        metrics.append(
            dict(zip(("world", "seed", "budget", "target"), key, strict=True))
            | {
                "n": len(rows),
                "mse": float(np.mean((m - y) ** 2)),
                "coverage": float(ok.mean()),
                "mean_width": float(np.mean(3.92 * np.sqrt(v))),
                "wilson_low": low,
                "wilson_high": high,
                "decision": decision,
            }
        )
    if predictions:
        table(args.output / "predictions.csv", predictions)
        table(args.output / "metrics.csv", metrics)
    table(
        args.output / "unsupported.csv",
        unsupported,
        ["world", "seed", "author_id", "budget", "failed_bootstraps"],
    )
    (args.output / "input_manifest.json").write_text(json.dumps(manifests) + "\n")
    summary = {
        "status": "AVAILABLE_COMPLETE_CELLS_AGGREGATED_AUDIT_PENDING",
        "cells": statuses,
        "prediction_rows": len(predictions),
        "unsupported_prefixes": len(unsupported),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
