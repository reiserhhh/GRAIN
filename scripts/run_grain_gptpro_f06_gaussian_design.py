#!/usr/bin/env python3
"""Resumable full F06 B: matched reactive/nonreactive target-design policies."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f00 import sha256
from grain_research.gptpro_foundation_v3.f02_recovery import paired_bootstrap
from grain_research.gptpro_foundation_v3.f06_gaussian_design import (
    CANDIDATES,
    TARGETS,
    expected_variance_gain,
    fit_prefix,
    policy_action,
)

ARMS = [("fixed", "a"), ("random", "a"), ("coverage", "a")] + [
    ("scalar_information", t) for t in TARGETS
]


def dump(path, value):
    """Write a finite JSON snapshot."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_author(seed, role, role_index, ordinal, source, events, budgets):
    """One environment supplies observations to a truth-free prefix estimator."""
    author = (
        "u_" + hashlib.sha256(f"F06B/{seed}/{role}/{ordinal}".encode()).hexdigest()[:16]
    )
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, 6, 1, role_index, ordinal, 0])
    )
    a, b, rho, beta = (
        rng.normal(),
        rng.normal(0.5, 0.5),
        rng.uniform(0.2, 0.8),
        rng.normal(0.4, 0.2),
    )
    initial = np.random.default_rng(
        np.random.SeedSequence([seed, 6, 1, role_index, ordinal, 1])
    ).normal(0, 0.5 / np.sqrt(1 - rho**2))
    innovations = np.random.default_rng(
        np.random.SeedSequence([seed, 6, 1, role_index, ordinal, 2])
    ).normal(0, 0.5, events)
    measurement = np.random.default_rng(
        np.random.SeedSequence([seed, 6, 1, role_index, ordinal, 3])
    ).normal(0, 0.5, events)
    random_choices = np.random.default_rng(
        np.random.SeedSequence([seed, 6, 1, role_index, ordinal, 4])
    ).integers(0, 4, events)
    steps, metrics = [], []
    for reactivity, effect in (("zero", 0.0), ("distributed", beta)):
        for policy, target in ARMS:
            arm = policy if policy != "scalar_information" else f"scalar_{target}"
            prior = fit_prefix([], source)
            if policy == "scalar_information":
                gains = [
                    expected_variance_gain(prior, z, 0.0, target)["gain"]
                    for z in (-1.0, 1.0)
                ]
                z = (-1.0, 1.0)[
                    int(
                        np.flatnonzero(
                            np.isclose(gains, max(gains), atol=1e-14, rtol=0)
                        )[0]
                    )
                ]
            else:
                z = policy_action(
                    prior, policy, 0, random_index=int(random_choices[0])
                )[0]
            state = reference = initial
            previous_h = 0.0
            rows = []
            perturbation, actions, successes = 0.0, 0, 0
            for t in range(events):
                state = rho * state + effect * previous_h + innovations[t]
                reference = rho * reference + innovations[t]
                y = a + b * z + state + measurement[t]
                rows.append(
                    {
                        "author_id": author,
                        "event_id": f"{author}:{t}",
                        "role": role,
                        "t": t,
                        "z": float(z),
                        "y": float(y),
                        "h": 0,
                    }
                )
                posterior = fit_prefix(rows, source)
                successes += int(posterior["success"])
                candidate_gains = [
                    expected_variance_gain(posterior, cz, ch, target)["gain"]
                    for cz, ch in CANDIDATES
                ]
                if t + 1 < events:
                    next_z, selected_h = policy_action(
                        posterior, policy, t + 1, target, int(random_choices[t + 1])
                    )
                else:
                    next_z, selected_h = z, 0.0
                rows[-1]["h"] = int(selected_h)
                perturbation += (state - reference) ** 2
                actions += int(selected_h)
                step = {
                    "author_id": author,
                    "seed": seed,
                    "role": role,
                    "reactivity": reactivity,
                    "arm": arm,
                    "design_target": target if policy == "scalar_information" else None,
                    "t": t,
                    "observation": dict(rows[-1]),
                    "next_z": float(next_z),
                    "posterior_mean": posterior["mean"].tolist(),
                    "posterior_covariance": posterior["covariance"].tolist(),
                    "candidate_gains": candidate_gains,
                    "fit_success": posterior["success"],
                    "attempts": posterior["attempts"],
                    "beta_status": posterior["beta_status"],
                    "state_true": float(state),
                    "state_no_action": float(reference),
                    "delta_state": float(state - reference),
                }
                steps.append(step)
                if t + 1 in budgets:
                    for measured, index in TARGETS.items():
                        truth = {"a": a, "b": b, "beta": effect, "state": state}[
                            measured
                        ]
                        estimate = posterior["mean"][index]
                        variance = posterior["covariance"][index, index]
                        metrics.append(
                            {
                                "author_id": author,
                                "seed": seed,
                                "role": role,
                                "reactivity": reactivity,
                                "arm": arm,
                                "design_target": target
                                if policy == "scalar_information"
                                else None,
                                "target": measured,
                                "budget": t + 1,
                                "estimate": float(estimate),
                                "truth": float(truth),
                                "MSE": float((estimate - truth) ** 2),
                                "bias": float(estimate - truth),
                                "variance": float(variance),
                                "coverage": float(
                                    abs(estimate - truth)
                                    <= 1.959963984540054 * np.sqrt(max(variance, 0))
                                ),
                                "sum_squared_perturbation": float(perturbation),
                                "actions_selected": actions,
                                "effective_actions": sum(row["h"] for row in rows[:-1]),
                                "fit_success": float(posterior["success"]),
                                "prefix_fit_rate": successes / (t + 1),
                                "beta_status": posterior["beta_status"],
                                "rho_estimate": float(posterior["rho"]),
                                "rho_MSE": float((posterior["rho"] - rho) ** 2),
                            }
                        )
                previous_h, z = selected_h, next_z
    return {
        "author_id": author,
        "seed": seed,
        "role": role,
        "truth": {
            "a": float(a),
            "b": float(b),
            "rho": float(rho),
            "beta": float(beta),
            "initial": float(initial),
            "innovations": innovations.tolist(),
            "measurement": measurement.tolist(),
            "random_choices": random_choices.tolist(),
        },
        "steps": steps,
        "metrics": metrics,
    }


def summarize(output):
    """Aggregate only complete authored packets and report paired application loss."""
    metrics = []
    for path in sorted((output / "authors").glob("*.json.gz")):
        if path.name.startswith("._"):
            continue
        with gzip.open(path, "rt") as handle:
            metrics.extend(json.load(handle)["metrics"])
    data = pd.DataFrame(metrics)
    data.to_csv(output / "metrics.csv", index=False)
    application = data[data.role == "application"]
    if application.empty:
        return len(data)
    measures = [
        "MSE",
        "bias",
        "variance",
        "coverage",
        "sum_squared_perturbation",
        "effective_actions",
        "fit_success",
        "prefix_fit_rate",
        "rho_MSE",
    ]
    keys = ["reactivity", "budget", "target", "arm"]
    for group_keys, name in (
        (keys, "summary_by_policy.csv"),
        (["seed"] + keys, "summary_by_seed.csv"),
    ):
        table = application.groupby(group_keys)[measures].mean().reset_index()
        table["RMSE"] = np.sqrt(table.MSE)
        table.to_csv(output / name, index=False)
    contrasts = []
    for (reactivity, budget, target, arm), group in application[
        application.arm != "random"
    ].groupby(keys):
        reference = application[
            (application.reactivity == reactivity)
            & (application.budget == budget)
            & (application.target == target)
            & (application.arm == "random")
        ]
        pairs = group.merge(
            reference,
            on=["author_id", "seed"],
            suffixes=("_new", "_reference"),
            validate="one_to_one",
        )
        for metric in ("MSE", "sum_squared_perturbation"):
            delta = {
                seed: (g[metric + "_new"] - g[metric + "_reference"]).to_numpy()
                for seed, g in pairs.sort_values("author_id").groupby("seed")
            }
            contrasts.append(
                {
                    "reactivity": reactivity,
                    "budget": int(budget),
                    "target": target,
                    "arm": arm,
                    "metric": metric,
                    **paired_bootstrap(delta),
                }
            )
    pd.DataFrame(contrasts).to_csv(output / "paired_comparisons.csv", index=False)
    return len(data)


def main():
    """Resume by verified whole-author checkpoints, never by partial success counts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=1800)
    args = parser.parse_args()
    start = time.monotonic()
    counts = (
        {"source": 12, "validation": 6, "application": 6}
        if args.smoke
        else {"source": 300, "validation": 100, "application": 200}
    )
    seeds = [20260915] if args.smoke else [20260915, 20260916, 20260917]
    events = 8 if args.smoke else 24
    budgets = [4, 8] if args.smoke else [8, 16, 24]
    paths = [
        Path(__file__),
        ROOT / "grain_research/gptpro_foundation_v3/f06_gaussian_design.py",
        ROOT / "grain_research/gptpro_foundation_v3/f06_policy_state.py",
    ]
    fingerprint = {str(p.resolve()): sha256(p) for p in paths}
    source_manifest = args.source / "output_manifest.json"
    for item in json.loads(source_manifest.read_text())["files"]:
        assert sha256(args.source / item["path"]) == item["sha256"]
    protocol = {
        "counts": counts,
        "seeds": seeds,
        "events": events,
        "budgets": budgets,
        "source_manifest_hash": sha256(source_manifest),
        "source": str(args.source.resolve()),
        "code": fingerprint,
        "world": "F05 core parameter distribution with beta zero/distributed; matched independent exogenous streams",
        "approximation": "MAP-local inverse expected information for parameters, delta-method joint state covariance; not an exact Bayesian posterior",
        "source_learning": "reuse frozen F05 core source prior/noise, independent declared beta prior; no new source fitting or validation selection",
        "timing": "select next_z and h_t after y_t; initial h_-1=0; final h=0; state target is next state when designing",
        "policy": "four policy families, scalar information evaluated independently for all4 targets; fixed/random/coverage paths reused across target readouts",
        "tie": "lexical candidate order, atol1e-14",
        "checkpoint": "whole author plus hash, no partial author promoted",
    }
    if args.output.exists():
        if not args.resume:
            raise FileExistsError("use explicit resume; do not overwrite")
        assert json.loads((args.output / "protocol.json").read_text()) == protocol
        checkpoint = json.loads((args.output / "checkpoint.json").read_text())
    else:
        args.output.mkdir(parents=True)
        (args.output / "authors").mkdir()
        (args.output / "source_archive").mkdir()
        for path in paths:
            (args.output / "source_archive" / path.name).write_bytes(path.read_bytes())
        dump(args.output / "protocol.json", protocol)
        checkpoint = {"completed": {}, "status": "RUNNING"}
        dump(args.output / "checkpoint.json", checkpoint)
    for filename, digest in checkpoint["completed"].items():
        assert sha256(args.output / "authors" / filename) == digest
    total = sum(counts.values()) * len(seeds)
    for seed in seeds:
        source = json.loads(
            (args.source / f"core_{seed}" / "source_model.json").read_text()
        )
        for role_index, (role, count) in enumerate(counts.items()):
            for i in range(count):
                filename = f"{seed}_{role}_{i:04d}.json.gz"
                if filename in checkpoint["completed"]:
                    continue
                if time.monotonic() - start >= args.max_seconds:
                    checkpoint.update(
                        status="CHECKPOINT_TIME_LIMIT",
                        total_expected=total,
                        invocation_elapsed=time.monotonic() - start,
                    )
                    dump(args.output / "checkpoint.json", checkpoint)
                    print(
                        json.dumps(
                            {k: v for k, v in checkpoint.items() if k != "completed"}
                        ),
                        flush=True,
                    )
                    return
                packet = run_author(seed, role, role_index, i, source, events, budgets)
                temporary = args.output / "authors" / (filename + ".partial")
                with gzip.open(temporary, "wt") as handle:
                    json.dump(packet, handle, allow_nan=False)
                destination = args.output / "authors" / filename
                os.replace(temporary, destination)
                checkpoint["completed"][filename] = sha256(destination)
                dump(args.output / "checkpoint.json", checkpoint)
                if len(checkpoint["completed"]) % 10 == 0:
                    print(
                        json.dumps(
                            {
                                "completed_authors": len(checkpoint["completed"]),
                                "total": total,
                                "elapsed_seconds": time.monotonic() - start,
                            }
                        ),
                        flush=True,
                    )
    metric_rows = summarize(args.output)
    checkpoint.update(
        status="COMPUTED_AUDIT_PENDING",
        total_expected=total,
        metric_rows=metric_rows,
        invocation_elapsed=time.monotonic() - start,
    )
    dump(args.output / "checkpoint.json", checkpoint)
    dump(
        args.output / "output_manifest.json",
        {
            "files": [
                {"path": str(p.relative_to(args.output)), "sha256": sha256(p)}
                for p in sorted(args.output.rglob("*"))
                if p.is_file() and not p.name.startswith("._")
            ]
        },
    )
    print(
        json.dumps({k: v for k, v in checkpoint.items() if k != "completed"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
