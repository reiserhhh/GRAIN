#!/usr/bin/env python3
"""F05 source-calibrated joint/independent state filtering on paired episodes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "2")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f00 import sha256
from grain_research.gptpro_foundation_v3.f05 import eta_from_rho, grouped
from grain_research.gptpro_foundation_v3.f05_joint import (
    joint_filter,
    learn_common_variance,
)
from scripts.run_grain_gptpro_f04_application_recovery import dump, table


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def panel_predictions(people, parameters, q, r, common, budget, method, smoke=False):
    """Condition on visible simultaneous histories with frozen prefix parameters."""
    ids = sorted(people)
    end = budget if smoke or budget != 48 else 63
    y = np.array([[people[a][t]["y"] for a in ids] for t in range(end)])
    z = np.array([[people[a][t]["z"] for a in ids] for t in range(end)])
    psi = np.array([parameters[a] for a in ids])
    rho = 0.99 * np.tanh(psi[:, 2])
    results = []
    for arm, shared in (("independent", 0.0), ("joint", common)):
        filtered = joint_filter(y, z, psi, q, r, shared)
        for t in [budget - 1] if end == budget else range(47, end):
            for i, a in enumerate(ids):
                state = float(filtered["state_mean"][t, i])
                row = {
                    "author_id": a,
                    "t": t,
                    "parameter_prefix": budget,
                    "method": method,
                    "arm": arm,
                    "state_mean": state,
                    "state_variance": float(filtered["state_variance"][t, i]),
                }
                for k in (1, 4):
                    if t + k >= len(people[a]):
                        continue
                    condition = -1.0 if ((t + k) // 4) % 2 == 0 else 1.0
                    row[f"forecast{k}"] = float(
                        psi[i, 0] + psi[i, 1] * condition + rho[i] ** k * state
                    )
                results.append(row)
    return results


def main():
    """Worker sees source/application observations; oracle computed only afterward."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--application", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.worker:
        model = json.loads((args.source / "source_model.json").read_text())
        moment = learn_common_variance(read_rows(args.source / "source.jsonl"), model)
        dump(args.output / "source_common_moment.json", moment)
        people = grouped(read_rows(args.source / "application.jsonl"))
        previous = json.loads(
            (args.application / "observed_predictions.json").read_text()
        )
        predictions = []
        for budget in [4] if args.smoke else [16, 32, 48]:
            parameters = {
                p["author_id"]: p["psi"]
                for p in previous
                if p["method"] == "personal_MAP" and p["n"] == budget
            }
            assert set(parameters) == set(people)
            predictions.extend(
                panel_predictions(
                    people,
                    parameters,
                    moment["total_variance"],
                    float(np.exp(2 * model["log_sigma"][1])),
                    moment["common_variance"],
                    budget,
                    "personal_MAP",
                    args.smoke,
                )
            )
        dump(args.output / "observed_predictions.json", predictions)
        return
    if args.output.exists():
        raise FileExistsError("preserve earlier runs")
    args.output.mkdir(parents=True)
    for directory in (args.source, args.application):
        for entry in json.loads((directory / "output_manifest.json").read_text())[
            "files"
        ]:
            assert sha256(directory / entry["path"]) == entry["sha256"]
    source = json.loads((args.source / "summary.json").read_text())
    if bool(source["smoke"]) != args.smoke:
        raise ValueError("source mode mismatch")
    inputs = json.loads((args.application / "input_manifest.json").read_text())
    assert inputs["source_manifest_sha256"] == sha256(
        args.source / "output_manifest.json"
    )
    paths = [Path(__file__), ROOT / "grain_research/gptpro_foundation_v3/f05_joint.py"]
    (args.output / "source_archive").mkdir()
    for path in paths:
        (args.output / "source_archive" / (sha256(path) + "_" + path.name)).write_bytes(
            path.read_bytes()
        )
    dump(
        args.output / "input_manifest.json",
        {
            "source_sha256": sha256(args.source / "output_manifest.json"),
            "application_sha256": sha256(args.application / "output_manifest.json"),
        },
    )
    scores = []
    moments = []
    for outcome in source["outcomes"]:
        world, seed = outcome["world"], outcome["seed"]
        root = args.source / f"{world}_{seed}"
        app = args.application / f"{world}_{seed}"
        out = args.output / f"{world}_{seed}"
        out.mkdir()
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--source",
            str(root.resolve()),
            "--application",
            str(app.resolve()),
            "--output",
            str(out.resolve()),
            "--worker",
        ]
        if args.smoke:
            command.append("--smoke")
        with (out / "worker.log").open("w") as log:
            subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        predictions = json.loads((out / "observed_predictions.json").read_text())
        moment = json.loads((out / "source_common_moment.json").read_text())
        moments.append({"world": world, "seed": seed, **moment})
        people = grouped(read_rows(root / "application.jsonl"))
        truth = {
            p["author_id"]: p for p in json.loads((root / "truth.json").read_text())
        }
        parameters = {
            a: [truth[a]["a"], truth[a]["b"], eta_from_rho(truth[a]["rho"])]
            for a in people
        }
        for budget in [4] if args.smoke else [16, 32, 48]:
            predictions.extend(
                panel_predictions(
                    people,
                    parameters,
                    0.25,
                    0.25,
                    0.125 if world == "shared_shock" else 0.0,
                    budget,
                    "oracle",
                    args.smoke,
                )
            )
        for row in predictions:
            a, t = row["author_id"], row["t"]
            target = truth[a]["states"][t]
            row.update(
                world=world,
                seed=seed,
                state_truth=target,
                state_MSE=(row["state_mean"] - target) ** 2,
                state_coverage=float(
                    abs(row["state_mean"] - target)
                    <= 1.959963984540054 * np.sqrt(row["state_variance"])
                ),
            )
            for k in (1, 4):
                if f"forecast{k}" in row:
                    future = people[a][t + k]
                    conditional = (
                        truth[a]["a"]
                        + truth[a]["b"] * future["z"]
                        + truth[a]["rho"] ** k * target
                    )
                    row[f"forecast{k}_MSE"] = (row[f"forecast{k}"] - future["y"]) ** 2
                    row[f"conditional{k}_MSE"] = (
                        row[f"forecast{k}"] - conditional
                    ) ** 2
            scores.append(row)
        dump(
            args.output / "checkpoint.json",
            {"last_world": world, "last_seed": seed, "rows": len(scores)},
        )
        print(
            json.dumps(
                {
                    "world": world,
                    "seed": seed,
                    "rows": len(scores),
                    "source_common": moment["common_variance"],
                }
            ),
            flush=True,
        )
    table(args.output / "joint_state_metrics.csv", scores)
    table(args.output / "source_common_moments.csv", moments)
    data = pd.DataFrame(scores)
    metrics = [
        "state_MSE",
        "state_coverage",
        "forecast1_MSE",
        "forecast4_MSE",
        "conditional1_MSE",
        "conditional4_MSE",
    ]
    data.groupby(["world", "seed", "t", "parameter_prefix", "method", "arm"])[
        metrics
    ].mean().reset_index().to_csv(args.output / "episode_summary.csv", index=False)
    data.groupby(["world", "t", "parameter_prefix", "method", "arm"])[
        metrics
    ].mean().reset_index().to_csv(args.output / "joint_summary.csv", index=False)
    dump(
        args.output / "summary.json",
        {
            "status": "JOINT_CHANNEL_COMPUTED_AUDIT_PENDING",
            "rows": len(scores),
            "source_dynamics_refitted": False,
            "new_source_estimate": "off-diagonal innovation covariance, raw/clipped stored",
            "personal_parameters": "reused matched existing prefix estimates",
            "uncertainty": "three independent episodes per world; no author-bootstrap population CI",
            "remaining": "full numerical replay, marginal equivalence replay, original task packaging",
        },
    )
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


if __name__ == "__main__":
    main()
