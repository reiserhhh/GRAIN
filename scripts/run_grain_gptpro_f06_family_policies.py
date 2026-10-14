#!/usr/bin/env python3
"""Run all five frozen F06 panel-A policies on matched finite environments."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f00 import sha256
from grain_research.gptpro_foundation_v3.f02_recovery import paired_bootstrap
from grain_research.gptpro_foundation_v3.f06_family_policies import (
    HYPOTHESES,
    LAWS,
    POLICIES,
    entropy,
    lexical_maximum,
    run_policy,
)
from scripts.run_grain_gptpro_f04_application_recovery import dump, table


def main():
    """Environment truth scores fixed policies, never informs their actions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous runs")
    args.output.mkdir(parents=True)
    counts = (
        {"source": 12, "validation": 6, "application": 6}
        if args.smoke
        else {"source": 300, "validation": 100, "application": 200}
    )
    seeds = [20260915] if args.smoke else [20260915, 20260916, 20260917]
    budgets = [4, 8] if args.smoke else [8, 16, 24]
    metrics = []
    truths = []
    action_rows = 0
    replay_error = 0.0
    with (
        gzip.open(args.output / "trajectories.jsonl.gz", "wt") as trajectory,
        gzip.open(args.output / "exogenous.jsonl.gz", "wt") as exogenous,
    ):
        for seed in seeds:
            for role_index, (role, count) in enumerate(counts.items()):
                assigned = np.arange(count) % 9
                np.random.default_rng(
                    np.random.SeedSequence([seed, 6, 0, role_index, 0])
                ).shuffle(assigned)
                for i, atom in enumerate(assigned):
                    author = (
                        "u_"
                        + hashlib.sha256(
                            f"{seed}/F06A/{role}/{i}".encode()
                        ).hexdigest()[:16]
                    )
                    rng = np.random.default_rng(
                        np.random.SeedSequence([seed, 6, 0, role_index, i, 1])
                    )
                    uniforms = rng.random(max(budgets))
                    random_actions = np.random.default_rng(
                        np.random.SeedSequence([seed, 6, 0, role_index, i, 2])
                    ).integers(0, 8, max(budgets))
                    future_u = np.random.default_rng(
                        np.random.SeedSequence([seed, 6, 0, role_index, i, 3])
                    ).random(8)
                    exogenous.write(
                        json.dumps(
                            {
                                "author_id": author,
                                "seed": seed,
                                "role": role,
                                "uniforms": uniforms.tolist(),
                                "random_actions": random_actions.tolist(),
                                "future_uniforms": future_u.tolist(),
                            }
                        )
                        + "\n"
                    )
                    true_p = LAWS[:, atom]
                    future_y = (future_u < true_p).astype(float)
                    truths.append(
                        {
                            "author_id": author,
                            "seed": seed,
                            "role": role,
                            "atom": int(atom),
                            "axis": HYPOTHESES[atom][0],
                            "beta": HYPOTHESES[atom][1],
                        }
                    )
                    for policy in POLICIES:
                        path = run_policy(int(atom), uniforms, random_actions, policy)
                        likelihood = np.ones(9) / 9
                        for row in path:
                            likelihood *= (
                                LAWS[row["action"]]
                                if row["response"]
                                else 1 - LAWS[row["action"]]
                            )
                            direct = likelihood / likelihood.sum()
                            replay_error = max(
                                replay_error,
                                float(np.max(np.abs(direct - row["posterior"]))),
                            )
                            trajectory.write(
                                json.dumps(
                                    {
                                        "author_id": author,
                                        "seed": seed,
                                        "role": role,
                                        "policy": policy,
                                        **row,
                                    }
                                )
                                + "\n"
                            )
                            action_rows += 1
                            if row["step"] not in budgets:
                                continue
                            weights = np.asarray(row["posterior"])
                            prediction = LAWS @ weights
                            family = weights.reshape(3, 3).sum(axis=1)
                            metrics.append(
                                {
                                    "author_id": author,
                                    "seed": seed,
                                    "role": role,
                                    "policy": policy,
                                    "budget": row["step"],
                                    "entropy": float(entropy(weights)),
                                    "atom_accuracy": float(
                                        lexical_maximum(weights) == atom
                                    ),
                                    "family_accuracy": float(
                                        lexical_maximum(family) == HYPOTHESES[atom][0]
                                    ),
                                    "probability_MSE": float(
                                        np.mean((prediction - true_p) ** 2)
                                    ),
                                    "expected_Brier": float(
                                        np.mean(
                                            (prediction - true_p) ** 2
                                            + true_p * (1 - true_p)
                                        )
                                    ),
                                    "future_Brier": float(
                                        np.mean((prediction - future_y) ** 2)
                                    ),
                                    "distinct_corners": len(
                                        {p["action"] for p in path[: row["step"]]}
                                    ),
                                }
                            )
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "metric_rows": len(metrics),
                        "trajectory_rows": action_rows,
                    }
                ),
                flush=True,
            )
    assert replay_error < 1e-12
    dump(args.output / "truth.json", truths)
    table(args.output / "policy_metrics.csv", metrics)
    data = pd.DataFrame(metrics)
    selected = data[data.role == "application"]
    measures = [
        "entropy",
        "atom_accuracy",
        "family_accuracy",
        "probability_MSE",
        "expected_Brier",
        "future_Brier",
        "distinct_corners",
    ]
    selected.groupby(["budget", "policy"])[measures].mean().reset_index().to_csv(
        args.output / "policy_summary.csv", index=False
    )
    selected.groupby(["seed", "budget", "policy"])[
        measures
    ].mean().reset_index().to_csv(args.output / "policy_by_seed.csv", index=False)
    contrasts = []
    for (budget, policy), group in selected[selected.policy != "random"].groupby(
        ["budget", "policy"]
    ):
        for metric in ("entropy", "probability_MSE", "expected_Brier", "future_Brier"):
            reference = selected[
                (selected.budget == budget) & (selected.policy == "random")
            ]
            joined = group.merge(
                reference,
                on=["author_id", "seed", "budget"],
                suffixes=("_new", "_reference"),
                validate="one_to_one",
            )
            differences = {
                seed: (g[metric + "_new"] - g[metric + "_reference"]).to_numpy()
                for seed, g in joined.sort_values("author_id").groupby("seed")
            }
            contrasts.append(
                {
                    "budget": int(budget),
                    "policy": policy,
                    "metric": metric,
                    **paired_bootstrap(differences),
                }
            )
    table(args.output / "paired_comparisons.csv", contrasts)
    dump(
        args.output / "summary.json",
        {
            "status": "PANEL_A_EXECUTED_OTHER_F06_PANELS_OPEN",
            "smoke": args.smoke,
            "counts_per_seed": counts,
            "metric_rows": len(metrics),
            "trajectory_rows": action_rows,
            "posterior_replay_max_error": replay_error,
            "prior": "uniform nine atoms, exact declared laws; no source parameter fitting needed",
            "remaining": [
                "independent policy/trajectory audit",
                "panelB Gaussian intervention",
                "IRT scalar comparator",
                "hidden-policy likelihood and sensitivity",
                "task contracts",
            ],
        },
    )
    paths = [
        Path(__file__),
        ROOT / "grain_research/gptpro_foundation_v3/f06_family_policies.py",
    ]
    (args.output / "source_archive").mkdir()
    for path in paths:
        (args.output / "source_archive" / (sha256(path) + "_" + path.name)).write_bytes(
            path.read_bytes()
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
