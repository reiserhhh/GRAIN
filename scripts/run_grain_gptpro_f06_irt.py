#!/usr/bin/env python3
"""Execute the separate frozen-grid scalar CAT reference at full role counts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f00 import sha256, validate_record
from grain_research.gptpro_foundation_v3.f02_recovery import paired_bootstrap
from grain_research.gptpro_foundation_v3.f06_irt import (
    DIFFICULTIES,
    GRID,
    LAWS,
    POLICIES,
    PRIOR,
    IRTLedger,
    choose,
)


def dump(path, value):
    """Write finite JSON without permitting silent NaN values."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    """Environment emits matched responses; selectors never receive truth."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve earlier runs")
    args.output.mkdir(parents=True)
    start = time.monotonic()
    counts = (
        {"source": 12, "validation": 6, "application": 6}
        if args.smoke
        else {"source": 300, "validation": 100, "application": 200}
    )
    seeds = [20260915] if args.smoke else [20260915, 20260916, 20260917]
    budgets = [4, 8] if args.smoke else [8, 16, 24]
    dump(
        args.output / "protocol.json",
        {
            "status": "FROZEN_BEFORE_OUTCOMES",
            "panel": "scalar IRT, not family or Gaussian state",
            "grid": GRID.tolist(),
            "difficulties": DIFFICULTIES.tolist(),
            "counts": counts,
            "seeds": seeds,
            "budgets": budgets,
            "policies": list(POLICIES),
            "prior": PRIOR.tolist(),
            "completion_of_unspecified_details": "Original protocol fixes grid/law/difficulties but not prior or fixed item. Declare discretized standard-normal prior, matched truth from same finite prior, fixed d=0, MAP Fisher and posterior-integrated EIG. No data-fitted item discrimination or calibration.",
            "selection": "truth-free selector/update interfaces; latent values only in simulated environment and terminal scorer",
            "interval": "equal-tail finite-grid posterior; coverage is prior-predictive simulation coverage",
            "common_events": "IDs include policy branch; repeated difficulty is a new independent event, not duplicate evidence",
        },
    )
    (args.output / "source_archive").mkdir()
    for path in (
        Path(__file__),
        ROOT / "grain_research/gptpro_foundation_v3/f06_irt.py",
        ROOT / "tests/test_grain_gptpro_f06_irt.py",
    ):
        (args.output / "source_archive" / path.name).write_bytes(path.read_bytes())
    dump(
        args.output / "input_manifest.json",
        {
            "files": [
                {"path": str(p.relative_to(args.output)), "sha256": sha256(p)}
                for p in sorted(args.output.rglob("*"))
                if p.is_file() and not p.name.startswith("._")
            ]
        },
    )
    metrics, truths = [], []
    steps, seen, splitmap = 0, set(), {}
    with (
        gzip.open(args.output / "trajectories.jsonl.gz", "wt") as trajectories,
        gzip.open(args.output / "common_records.jsonl.gz", "wt") as common,
        gzip.open(args.output / "exogenous.jsonl.gz", "wt") as external,
        gzip.open(args.output / "budget_posteriors.jsonl.gz", "wt") as posteriors,
    ):
        for seed in seeds:
            for role_index, (role, count) in enumerate(counts.items()):
                for i in range(count):
                    author = (
                        "u_"
                        + hashlib.sha256(
                            f"F06IRT/{seed}/{role}/{i}".encode()
                        ).hexdigest()[:16]
                    )
                    splitmap[author] = role
                    atom = int(
                        np.random.default_rng(
                            np.random.SeedSequence([seed, 6, 2, role_index, i, 0])
                        ).choice(len(GRID), p=PRIOR)
                    )
                    uniforms = np.random.default_rng(
                        np.random.SeedSequence([seed, 6, 2, role_index, i, 1])
                    ).random(max(budgets))
                    actions = np.random.default_rng(
                        np.random.SeedSequence([seed, 6, 2, role_index, i, 2])
                    ).integers(0, 3, max(budgets))
                    future_u = np.random.default_rng(
                        np.random.SeedSequence([seed, 6, 2, role_index, i, 3])
                    ).random(3)
                    external.write(
                        json.dumps(
                            {
                                "author_id": author,
                                "seed": seed,
                                "role": role,
                                "uniforms": uniforms.tolist(),
                                "random_actions": actions.tolist(),
                                "future_uniforms": future_u.tolist(),
                            }
                        )
                        + "\n"
                    )
                    truths.append(
                        {
                            "author_id": author,
                            "seed": seed,
                            "role": role,
                            "atom": atom,
                            "a": float(GRID[atom]),
                        }
                    )
                    for policy in POLICIES:
                        ledger = IRTLedger()
                        previous_entropy = float(-np.sum(PRIOR * np.log(PRIOR)))
                        visited = set()
                        for t, uniform in enumerate(uniforms):
                            action = choose(ledger.weights, policy, t, int(actions[t]))
                            response = int(uniform < LAWS[action, atom])
                            event_id = f"{author}:{policy}:{t}"
                            summary = ledger.append(event_id, action, response)
                            visited.add(action)
                            record = {
                                "author_id": author,
                                "event_id": event_id,
                                "episode_id": f"{author}:{policy}",
                                "t": t,
                                "condition": {
                                    "difficulty": float(DIFFICULTIES[action])
                                },
                                "opportunity": True,
                                "opportunity_status": "observed_independent_binary_trial",
                                "selected": True,
                                "observation": response,
                                "text_ref": "synthetic:no_text/irt_trial",
                                "reader_id": "f06_exact_irt_grid",
                                "role": role,
                            }
                            validate_record(record, splitmap, seen)
                            common.write(json.dumps(record) + "\n")
                            trajectories.write(
                                json.dumps(
                                    {
                                        "author_id": author,
                                        "seed": seed,
                                        "role": role,
                                        "policy": policy,
                                        "t": t,
                                        "action": action,
                                        "response": response,
                                        **summary,
                                        "entropy_increase": summary["entropy"]
                                        > previous_entropy,
                                    }
                                )
                                + "\n"
                            )
                            previous_entropy = summary["entropy"]
                            steps += 1
                            if t + 1 not in budgets:
                                continue
                            prediction = LAWS @ ledger.weights
                            p = LAWS[:, atom]
                            future_y = future_u < p
                            metrics.append(
                                {
                                    "author_id": author,
                                    "seed": seed,
                                    "role": role,
                                    "policy": policy,
                                    "budget": t + 1,
                                    **summary,
                                    "a_MSE": (summary["mean"] - GRID[atom]) ** 2,
                                    "a_error": summary["mean"] - GRID[atom],
                                    "coverage": float(
                                        summary["ci_low"]
                                        <= GRID[atom]
                                        <= summary["ci_high"]
                                    ),
                                    "probability_MSE": float(
                                        np.mean((prediction - p) ** 2)
                                    ),
                                    "expected_Brier": float(
                                        np.mean((prediction - p) ** 2 + p * (1 - p))
                                    ),
                                    "future_Brier": float(
                                        np.mean((prediction - future_y) ** 2)
                                    ),
                                    "distinct_items": len(visited),
                                }
                            )
                            posteriors.write(
                                json.dumps(
                                    {
                                        "author_id": author,
                                        "policy": policy,
                                        "budget": t + 1,
                                        "weights": ledger.weights.tolist(),
                                    }
                                )
                                + "\n"
                            )
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "steps": steps,
                        "elapsed_seconds": time.monotonic() - start,
                    }
                ),
                flush=True,
            )
    dump(args.output / "truth.json", truths)
    dump(args.output / "splitmap.json", splitmap)
    data = pd.DataFrame(metrics)
    data.to_csv(args.output / "metrics.csv", index=False)
    selected = data[data.role == "application"]
    measures = [
        "a_MSE",
        "a_error",
        "variance",
        "entropy",
        "coverage",
        "probability_MSE",
        "expected_Brier",
        "future_Brier",
        "distinct_items",
        "boundary_mass",
    ]
    for group_keys, name in (
        (["budget", "policy"], "summary_by_policy.csv"),
        (["seed", "budget", "policy"], "summary_by_seed.csv"),
    ):
        table = selected.groupby(group_keys)[measures].mean().reset_index()
        table["a_RMSE"] = np.sqrt(table.a_MSE)
        table.to_csv(args.output / name, index=False)
    contrasts = []
    for reference in ("random", "fixed_center"):
        for (budget, policy), group in selected[selected.policy != reference].groupby(
            ["budget", "policy"]
        ):
            baseline = selected[
                (selected.policy == reference) & (selected.budget == budget)
            ]
            joined = group.merge(
                baseline,
                on=["author_id", "seed", "budget"],
                suffixes=("_new", "_reference"),
                validate="one_to_one",
            )
            for metric in ("a_MSE", "probability_MSE", "future_Brier", "variance"):
                differences = {
                    seed: (g[metric + "_new"] - g[metric + "_reference"]).to_numpy()
                    for seed, g in joined.sort_values("author_id").groupby("seed")
                }
                contrasts.append(
                    {
                        "budget": int(budget),
                        "policy": policy,
                        "reference": reference,
                        "metric": metric,
                        **paired_bootstrap(differences),
                    }
                )
    pd.DataFrame(contrasts).to_csv(args.output / "paired_comparisons.csv", index=False)
    dump(
        args.output / "summary.json",
        {
            "status": "SCALAR_IRT_FULL_GRID_EXECUTED_F06_OPEN",
            "smoke": args.smoke,
            "authors": len(truths),
            "steps": steps,
            "metrics": len(data),
            "paired_intervals": len(contrasts),
            "common_records_validated": len(seen),
            "elapsed_seconds": time.monotonic() - start,
            "remaining": [
                "independent numerical audit",
                "full Gaussian four-policy B panel",
                "original task DAG/contracts",
            ],
            "scope": "known finite-grid IRT law; no text or personality validity measured",
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
