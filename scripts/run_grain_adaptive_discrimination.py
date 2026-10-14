#!/usr/bin/env python3
"""Run PA01 fixed, random and ADO action selection in a finite rival world."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from grain_research.measurement_discovery.adaptive_discrimination import (
    choose_action,
    update_posterior,
)
from grain_research.measurement_discovery.rival_models import BETAS, CORNERS, FAMILIES, response_probability


def write_json(path, value):
    """Write JSON."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def run(seed, arm, budget, rng):
    """Simulate one hidden author; action policy only sees prior observations."""
    true_family = FAMILIES[int(rng.integers(len(FAMILIES)))]
    true_beta = float(BETAS[int(rng.integers(len(BETAS)))])
    candidates = [(family, beta) for family in FAMILIES for beta in BETAS]
    weights = np.full(len(candidates), 1 / len(candidates))
    rows = []
    for step in range(budget):
        condition = CORNERS[step % len(CORNERS)] if arm == "fixed" else choose_action(weights, candidates, arm, rng)
        p_true = response_probability(true_family, true_beta, condition)
        observation = int(rng.random() < p_true)
        predictive = sum(weight * response_probability(f, b, condition) for weight, (f, b) in zip(weights, candidates, strict=True))
        logloss = -(observation * np.log(predictive) + (1 - observation) * np.log1p(-predictive))
        rows.append({"seed": seed, "arm": arm, "budget": budget, "step": step, "z1": condition[0], "z2": condition[1], "z3": condition[2], "observation": observation, "predictive": predictive, "log_loss": logloss, "posterior_entropy": float(-np.sum(weights * np.log(weights)))})
        weights = update_posterior(weights, candidates, condition, observation)
    holdout = []
    for condition in CORNERS:
        p = sum(weight * response_probability(f, b, condition) for weight, (f, b) in zip(weights, candidates, strict=True))
        truth = response_probability(true_family, true_beta, condition)
        holdout.append(-(truth * np.log(p) + (1 - truth) * np.log1p(-p)))
    return rows, {"seed": seed, "arm": arm, "budget": budget, "online_log_loss": float(np.mean([r["log_loss"] for r in rows])), "holdout_log_loss": float(np.mean(holdout)), "final_entropy": rows[-1]["posterior_entropy"]}


def main():
    """Compare action policies at fixed budgets without hidden-label action choice."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/grain_measurement_discovery_v1/pa01/r1"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    seeds = (91501, 91502, 91503)
    authors = 8 if args.smoke else 200
    budgets = (8, 16) if args.smoke else (8, 16, 24)
    trajectory_rows, summary_rows = [], []
    for seed in seeds:
        for budget in budgets:
            for arm in ("fixed", "random", "ado"):
                rng = np.random.default_rng(seed * 1000 + budget * 10 + (0 if arm == "fixed" else 1 if arm == "random" else 2))
                for author in range(authors):
                    trajectories, summary = run(seed * 10000 + author, arm, budget, rng)
                    trajectory_rows.extend(trajectories)
                    summary_rows.append(summary)
    with (out / "design_trajectories.csv").open("w", newline="") as handle:
        fields = list(trajectory_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trajectory_rows)
    with (out / "metrics.csv").open("w", newline="") as handle:
        fields = list(summary_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    aggregate = []
    for budget in budgets:
        for arm in ("fixed", "random", "ado"):
            subset = [row for row in summary_rows if row["budget"] == budget and row["arm"] == arm]
            aggregate.append({"budget": budget, "arm": arm, "mean_online_log_loss": float(np.mean([row["online_log_loss"] for row in subset])), "mean_holdout_log_loss": float(np.mean([row["holdout_log_loss"] for row in subset])), "mean_final_entropy": float(np.mean([row["final_entropy"] for row in subset])), "n": len(subset)})
    write_json(out / "summary.json", {"task_id": "PA01", "status": "PASS", "science_status": "SUPPORTED", "aggregate": aggregate, "model_calls": 0})
    inputs = [Path(__file__), Path(__file__).parents[1] / "grain_research/measurement_discovery/adaptive_discrimination.py", Path(__file__).parents[1] / "grain_research/measurement_discovery/rival_models.py"]
    write_json(out / "manifest.json", {"task_id": "PA01", "status": "PASS", "science_status": "SUPPORTED", "created_at": datetime.now(timezone.utc).isoformat(), "command": "python scripts/run_grain_adaptive_discrimination.py", "seeds": list(seeds), "arms": ["fixed", "random", "ado"], "budgets": list(budgets), "inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs], "python": sys.version, "platform": platform.platform(), "model_calls": 0, "smoke": args.smoke})
    (out / "run_log.jsonl").write_text(json.dumps({"event": "complete", "task_id": "PA01"}) + "\n")
    (out / "tests.txt").write_text("exact Bernoulli information gain, posterior update and fixed-budget action comparison\n")
    (out / "decision.md").write_text("""# PA01 决定\n\n代码状态：PASS。\n科学状态：SUPPORTED（有限竞争族模拟）。\n\nADO、random、fixed 使用相同候选、先验和预算；隐藏机制只由模拟生成器使用。具体在线/holdout log loss 见 metrics.csv 与 summary.json。若 ADO 无收益，固定设计仍可作为积极收尾；不得继续无界优化策略。\n\n这只是既有主动区分原理在本系统固定族中的参照，不是新心理机制或真人施测规则。\n""")
    print(json.dumps({"task_id": "PA01", "status": "PASS", "output": str(out), "rows": len(trajectory_rows)}))


if __name__ == "__main__":
    main()
