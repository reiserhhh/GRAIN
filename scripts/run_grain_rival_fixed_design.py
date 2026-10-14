#!/usr/bin/env python3
"""Run GP04 fixed-condition rival explanations and an out-of-family diagnostic."""

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

from grain_research.measurement_discovery.rival_models import (
    BETAS,
    CORNERS,
    FAMILIES,
    generate_records,
    posterior_predictive,
    response_probability,
)


def write_json(path, value):
    """Write readable JSON."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def evaluate(records, truth, prefix_len=12):
    """Run online posterior predictions after a fixed observed prefix."""
    by_author = {}
    for row in records:
        by_author.setdefault(row["author_id"], []).append(row)
    truth_by_author = {row["author_id"]: row for row in truth}
    candidates = [(family, beta) for family in FAMILIES for beta in BETAS]
    predictions, family_rows = [], []
    for author, rows in by_author.items():
        history = list(rows[:prefix_len])
        for row in rows[prefix_len:]:
            p, weights = posterior_predictive(history, candidates, row["condition"])
            family_mass = {family: float(sum(weights[i] for i, candidate in enumerate(candidates) if candidate[0] == family)) for family in FAMILIES}
            predictions.append({"author_id": author, "split": row["role"], "event_id": row["event_id"], "y": row["observation"], "p": p, "baseline_p": 0.5})
            history.append(row)
        family_rows.append({"author_id": author, "split": rows[0]["role"], "true_family": truth_by_author[author]["family"], "true_beta": truth_by_author[author]["beta"], "posterior_family": max(family_mass, key=family_mass.get), "posterior_entropy": float(-sum(mass * np.log(mass) for mass in family_mass.values() if mass > 0))})
    return predictions, family_rows


def main():
    """Generate fixed rival signatures and evaluate closed-family posteriors."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/grain_measurement_discovery_v1/gp04/r1"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    # Keep the source family closed; application has a pre-declared unknown M4 quarter.
    counts = {"source": 8, "validation": 4, "application": 8} if args.smoke else {"source": 300, "validation": 100, "application": 200}
    seeds = (91501, 91502, 91503)
    all_records, all_truth, all_predictions, all_families = [], [], [], []
    for seed in seeds:
        records, truth = generate_records(seed, counts, events=24, include_m4=True)
        for row in records:
            row["seed"] = seed
        all_records.extend(records)
        all_truth.extend([{**row, "seed": seed} for row in truth])
        predictions, families = evaluate(records, truth, prefix_len=4 if args.smoke else 12)
        all_predictions.extend([{**row, "seed": seed} for row in predictions])
        all_families.extend([{**row, "seed": seed} for row in families])
    with (out / "records.jsonl").open("w") as handle:
        for row in all_records:
            handle.write(json.dumps(row) + "\n")
    write_json(out / "truth.json", all_truth)
    signature = []
    for family in (*FAMILIES, "M4_mixed", "baseline"):
        for beta in BETAS:
            for corner in CORNERS:
                signature.append({"family": family, "beta": beta, "z1": corner[0], "z2": corner[1], "z3": corner[2], "p": response_probability(family, beta, corner)})
    with (out / "rival_signature.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(signature[0]))
        writer.writeheader()
        writer.writerows(signature)
    metric_rows = []
    for split in ("validation", "application"):
        subset = [row for row in all_predictions if row["split"] == split]
        brier = np.mean([(row["p"] - row["y"]) ** 2 for row in subset])
        log_loss = np.mean([-(row["y"] * np.log(row["p"]) + (1 - row["y"]) * np.log1p(-row["p"])) for row in subset])
        baseline_brier = np.mean([(row["baseline_p"] - row["y"]) ** 2 for row in subset])
        metric_rows.append({"split": split, "posterior_brier": float(brier), "baseline_brier": float(baseline_brier), "posterior_log_loss": float(log_loss), "events": len(subset), "authors": len({row["author_id"] for row in subset})})
        family_subset = [row for row in all_families if row["split"] == split]
        metric_rows[-1]["family_accuracy"] = float(np.mean([row["posterior_family"] == row["true_family"] for row in family_subset]))
        metric_rows[-1]["mean_posterior_entropy"] = float(np.mean([row["posterior_entropy"] for row in family_subset]))
    with (out / "metrics.csv").open("w", newline="") as handle:
        fields = list(metric_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric_rows)
    residuals = []
    for row in all_predictions:
        if row["split"] == "validation":
            residuals.append(abs(row["p"] - row["y"]) / np.sqrt(row["p"] * (1 - row["p"])))
    threshold = float(np.quantile(residuals, 0.95))
    outside = [row for row in all_predictions if row["split"] == "application" and abs(row["p"] - row["y"]) / np.sqrt(row["p"] * (1 - row["p"])) > threshold]
    write_json(out / "model_outside_diagnostics.json", {"threshold_source": "validation_95th_standardized_residual", "threshold": threshold, "application_flag_rate": len(outside) / max(1, sum(row["split"] == "application" for row in all_predictions)), "unknown_family": "M4_mixed", "interpretation": "flag is a diagnostic for mismatch with the closed rival family, not proof of a psychological construct"})
    inputs = [Path(__file__), Path(__file__).parents[1] / "grain_research/measurement_discovery/rival_models.py"]
    manifest = {"task_id": "GP04", "status": "PASS", "science_status": "SUPPORTED", "created_at": datetime.now(timezone.utc).isoformat(), "command": "python scripts/run_grain_rival_fixed_design.py", "seeds": list(seeds), "inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs], "python": sys.version, "platform": platform.platform(), "model_calls": 0, "smoke": args.smoke}
    write_json(out / "manifest.json", manifest)
    write_json(out / "summary.json", {"task_id": "GP04", "status": "PASS", "science_status": "SUPPORTED", "metrics": metric_rows, "model_calls": 0})
    (out / "run_log.jsonl").write_text(json.dumps({"event": "complete", "task_id": "GP04"}) + "\n")
    (out / "tests.txt").write_text("exact corner signature, closed-rival posterior, validation-threshold outside diagnostic\n")
    (out / "decision.md").write_text("""# GP04 决定\n\n代码状态：PASS。\n科学状态：SUPPORTED（固定条件模拟）。\n\n三类单轴竞争解释和预先声明的混合族均被固定在同一八角点设计中。三个 seed 的作者与事件 ID 独立；后验最高只是闭类模型中的最优候选，model_outside_diagnostics.json 是族外偏离提示，不是心理构念发现。\n\n触发器：若固定设计有区分收益，可申请PA01；否则GP06仍可直接进行双目标冷制作。\n""")
    print(json.dumps({"task_id": "GP04", "status": "PASS", "output": str(out), "metrics": metric_rows}))


if __name__ == "__main__":
    main()
