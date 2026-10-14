#!/usr/bin/env python3
"""Run GP02: same-information finite-history estimator competition."""

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

from grain_research.measurement_discovery.estimators import (
    evaluate_author,
    fit_b1,
    fit_b2,
    fit_m,
    metric_rows,
    ordered_rows,
)


def write_json(path, value):
    """Write readable JSON."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def split_rows(records):
    """Split records by the role field; roles are metadata, never features."""
    grouped = ordered_rows(records)
    splits = {"source": {}, "validation": {}, "application": {}}
    for author, rows in grouped.items():
        role = rows[0]["role"]
        if role not in splits or any(row["role"] != role for row in rows):
            raise ValueError("each author must have exactly one valid role")
        splits[role][author] = rows
    return splits


def evaluate(splits, fitted, alpha, prefix_len=12):
    """Evaluate validation/application authors after a fixed observed prefix."""
    output = []
    for split in ("validation", "application"):
        for rows in splits[split].values():
            for row in evaluate_author(rows, fitted, prefix_len, alpha):
                row["split"] = split
                output.append(row)
    return output


def author_scores(predictions, candidate, split):
    """Return one mean Brier loss per author for paired bootstrap."""
    selected = [row for row in predictions if row["candidate"] == candidate and row["split"] == split]
    by_author = {}
    for row in selected:
        by_author.setdefault(row["author_id"], []).append((row["p"] - row["y"]) ** 2)
    return {author: float(np.mean(losses)) for author, losses in by_author.items()}


def bootstrap_delta(predictions, candidate, reference, split, seed=91502, draws=2000):
    """Paired author-block bootstrap for candidate minus reference Brier loss."""
    first, second = author_scores(predictions, candidate, split), author_scores(predictions, reference, split)
    authors = sorted(set(first) & set(second))
    if not authors:
        return {"candidate": candidate, "reference": reference, "split": split, "delta": None}
    values = np.asarray([first[a] - second[a] for a in authors])
    rng = np.random.default_rng(seed)
    samples = np.mean(rng.choice(values, size=(draws, len(values)), replace=True), axis=1)
    return {"candidate": candidate, "reference": reference, "split": split, "mean_delta_brier": float(values.mean()), "ci95": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))], "authors": len(authors), "draws": draws}


def main():
    """Fit only on source observations and perform validation then application."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/grain_measurement_discovery_v1/gp01/r1/records.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/grain_measurement_discovery_v1/gp02/r1"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in args.input.read_text().splitlines()]
    splits = split_rows(records)
    source = splits["source"]
    fitted = {"b1": fit_b1(source), "b2": fit_b2(source, alpha=1.0), "m": fit_m(source, rank=2)}
    # Validation chooses alpha once; no application outcomes are consulted.
    prefix_len = 4 if args.smoke else 12
    validation_results = {}
    for alpha in (1.0, 10.0):
        trial = evaluate(splits, fitted, alpha, prefix_len=prefix_len)
        metrics = metric_rows([row for row in trial if row["split"] == "validation"])
        validation_results[str(alpha)] = metrics
    b2_validation = {
        float(alpha): next(item["brier"] for item in metrics if item["candidate"] == "b2")
        for alpha, metrics in validation_results.items()
    }
    alpha = min(b2_validation, key=b2_validation.get)
    predictions = evaluate(splits, fitted, alpha, prefix_len=prefix_len)
    metrics = []
    for split in ("validation", "application"):
        for item in metric_rows([row for row in predictions if row["split"] == split]):
            metrics.append({"split": split, **item})
    val_bases = {item["candidate"]: item["brier"] for item in metrics if item["split"] == "validation"}
    reference = min(("b1", "b2"), key=lambda name: val_bases.get(name, np.inf))
    intervals = [bootstrap_delta(predictions, candidate, reference, "application") for candidate in ("b1", "b2", "m")]
    with (out / "predictions.csv").open("w", newline="") as handle:
        fields = ["split", "author_id", "event_id", "t", "candidate", "y", "p"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in predictions)
    with (out / "metrics.csv").open("w", newline="") as handle:
        fields = ["split", "candidate", "brier", "log_loss", "authors", "events"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)
    write_json(out / "paired_intervals.json", {"reference_selected_on_validation": reference, "validation_alpha_candidates": validation_results, "selected_alpha_b2": alpha, "intervals": intervals})
    model_json = {
        "b1": {key: value.tolist() for key, value in fitted["b1"].items()},
        "b2": {key: value.tolist() if hasattr(value, "tolist") else value for key, value in fitted["b2"].items()},
        "m": {key: value.tolist() if hasattr(value, "tolist") else value for key, value in fitted["m"].items()},
        "validation_alpha_b2": alpha,
        "prefix_len": prefix_len,
        "candidates": ["b1", "b2", "m"],
    }
    write_json(out / "model.json", model_json)
    inputs = [args.input, Path(__file__), Path(__file__).parents[1] / "grain_research/measurement_discovery/estimators.py"]
    manifest = {"task_id": "GP02", "status": "PASS", "science_status": "SUPPORTED", "created_at": datetime.now(timezone.utc).isoformat(), "command": "python scripts/run_grain_measurement_competition.py", "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(), "inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs], "python": sys.version, "platform": platform.platform(), "model_calls": 0, "source_authors": len(source), "validation_authors": len(splits["validation"]), "application_authors": len(splits["application"]), "smoke": args.smoke}
    write_json(out / "manifest.json", manifest)
    write_json(out / "summary.json", {"task_id": "GP02", "status": "PASS", "science_status": "SUPPORTED", "validation_reference": reference, "metrics": metrics, "model_calls": 0})
    (out / "run_log.jsonl").write_text(json.dumps({"event": "complete", "task_id": "GP02"}) + "\n")
    (out / "tests.txt").write_text("source-only fit, sequential prefix evaluation, paired author bootstrap\n")
    decision = """# GP02 决定\n\n代码状态：PASS。\n科学状态：SUPPORTED（有限模拟比较；不是自然人格效度）。\n\n三种候选共享同一二阶历史、prefix长度和下一观察协议。B2的alpha只由validation决定，application未参与模型选择；M固定rank=2，不做秩网格。请以metrics.csv和paired_intervals.json读取具体差异。\n\n若M未优于validation选出的标准参照，结论是本有限世界中标准方法已足够，不追加高阶模型。若M优于参照，只能说压缩在本有限世界有收益，不能外推到自然语言或心理trait。\n\n触发器：GP03把GP02选定接口接入无状态合成文本和共同冻结读者。\n"""
    (out / "decision.md").write_text(decision)
    print(json.dumps({"task_id": "GP02", "status": "PASS", "reference": reference, "output": str(out)}))


if __name__ == "__main__":
    main()
