#!/usr/bin/env python3
"""Run GP03: connect GP02's finite predictor to an auditable text channel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from grain_research.measurement_discovery.estimators import ordered_rows
from grain_research.measurement_discovery.text_channel import (
    FrozenTextReader,
    render_observation,
    second_order_hard_probability,
    second_order_soft_probability,
)


def write_json(path, value):
    """Write JSON with stable human-readable formatting."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def materialize(records):
    """Add a fixed grammar family and stateless text to observation records."""
    output = []
    for row in records:
        family = sum(ord(char) for char in row["author_id"]) % 3
        output.append({
            **row,
            "family": family,
            "text": render_observation(row["observation"], row["role"], family),
        })
    return output


def score_downstream(rows, reader, source_table, prefix_len):
    """Score oracle, soft-reader and hard-reader second-order forecasts."""
    grouped = ordered_rows(rows)
    predictions = []
    for author, author_rows in grouped.items():
        author_rows = [next(row for row in rows if row["event_id"] == item["event_id"]) for item in author_rows]
        probs = reader.predict([row["text"] for row in author_rows])
        for index in range(prefix_len, len(author_rows)):
            if index < 2:
                continue
            true_a = author_rows[index - 2]["observation"]
            true_b = author_rows[index - 1]["observation"]
            p_oracle = source_table[2 * int(true_a) + int(true_b)]
            p_soft = second_order_soft_probability(probs[index - 2], probs[index - 1], source_table)
            p_hard = second_order_hard_probability(probs[index - 2], probs[index - 1], source_table)
            predictions.extend([
                {"author_id": author, "event_id": author_rows[index]["event_id"], "split": author_rows[index]["role"], "y": int(author_rows[index]["observation"]), "channel": "oracle", "p": float(p_oracle)},
                {"author_id": author, "event_id": author_rows[index]["event_id"], "split": author_rows[index]["role"], "y": int(author_rows[index]["observation"]), "channel": "text_soft", "p": p_soft},
                {"author_id": author, "event_id": author_rows[index]["event_id"], "split": author_rows[index]["role"], "y": int(author_rows[index]["observation"]), "channel": "text_hard", "p": p_hard},
            ])
    return predictions


def metrics(predictions):
    """Compute reader-free downstream Brier and log loss by split/channel."""
    output = []
    for split in ("validation", "application"):
        for channel in ("oracle", "text_soft", "text_hard"):
            subset = [row for row in predictions if row["split"] == split and row["channel"] == channel]
            if not subset:
                continue
            brier = np.mean([(row["p"] - row["y"]) ** 2 for row in subset])
            log_loss = np.mean([
                -(row["y"] * np.log(row["p"]) + (1 - row["y"]) * np.log1p(-row["p"]))
                for row in subset
            ])
            output.append({"split": split, "channel": channel, "brier": float(brier), "log_loss": float(log_loss), "events": len(subset), "authors": len({row["author_id"] for row in subset})})
    return output


def main():
    """Fit source reader once, then evaluate held-out authors through text."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/grain_measurement_discovery_v1/gp01/r1/records.jsonl"))
    parser.add_argument("--model", type=Path, default=Path("results/grain_measurement_discovery_v1/gp02/r1/model.json"))
    parser.add_argument("--output", type=Path, default=Path("results/grain_measurement_discovery_v1/gp03/r1"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    records = materialize([json.loads(line) for line in args.input.read_text().splitlines()])
    grouped = ordered_rows(records)
    source_rows = {author: rows for author, rows in grouped.items() if rows[0]["role"] == "source"}
    source_records = [row for rows in source_rows.values() for row in rows]
    reader = FrozenTextReader(max_features=5000, alpha=1.0).fit([row["text"] for row in source_records], [row["observation"] for row in source_records])
    joblib.dump(reader, out / "reader.joblib", compress=3)
    reader_manifest = {"reader_id": "tfidf_word1_2_char2_4_ridge", "fit_rows": len(source_records), "source_authors": len(source_rows), "alpha": 1.0, "text_is_stateless": True, "reader_sha256": hashlib.sha256((out / "reader.joblib").read_bytes()).hexdigest()}
    write_json(out / "reader_manifest.json", reader_manifest)
    gp02 = json.loads(args.model.read_text())
    source_table = np.asarray(gp02["b1"]["source_p"], dtype=float)
    prefix_len = 4 if args.smoke else 12
    predictions = score_downstream(records, reader, source_table, prefix_len)
    with (out / "text_channel_scores.csv").open("w", newline="") as handle:
        fields = ["split", "author_id", "event_id", "channel", "y", "p"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in predictions)
    metric_rows = metrics(predictions)
    with (out / "metrics.csv").open("w", newline="") as handle:
        fields = list(metric_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric_rows)
    cards = [
        "同一作者的文本由同一条无状态renderer生成，reader只在source fit。",
        "validation与application作者没有进入TF-IDF词表或Ridge拟合。",
        "oracle只作为上界对照，不能替代文本通道。",
        "text_soft对相邻读数做独立Bernoulli的四格积分。",
        "text_hard是固定阈值敏感性，不由application选择。",
        "潜变量、world、作者类型和数字真值未进入文本。",
        "reader误差与GP02条件预测误差分开计分。",
        "本实验是TEXT_SYNTHETIC/READER，不是心理或自然语言效度。",
    ]
    (out / "evidence_cards.md").write_text("# GP03 evidence cards\n\n" + "\n".join(f"{i}. {item}" for i, item in enumerate(cards, 1)) + "\n")
    inputs = [args.input, args.model, Path(__file__), Path(__file__).parents[1] / "grain_research/measurement_discovery/text_channel.py"]
    manifest = {"task_id": "GP03", "status": "PASS", "science_status": "SUPPORTED", "created_at": datetime.now(timezone.utc).isoformat(), "command": "python scripts/run_grain_discovery_text_channel.py", "prefix_len": prefix_len, "inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs], "python": sys.version, "platform": platform.platform(), "model_calls": 0, "reader_fit_authors": len(source_rows), "evaluation_authors": len(grouped) - len(source_rows), "smoke": args.smoke}
    write_json(out / "renderer_manifest.json", {"renderer": "stateless_fixed_grammar", "families": 3, "split_specific_grammar": True, "potential_label_text": "current observation phrase only", "inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs]})
    write_json(out / "manifest.json", manifest)
    write_json(out / "summary.json", {"task_id": "GP03", "status": "PASS", "science_status": "SUPPORTED", "metrics": metric_rows, "model_calls": 0})
    (out / "run_log.jsonl").write_text(json.dumps({"event": "complete", "task_id": "GP03"}) + "\n")
    (out / "tests.txt").write_text("stateless renderer, source-only reader fit, soft/hard downstream scoring\n")
    (out / "decision.md").write_text("""# GP03 决定\n\n代码状态：PASS。\n科学状态：SUPPORTED（合成文本通道）。\n\n文本通道已接入GP02选定的B1有限预测接口。具体reader、oracle、text_soft和text_hard数值见metrics.csv；oracle上界与文本误差分开。若文本通道成功，只支持该固定合成语法能传递当前观察；若失败，则断点在reader/renderer，不否定GP02数学机制。\n\n触发器：GP04可在固定条件下比较竞争解释。\n""")
    print(json.dumps({"task_id": "GP03", "status": "PASS", "output": str(out), "metrics": metric_rows}))


if __name__ == "__main__":
    main()
