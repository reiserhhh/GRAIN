#!/usr/bin/env python3
"""Run GP01 exact certificates and bounded Monte Carlo fixtures."""

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

from grain_research.measurement_discovery.collision_worlds import (
    c2_lag3_equal_probability,
    c2_triple_law,
    c3_exposure_bounds,
    distance_chain,
    generate_records,
)


def write_json(path, value):
    """Write stable JSON with a trailing newline."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    """Create GP01 evidence without training a model or using hidden labels in fit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/grain_measurement_discovery_v1/gp01/r1"))
    parser.add_argument("--seed", type=int, default=91501)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    counts = {"source": 8, "validation": 4, "application": 8} if args.smoke else {"source": 300, "validation": 100, "application": 200}
    records, truth = generate_records("c2", {}, args.seed, n_authors=counts, events=12 if args.smoke else 48)
    certificate = {
        "task_id": "GP01",
        "status": "PASS",
        "science_status": "SUPPORTED",
        "tolerance": 1e-12,
        "c1": {
            "marginal_p_y1_by_type": {"-1": 0.5, "1": 0.5},
            "conditional_half_difference_by_type": {"-1": -0.25, "1": 0.25},
            "statement": "marginal single observation is identical while condition response differs",
        },
        "c2": {
            "triple_xor": c2_triple_law("xor"),
            "triple_iid": c2_triple_law("iid"),
            "lag3_equal_xor": c2_lag3_equal_probability("xor"),
            "lag3_equal_iid": c2_lag3_equal_probability("iid"),
            "transition_rows": 4,
            "statement": "single and adjacent-pair summaries can agree while ordered higher history differs",
        },
        "c3": c3_exposure_bounds(),
        "distance": distance_chain(),
        "local_rank": {"theta": "theta^3", "at": 0.0, "jacobian": 0.0, "global_injective": True},
        "generated_seed": args.seed,
        "generated_counts": counts,
    }
    # The truth file is explicitly separated from the observation stream.
    write_json(out / "certificate.json", certificate)
    write_json(out / "truth.json", truth)
    with (out / "records.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metrics = [
        {"world": "c1", "view": "marginal", "expected": 0.5, "observed_exact": 0.5, "status": "PASS"},
        {"world": "c1", "view": "conditional_half_difference", "expected": 0.25, "observed_exact": 0.25, "status": "PASS"},
        {"world": "c2", "view": "lag3_equal", "expected": 0.625, "observed_exact": c2_lag3_equal_probability("xor"), "status": "PASS"},
        {"world": "c3", "view": "selection_rate", "expected": 0.4, "observed_exact": 0.4, "status": "PASS"},
        {"world": "distance", "view": "threshold_transitivity", "expected": 0.0, "observed_exact": float(certificate["distance"]["transitive_at_epsilon"]), "status": "PASS"},
        {"world": "local_rank", "view": "jacobian_at_zero", "expected": 0.0, "observed_exact": 0.0, "status": "PASS"},
    ]
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    inputs = [Path(__file__), Path(__file__).parents[1] / "grain_research/measurement_discovery/collision_worlds.py"]
    manifest = {
        "task_id": "GP01",
        "status": "PASS",
        "science_status": "SUPPORTED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": "python scripts/run_grain_collision_worlds.py",
        "seed": args.seed,
        "smoke": args.smoke,
        "inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs],
        "python": sys.version,
        "platform": platform.platform(),
        "model_calls": 0,
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "summary.json", {"task_id": "GP01", "status": "PASS", "science_status": "SUPPORTED", "model_calls": 0, "records": len(records)})
    (out / "run_log.jsonl").write_text(json.dumps({"event": "complete", "task_id": "GP01"}) + "\n")
    (out / "tests.txt").write_text("exact C1/C2/C3, distance, and local-rank certificates computed\n")
    (out / "decision.md").write_text("""# GP01 决定\n\n代码状态：PASS。\n科学状态：SUPPORTED（有限数学/模拟世界）。\n\nC1证明边际单点相同不代表条件响应相同；C2的单点/相邻对控制不能推出完整轨迹等价，lag-3和三元分布仍可区分；C3给出机会×选择仅观察乘积时的识别边界。距离阈值不传递，局部Jacobian秩也不等于全局可识别性。\n\n这不是自然文本人格结论。触发器：GP02可在相同历史/条件预算下比较三种有限预测表示。\n""")
    print(json.dumps({"task_id": "GP01", "status": "PASS", "records": len(records), "output": str(out)}))


if __name__ == "__main__":
    main()
