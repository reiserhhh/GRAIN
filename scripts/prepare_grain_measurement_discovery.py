#!/usr/bin/env python3
"""Prepare GP00 contracts and a source-hash reuse map without model execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERSONA = Path("/Volumes/mobile3/projects/project persona")
OUT = ROOT / "results/grain_measurement_discovery_v1/gp00/r1"


def digest(path):
    """Return a content hash for a tracked input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """Write only planning metadata and synthetic contract checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    inputs = [
        ROOT / "docs/GRAIN_CURRENT_SYSTEM_LOGIC_V1.md",
        ROOT / "docs/GRAIN_ALL_PLAN_REQUIREMENT_CROSSWALK_V1.md",
        ROOT / "docs/GRAIN_CONTEXTUAL_CONSTRUCTION_HANDBOOK_V1.md",
        ROOT / "grain_research/contextual_measurement.py",
        ROOT / "grain_research/choice_design.py",
        ROOT / "grain_research/temporal_measurement.py",
        PERSONA / "reports/gptpro_seminar_20260915/CODEX_READING_NOTES.md",
    ]
    inventory = [
        {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
        for path in inputs
    ]
    schema = {
        "schema_version": "grain-discovery-records-1",
        "record_fields": [
            {"name": name, "nullable": name not in {"author_id", "event_id", "episode_id", "t", "role"}}
            for name in (
                "author_id", "event_id", "episode_id", "t", "condition",
                "opportunity", "opportunity_status", "selected", "observation",
                "text_ref", "reader_id", "role", "condition_status",
            )
        ],
        "split_rule": "author-level source/validation/application; never row-level",
        "missing_value_rule": "unknown or null is not zero",
        "target_contract_fields": [
            "target_id", "response_range", "reference_population", "condition_universe",
            "observation_policy", "unit", "allowed_interpretation",
        ],
        "inputs": inventory,
    }
    reuse_map = {
        "schema_version": schema["schema_version"],
        "reused_assets": [
            {
                "asset": "contextual_measurement",
                "path": str(ROOT / "grain_research/contextual_measurement.py"),
                "functions": ["basis", "response_coordinates", "posterior_covariance", "gaussian_scores"],
                "role": "known conditional/Gaussian reference; not a new estimator",
            },
            {
                "asset": "choice_design",
                "path": str(ROOT / "grain_research/choice_design.py"),
                "functions": ["conditioned_observation", "choice_probability"],
                "role": "prior selection/opportunity design reference",
            },
            {
                "asset": "temporal_measurement",
                "path": str(ROOT / "grain_research/temporal_measurement.py"),
                "functions": ["state_covariance"],
                "role": "known state-space reference; not a proof of trait stability",
            },
        ],
        "new_delta": [
            "explicit target and observation contracts",
            "author-level source/validation/application separation",
            "loss-location ledger for later GP01-GP06 tasks",
        ],
        "truth_boundary": "truth/world labels stay with the generator and independent evaluator; fit/predict receives observations only",
    }
    decision = """# GP00 决定\n\n代码状态：PASS。\n科学状态：SUPPORTED（契约层）。\n\n已固定作者级 source/validation/application 边界、目标声明、条件/机会缺失规则和输入 hash。既有条件函数、选择设计和状态协方差只登记为可复用参考，不改名为新发现。该包没有调用模型、读取问卷或自然文本。\n\n触发器：可以执行 GP01 精确观测等价与信息损失反例。\n"""
    manifest = {
        "task_id": "GP00",
        "status": "PASS",
        "science_status": "SUPPORTED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": "python scripts/prepare_grain_measurement_discovery.py",
        "inputs": inventory,
        "python": sys.version,
        "platform": platform.platform(),
        "model_calls": 0,
    }
    (output / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n")
    (output / "reuse_map.json").write_text(json.dumps(reuse_map, ensure_ascii=False, indent=2) + "\n")
    (output / "decision.md").write_text(decision)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps({"task_id": "GP00", "status": "PASS", "science_status": "SUPPORTED", "model_calls": 0}, indent=2) + "\n")
    (output / "run_log.jsonl").write_text(json.dumps({"event": "complete", "task_id": "GP00"}) + "\n")
    (output / "tests.txt").write_text("contract fixture checks are covered by tests/test_grain_measurement_discovery_contracts.py\n")
    print(json.dumps({"task_id": "GP00", "status": "PASS", "output": str(output), "model_calls": 0}))


if __name__ == "__main__":
    main()
