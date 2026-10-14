#!/usr/bin/env python3
"""HM35 companion: normalize only the MPS device label, preserving frozen r1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest
from scripts.run_grain_math_response import write_json
from scripts.run_grain_upstream_routes import collector_guard

ORIGINAL = ROOT / "scripts/run_grain_math_mixed_query_path.py"
EXPECTED = "2d2889cd43dbccee5c94c76fd443b6ab03071e2e61ddfebb7dbb59dd22f99a9b"
OLD = 'str(next(observer.model.parameters()).device) != spec["device"]'
NEW = 'next(observer.model.parameters()).device.type != spec["device"]'
OUT = ROOT / "results/grain_math_response_v1/mixed_query_path_r2"


def runtime():
    """Execute known local frozen code with exactly one device-label expression repair."""
    if digest(ORIGINAL) != EXPECTED:
        raise ValueError("frozen HM35 source changed")
    source = ORIGINAL.read_text()
    if source.count(OLD) != 1:
        raise ValueError("one exact device-label expression required")
    scope = {"__name__": "hm35_device_companion", "__file__": str(ORIGINAL)}
    exec(compile(source.replace(OLD, NEW), str(ORIGINAL), "exec"), scope)  # noqa: S102
    scope["OUT"] = OUT
    scope["CODE"] += (
        "scripts/run_grain_math_mixed_query_path_r2.py",
        "tests/test_grain_math_mixed_query_path_r2.py",
        "docs/GRAIN_MIXED_QUERY_PATH_DEVICE_REPAIR_V1.md",
    )
    return scope


def main():
    """Run the unchanged finite stages in a separate companion result directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "source", "checkpoint", "application", "evaluate")
    )
    parser.add_argument("--out", type=Path, default=OUT)
    a = parser.parse_args()
    scope = runtime()
    with collector_guard(a.out):
        if a.stage == "prepare":
            scope["prepare"](a.out)
            write_json(
                a.out / "00/device_adapter.json",
                {
                    "primary_sha256": EXPECTED,
                    "old_expression": OLD,
                    "new_expression": NEW,
                    "changes": 1,
                    "backend_changed": False,
                    "fixed_tolerances_changed": False,
                    "models_or_inputs_changed": False,
                },
            )
        elif a.stage == "checkpoint":
            scope["freeze_source"](a.out)
        elif a.stage == "evaluate":
            scope["evaluate"](a.out)
        else:
            scope["calls"](a.out, a.stage == "source")


if __name__ == "__main__":
    main()
