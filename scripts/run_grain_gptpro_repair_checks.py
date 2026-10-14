#!/usr/bin/env python3
"""Replay repaired failure cases without promoting incomplete V3 task cards."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent / "project persona"
sys.path.insert(0, str(ROOT))

from grain_research.gptpro_foundation_v3.f00 import sha256


def load_runner(name: str):
    """Import a stage implementation without invoking its historical CLI."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dump(path: Path, value: object) -> None:
    """Write a generated diagnostic artifact."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    """Execute tests and finite corrected subcases; leave parent cards open."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("choose a fresh diagnostic revision")
    out.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    tests = [
        "tests/test_grain_gptpro_acceptance.py",
        "tests/test_grain_gptpro_execution.py",
        "tests/test_grain_gptpro_f15_certificates.py",
    ]
    code = [
        "grain_research/gptpro_foundation_v3/acceptance.py",
        "grain_research/gptpro_foundation_v3/execution.py",
        "grain_research/gptpro_foundation_v3/f15.py",
        "grain_research/gptpro_foundation_v3/f00.py",
        "scripts/run_grain_gptpro_f00.py",
        "scripts/run_grain_gptpro_f15.py",
        "scripts/run_grain_gptpro_f17.py",
        "scripts/run_grain_gptpro_repair_checks.py",
    ]
    inputs = [
        {"path": str(ROOT / path), "sha256": sha256(ROOT / path)}
        for path in code + tests
    ]
    for path in [
        PARENT / "docs/GRAIN_GPTPRO_ATOMIC_PROTOCOLS_V3.md",
        PARENT / "reports/gptpro_seminar_20260915/atomic_v3/execution_progress.json",
        PARENT / "reports/gptpro_seminar_20260915/atomic_v3/SOURCE_TASK_CROSSWALK.csv",
    ]:
        inputs.append({"path": str(path), "sha256": sha256(path)})
    dump(
        out / "input_manifest.json",
        {"scope": "repair diagnostics only; not DAG card acceptance", "inputs": inputs},
    )
    command = [sys.executable, "-m", "pytest", "-q", *tests]
    test = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    (out / "tests.txt").write_text(test.stdout + test.stderr)
    if test.returncode:
        raise RuntimeError("repair regression tests failed; see tests.txt")
    f00 = load_runner("run_grain_gptpro_f00")
    f15 = load_runner("run_grain_gptpro_f15")
    f17 = load_runner("run_grain_gptpro_f17")
    for subdir, stage, runner in [
        (
            "null_contract",
            "F00-02",
            lambda path, a: f00.run_contract_stage(a, path, started),
        ),
        (
            "garbling_subcase",
            "F15-02",
            lambda path, a: f15.run_f15_02(path, a, started),
        ),
        (
            "current_acceptance",
            "F17-05",
            lambda path, a: f17.run_f17_05(path, a, started),
        ),
    ]:
        path = out / subdir
        path.mkdir()
        runner(
            path,
            SimpleNamespace(
                stage=stage, seeds=[20260915, 20260916, 20260917], smoke=False
            ),
        )
    results = {
        name: json.loads((out / name / "summary.json").read_text())
        for name in ("null_contract", "garbling_subcase", "current_acceptance")
    }
    passed = (
        results["null_contract"]["status"] == "PASS"
        and all(results["garbling_subcase"]["checks"].values())
        and results["current_acceptance"]["status"] == "OPEN"
    )
    dump(
        out / "summary.json",
        {
            "status": "REPAIR_CHECKS_PASS" if passed else "REPAIR_CHECKS_FAIL",
            "parent_card_promotion": False,
            "research_goal_complete": False,
            "checks": results,
            "test_command": command,
            "test_exit_code": test.returncode,
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    dump(
        out / "output_manifest.json",
        {
            "files": [
                {"path": str(path.relative_to(out)), "sha256": sha256(path)}
                for path in sorted(out.rglob("*"))
                if path.is_file()
            ]
        },
    )
    print(
        json.dumps(
            {"output": str(out), "repair_checks_pass": passed, "goal_complete": False}
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
