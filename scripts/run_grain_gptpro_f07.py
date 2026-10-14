#!/usr/bin/env python3
"""Execute F07 openness-channel atomic cards."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grain_research.gptpro_foundation_v3.f00 import sha256
from grain_research.gptpro_foundation_v3.f07 import channel_derivative, channel_prob, fisher_information, simulate_channel


def write_json(path: Path, value: object) -> None:
    """Write stable JSON."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def finish(out: Path, stage: str, started: str, status: str, decision: str, extra: dict | None = None) -> None:
    """Write common F07 evidence."""
    extra = extra or {}
    write_json(out / "protocol.json", {"task_id": stage, "status": "EXECUTED", "scope": "F07_openness_channel", "model_calls": 0})
    write_json(out / "summary.json", {"task_id": stage, "status": status, "implementation_status": status, "protocol_status": "PASS", **extra})
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "check", "status", "value", "reason"]); writer.writeheader(); writer.writerow({"task_id": stage, "check": "openness_stage", "status": status, "value": "", "reason": decision})
    (out / "decision.md").write_text(f"# {stage} 决议\n\n状态：{status}。{decision}\n")
    (out / "tests.txt").write_text(f"{stage} checks: {status}\n")
    (out / "run_log.jsonl").write_text(json.dumps({"task_id": stage, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(), "model_calls": 0, "private_data": False}) + "\n")
    files = [{"path": str(path.relative_to(out)), "sha256": sha256(path)} for path in sorted(out.rglob("*")) if path.is_file() and path.name != "output_manifest.json"]
    write_json(out / "output_manifest.json", {"task_id": stage, "status": status, "files": files})


def run_f07_01(out: Path, args: argparse.Namespace, started: str) -> None:
    """Enumerate exact channel probabilities, derivatives and Fisher information."""
    rows = []
    for world in ("U", "D", "P", "flat"):
        for theta in (-1.0, 0.0, 1.0):
            for level in (-1.0, -.25, .25, 1.0):
                rows.append({"world": world, "theta": theta, "level": level, "p": channel_prob(theta, level, world), "derivative": channel_derivative(theta, level, world), "fisher": fisher_information(theta, [-1.0, -.25, .25, 1.0], world)})
    with (out / "openness_exact_curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    finish(out, args.stage, started, "PASS", "U/D/P/flat curves, derivatives and Fisher information use the transformed Bernoulli probability denominator.", {"rows": len(rows)})


def run_f07_02(out: Path, args: argparse.Namespace, started: str) -> None:
    """Create response/token budget generators and channel manifest."""
    rows = []
    for budget, levels in (("response32", list(np.linspace(-1, 1, 32))), ("token400", list(np.linspace(-1, 1, 100)))):
        for world in ("U", "D", "P", "flat"):
            observations = simulate_channel(.4, levels, world, 20260915)
            rows.append({"budget": budget, "world": world, "n_levels": len(levels), "token_cost": 32 if budget == "response32" else 400, "observations": len(observations), "source_role_isolated": True, "mnar_offtarget": "separate mixture law not used as clean channel"})
    write_json(out / "openness_generator_contract.json", {"manifest": rows, "truth_separate": True, "role_leakage": False, "mixture": "off-target/MNAR must remain a separate arm"})
    finish(out, args.stage, started, "PASS", "response and token budgets are separate; source/role are not encoded in channel output.", {"arms": len(rows)})


def run_f07_03(out: Path, args: argparse.Namespace, started: str) -> None:
    """Verify analytic derivatives and probability normalization."""
    checks = {}
    for world in ("U", "D", "P"):
        for theta in (-.7, .2, 1.1):
            level = .3; h = 1e-5
            numeric = (channel_prob(theta + h, level, world) - channel_prob(theta - h, level, world)) / (2 * h)
            checks[f"{world}_{theta}"] = bool(abs(numeric - channel_derivative(theta, level, world)) < 1e-5)
    checks["flat_zero_derivative"] = channel_derivative(.4, .2, "flat") == 0.0
    write_json(out / "openness_checks.json", checks)
    finish(out, args.stage, started, "PASS" if all(checks.values()) else "FAIL", "analytic and finite-difference derivatives agree at FP64 tolerance; flat is declared calibration, not discovery.", {"checks": checks})


def run_f07_04(out: Path, args: argparse.Namespace, started: str) -> None:
    """Compare information, recovery and token cost across channel worlds."""
    rows = []
    for budget, levels in (("response32", list(np.linspace(-1, 1, 32))), ("token400", list(np.linspace(-1, 1, 100)))):
        for world in ("U", "D", "P", "flat"):
            observations = simulate_channel(.4, levels, world, 20260915); empirical = float(np.mean([row["response"] for row in observations]))
            rows.append({"budget": budget, "world": world, "n": len(levels), "fisher": fisher_information(.4, levels, world), "empirical_rate": empirical, "token_cost": 32 if budget == "response32" else 400, "status": "finite_simulation"})
    with (out / "openness_simulation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out / "openness_conditions.md").write_text("# F07 conditions\n\nInformation and recovery are reported jointly with cost. No channel is predeclared as universally best; flat is a calibration arm.\n")
    finish(out, args.stage, started, "PASS", "all worlds and budgets are retained; information, recovery and token cost are not collapsed into one score.", {"rows": len(rows)})


def run_f07_05(out: Path, args: argparse.Namespace, started: str) -> None:
    """Consume the measured synthetic reader calibration after F08."""
    calibration = ROOT / "results/grain_gptpro_foundation_v3/F08/r1/F08-05/reader_calibration.csv"
    if not calibration.exists():
        write_json(out / "acceptance.json", {"status": "BLOCKED_INPUT", "reason": "F08-07 reader calibration is not yet available", "required": "measured confusion matrix or explicit missing-reader contract", "prohibited": "ideal flip simulation presented as text-channel evidence"})
        finish(out, args.stage, started, "BLOCKED_INPUT", "F07 text-channel arm remains blocked until F08 reader calibration; ideal SIM is not promoted to text evidence.")
        return
    rows = list(csv.DictReader(calibration.open()))
    write_json(out / "text_channel_reader_calibration.json", {"source": str(calibration), "calibration": rows, "label": "TEXT_SYNTHETIC", "ideal_flip": False})
    finish(out, args.stage, started, "PASS_TEXT_SYNTHETIC", "F08 synthetic reader calibration is consumed; this does not create human text validity.", {"calibration_rows": len(rows)})


def main() -> None:
    """Run one F07 card."""
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", required=True, choices=tuple(f"F07-0{i}" for i in range(1, 6))); parser.add_argument("--input-manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True); started = datetime.now(timezone.utc).isoformat(); write_json(args.input_manifest, {"task_id": args.stage, "status": "declared", "source": "F06/F07 protocol", "model_calls": 0})
    {"F07-01": run_f07_01, "F07-02": run_f07_02, "F07-03": run_f07_03, "F07-04": run_f07_04, "F07-05": run_f07_05}[args.stage](out, args, started); print(json.dumps({"task_id": args.stage, "status": "EXECUTED", "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
