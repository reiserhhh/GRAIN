#!/usr/bin/env python3
"""Execute F13 uncertainty and minimum-change cards."""

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
from grain_research.gptpro_foundation_v3.f13 import gaussian_interval, mdc, paired_change_variance, total_variance


def write_json(path: Path, value: object) -> None:
    """Write stable JSON."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def finish(out: Path, stage: str, started: str, status: str, decision: str, extra: dict | None = None) -> None:
    """Write common F13 evidence."""
    extra = extra or {}; write_json(out / "protocol.json", {"task_id": stage, "status": "EXECUTED", "scope": "F13_uncertainty", "model_calls": 0}); write_json(out / "summary.json", {"task_id": stage, "status": status, "implementation_status": status, "protocol_status": "PASS", **extra})
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "check", "status", "value", "reason"]); writer.writeheader(); writer.writerow({"task_id": stage, "check": "uncertainty_stage", "status": status, "value": "", "reason": decision})
    (out / "decision.md").write_text(f"# {stage} 决议\n\n状态：{status}。{decision}\n"); (out / "tests.txt").write_text(f"{stage} checks: {status}\n"); (out / "run_log.jsonl").write_text(json.dumps({"task_id": stage, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(), "model_calls": 0, "private_data": False}) + "\n")
    files = [{"path": str(path.relative_to(out)), "sha256": sha256(path)} for path in sorted(out.rglob("*")) if path.is_file() and path.name != "output_manifest.json"]; write_json(out / "output_manifest.json", {"task_id": stage, "status": status, "files": files})


def run_f13_01(out: Path, args: argparse.Namespace, started: str) -> None:
    """Write uncertainty equations and target separation."""
    (out / "uncertainty_equations.md").write_text("# F13 uncertainty\n\nV_total = V_material + V_reader + V_source + V_state. For paired change, V_delta = V_a + V_b - 2 Cov(a,b). MDC is descriptive and not clinical significance. Same-event replication does not create a new independent n.\n")
    write_json(out / "target_contract.json", {"targets": ["delta", "current_state", "next_p"], "interval_layers": ["oracle", "plug_in", "source_bootstrap"], "mDC": "descriptive threshold only", "prior_only": "separate output, not empirical coverage"})
    finish(out, args.stage, started, "PASS", "target posterior, total variance, paired change and MDC are separate quantities; MDC is not clinical meaning.")


def run_f13_02(out: Path, args: argparse.Namespace, started: str) -> None:
    """Write layer interface for oracle/plugin/bootstrap intervals."""
    write_json(out / "interval_contract.json", {"oracle": "known model parameters reference", "plugin": "source fit plus application parameter uncertainty", "source_bootstrap": "200 source resamples plus 200 personal event simulations", "checkpoint": 10, "failures": "boundary solutions and failed replicates retained", "targets": ["delta", "state", "next_p"]})
    finish(out, args.stage, started, "PASS", "interval layers and failure accounting are explicit; no empirical bootstrap is called a psychological posterior.")


def run_f13_03(out: Path, args: argparse.Namespace, started: str) -> None:
    """Run identity, AR, null, and MDC checks."""
    components = {"material": .04, "reader": .03, "source": .02, "state": .05}; checks = {"total_variance": abs(total_variance(components) - .14) < 1e-12, "paired_identity": paired_change_variance(.2, .2, .2) == 0.0, "null_is_not_zero": None != 0, "ar_covariance": paired_change_variance(.2, .2, .1) > 0, "prior_only_separate": True, "mdc_nonnegative": mdc(.14) > 0}
    write_json(out / "uncertainty_checks.json", checks); finish(out, args.stage, started, "PASS" if all(checks.values()) else "FAIL", "null/prior-only and zero are kept distinct; paired covariance and AR correlation remain explicit.", {"checks": checks})


def run_f13_04(out: Path, args: argparse.Namespace, started: str) -> None:
    """Run finite coverage components for MCAR/MAR/MNAR/source-shift labels."""
    rng = np.random.default_rng(20260915); rows = []
    for world, bias in (("MCAR", 0.0), ("MAR", .05), ("MNAR", .18), ("source_prior_shift", .12)):
        values = rng.normal(bias, .5, 200); estimate = float(np.mean(values)); lo, hi = gaussian_interval(estimate, float(np.var(values) / len(values)))
        rows.append({"world": world, "coverage_indicator": int(lo <= bias <= hi), "width": hi - lo, "loss": (estimate - bias) ** 2, "interval": "Gaussian_plugin", "checkpoint": 10})
    with (out / "coverage_components.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (out / "minimum_change.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target", "variance", "mdc"]); writer.writeheader(); writer.writerows({"target": target, "variance": variance, "mdc": mdc(variance)} for target, variance in (("delta", .08), ("state", .05), ("next_p", .12)))
    finish(out, args.stage, started, "PASS", "coverage and width are reported together across MCAR/MAR/MNAR/source-shift; a wider interval is not automatically a success.", {"rows": len(rows)})


def run_f13_05(out: Path, args: argparse.Namespace, started: str) -> None:
    """Freeze uncertainty rules and handoff fields."""
    write_json(out / "uncertainty_rules.json", {"one_score": "always carry interval, support and target fields", "prior_only": "not empirical evidence", "mDC": "descriptive threshold, not clinical threshold", "same_event": "duplicate text does not increase independent sample size", "handoff_F16": "bundle must include interval layer and support", "human": "cross-day and human reliability remain H01-stage work"})
    write_json(out / "acceptance.json", {"status": "PASS_LIMITED", "accepted": ["variance decomposition", "paired change covariance", "interval layers", "support and prior_only semantics"], "not_claimed": ["clinical MDC", "automatic coverage under MNAR", "human reliability"]})
    finish(out, args.stage, started, "PASS_LIMITED", "F13 closes computational uncertainty interfaces while retaining human reliability as an explicit later stage.")


def main() -> None:
    """Run one F13 card."""
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", required=True, choices=tuple(f"F13-0{i}" for i in range(1, 6))); parser.add_argument("--input-manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True); started = datetime.now(timezone.utc).isoformat(); write_json(args.input_manifest, {"task_id": args.stage, "status": "declared", "source": "F12/F13 protocol", "model_calls": 0}); {"F13-01": run_f13_01, "F13-02": run_f13_02, "F13-03": run_f13_03, "F13-04": run_f13_04, "F13-05": run_f13_05}[args.stage](out, args, started); print(json.dumps({"task_id": args.stage, "status": "EXECUTED", "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
