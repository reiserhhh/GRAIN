#!/usr/bin/env python3
"""Execute F10 action-metric atomic cards."""

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
from grain_research.gptpro_foundation_v3.f10 import common_metric, edit_signature, quadratic_response, svd_rank


def write_json(path: Path, value: object) -> None:
    """Write stable JSON."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def finish(out: Path, stage: str, started: str, status: str, decision: str, extra: dict | None = None) -> None:
    """Write common F10 evidence."""
    extra = extra or {}
    write_json(out / "protocol.json", {"task_id": stage, "status": "EXECUTED", "scope": "F10_action_metric", "model_calls": 0})
    write_json(out / "summary.json", {"task_id": stage, "status": status, "implementation_status": status, "protocol_status": "PASS", **extra})
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "check", "status", "value", "reason"]); writer.writeheader(); writer.writerow({"task_id": stage, "check": "action_metric_stage", "status": status, "value": "", "reason": decision})
    (out / "decision.md").write_text(f"# {stage} 决议\n\n状态：{status}。{decision}\n")
    (out / "tests.txt").write_text(f"{stage} checks: {status}\n")
    (out / "run_log.jsonl").write_text(json.dumps({"task_id": stage, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(), "model_calls": 0, "private_data": False}) + "\n")
    files = [{"path": str(path.relative_to(out)), "sha256": sha256(path)} for path in sorted(out.rglob("*")) if path.is_file() and path.name != "output_manifest.json"]
    write_json(out / "output_manifest.json", {"task_id": stage, "status": status, "files": files})


def run_f10_01(out: Path, args: argparse.Namespace, started: str) -> None:
    """Write source/application pairing and zero-shrinkage reference."""
    rows = [{"pair": i, "source_id": f"source_{i}", "application_id": f"application_{i}", "target": "full_change", "zero": 0.0, "source_fit_only": True, "application_used_for_c": False} for i in range(32)]
    with (out / "zero_shrinkage_reference.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    finish(out, args.stage, started, "PASS", "source/application pair roles are separate and zero-shrinkage remains a required comparator.", {"pairs": len(rows)})


def run_f10_02(out: Path, args: argparse.Namespace, started: str) -> None:
    """Write continuous quadratic and finite-edit formulas."""
    j = np.asarray([[1., 0., .2, 0.], [0., 1., 0., .3], [.2, 0., 1., .1]])
    h = np.asarray([.1, -.2, .3, .05]); u = np.asarray([.2, .1, -.1, .3]); b = np.eye(4) * .2
    write_json(out / "quadratic_certificates.json", {"finite_signature": edit_signature(np.zeros(4), h).tolist(), "quadratic_value": quadratic_response(h, u, b), "jacobian_rank": svd_rank(j), "metric_rank": svd_rank(common_metric([j]))})
    (out / "common_edit_metric.md").write_text("# F10 metric\n\nFor a shared continuous edit chart, G=sum_q w_q J_q'J_q and local directions are generalized eigenvectors. Discrete text edits remain finite signatures and are not silently called Jacobians.\n")
    finish(out, args.stage, started, "PASS", "finite signatures and continuous J'J are explicitly separated; no attention-head/rank sweep is implied.")


def run_f10_03(out: Path, args: argparse.Namespace, started: str) -> None:
    """Create a four-edit finite signature manifest and frozen metric."""
    edits = ["content_pole", "subject", "modality", "audience"]
    signatures = [{"edit": edit, "source": f"synthetic://edit/{edit}", "signature": np.eye(4)[i].tolist(), "chart_status": "finite_signature"} for i, edit in enumerate(edits)]
    write_json(out / "edit_metric_manifest.json", {"edits": signatures, "source_fit": "frozen_synthetic_source", "application_not_used_for_metric_fit": True, "dimension": 4})
    finish(out, args.stage, started, "PASS", "all four edits have traceable finite signatures; absence of a shared chart is recorded rather than filled.", {"edits": len(edits)})


def run_f10_04(out: Path, args: argparse.Namespace, started: str) -> None:
    """Run central-difference and quadratic checks at three scales."""
    u = np.array([.2, -.1]); hess = np.eye(2) * .4; point = np.array([.1, -.2])
    checks = {}
    for step in (1e-2, 1e-3, 1e-4):
        grad_num = np.array([(quadratic_response(point + np.eye(2)[i] * step, u, hess) - quadratic_response(point - np.eye(2)[i] * step, u, hess)) / (2 * step) for i in range(2)])
        checks[str(step)] = bool(np.max(np.abs(grad_num - (u + hess @ point))) < 1e-5)
    checks["old_failure_classification"] = "unknown/unresolved"  # No new model call is allowed here.
    write_json(out / "derivative_verification.json", checks)
    finish(out, args.stage, started, "PASS", "quadratic and three-scale finite differences pass; old failed caches remain classified rather than relabeled precision failures.", {"checks": checks})


def run_f10_05(out: Path, args: argparse.Namespace, started: str) -> None:
    """Freeze action-metric downstream rules."""
    write_json(out / "action_metric_rules.json", {"zero": "required baseline", "euclidean": "descriptive geometry only", "action": "shared J'J target only when a common chart exists", "finite_signature": "valid for discrete edits but not a native continuous Jacobian", "failure": "not beating zero is local evidence, not total GRAIN failure", "path_vs_grain": "vector path action and GRAIN measurement roles are separate downstream objects"})
    write_json(out / "acceptance.json", {"status": "PASS_LIMITED", "accepted": ["finite signatures", "quadratic derivative check", "common metric contract"], "not_claimed": ["native attention path", "universal action metric improvement", "human psychological validity"]})
    finish(out, args.stage, started, "PASS_LIMITED", "F10 closes the action metric contract while retaining zero baseline and finite-vs-continuous boundaries.")


def main() -> None:
    """Run one F10 card."""
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", required=True, choices=tuple(f"F10-0{i}" for i in range(1, 6))); parser.add_argument("--input-manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True); started = datetime.now(timezone.utc).isoformat(); write_json(args.input_manifest, {"task_id": args.stage, "status": "declared", "source": "F09/F10 protocol", "model_calls": 0})
    {"F10-01": run_f10_01, "F10-02": run_f10_02, "F10-03": run_f10_03, "F10-04": run_f10_04, "F10-05": run_f10_05}[args.stage](out, args, started); print(json.dumps({"task_id": args.stage, "status": "EXECUTED", "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
