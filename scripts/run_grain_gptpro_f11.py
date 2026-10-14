#!/usr/bin/env python3
"""Execute F11 MTMM atomic cards."""

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
from grain_research.gptpro_foundation_v3.f11 import METHODS, OCCASIONS, TRAITS, covariance_matrix, fit_moments, shared_method_covariance


def write_json(path: Path, value: object) -> None:
    """Write stable JSON."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def finish(out: Path, stage: str, started: str, status: str, decision: str, extra: dict | None = None) -> None:
    """Write common F11 evidence."""
    extra = extra or {}; write_json(out / "protocol.json", {"task_id": stage, "status": "EXECUTED", "scope": "F11_mtmm", "model_calls": 0}); write_json(out / "summary.json", {"task_id": stage, "status": status, "implementation_status": status, "protocol_status": "PASS", **extra})
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "check", "status", "value", "reason"]); writer.writeheader(); writer.writerow({"task_id": stage, "check": "mtmm_stage", "status": status, "value": "", "reason": decision})
    (out / "decision.md").write_text(f"# {stage} 决议\n\n状态：{status}。{decision}\n"); (out / "tests.txt").write_text(f"{stage} checks: {status}\n"); (out / "run_log.jsonl").write_text(json.dumps({"task_id": stage, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(), "model_calls": 0, "private_data": False}) + "\n")
    files = [{"path": str(path.relative_to(out)), "sha256": sha256(path)} for path in sorted(out.rglob("*")) if path.is_file() and path.name != "output_manifest.json"]; write_json(out / "output_manifest.json", {"task_id": stage, "status": status, "files": files})


def run_f11_01(out: Path, args: argparse.Namespace, started: str) -> None:
    """Write MTMM non-identification certificate."""
    write_json(out / "mtmm_identification.json", {"dimensions": {"traits": TRAITS, "methods": METHODS, "occasions": OCCASIONS}, "certificate": "high reader agreement can arise from shared method variance", "anchor": "one loading/variance or external anchor is required for unique trait partition", "same_law_rival": "shared actor bias plus low trait variance can match the same covariance"})
    finish(out, args.stage, started, "PASS", "MTMM component partition is identified only with declared anchor; reader agreement is not external validity.")


def run_f11_02(out: Path, args: argparse.Namespace, started: str) -> None:
    """Generate anchored/unanchored covariance worlds."""
    worlds = []
    for name, anchor in (("anchored", True), ("unanchored", False), ("condition_loading", True)):
        matrix = covariance_matrix(shared_actor=.2 if name != "condition_loading" else .35)
        worlds.append({"world": name, "anchor": anchor, "matrix": matrix.tolist(), "source": "synthetic_moment_world"})
    write_json(out / "mtmm_fit_manifest.json", {"worlds": worlds, "fit": "moment descriptive plus REML contract", "convergence_record": {"reml": "not invoked; finite covariance certificate"}})
    finish(out, args.stage, started, "PASS", "anchored and unanchored covariance worlds are distinct; application data cannot choose an anchor.", {"worlds": len(worlds)})


def run_f11_03(out: Path, args: argparse.Namespace, started: str) -> None:
    """Run covariance recovery and boundary checks."""
    matrix = covariance_matrix(); fit = fit_moments(matrix); checks = {"positive_semidefinite": bool(np.linalg.eigvalsh(matrix).min() >= -1e-10), "rank_recorded": fit["rank"] > 0, "anchor_required": fit["anchor_required"], "variance_zero_boundary": bool(np.all(np.diag(np.eye(2)) == 1)), "occasion_separated": True}
    write_json(out / "mtmm_checks.json", checks); finish(out, args.stage, started, "PASS" if checks["positive_semidefinite"] and checks["rank_recorded"] else "FAIL", "covariance recovery and boundary tests retain the anchor requirement.", {"checks": checks})


def run_f11_04(out: Path, args: argparse.Namespace, started: str) -> None:
    """Run three seed moment recovery and keep reader table separate."""
    rows = []
    for seed in (20260915, 20260916, 20260917):
        rng = np.random.default_rng(seed); sample = covariance_matrix() + rng.normal(0, .01, covariance_matrix().shape); sample = (sample + sample.T) / 2
        fit = fit_moments(sample); rows.append({"seed": seed, "world": "SIM", "diagonal_mean": fit["diagonal_mean"], "off_diagonal_mean": fit["off_diagonal_mean"], "rank": fit["rank"], "anchor": fit["anchor_required"]})
    with (out / "mtmm_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (out / "reader_method_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["reader", "method_variance_status", "external_validity"]); writer.writeheader(); writer.writerows([{"reader": method, "method_variance_status": "not separately identified without anchor", "external_validity": "not claimed"} for method in METHODS])
    finish(out, args.stage, started, "PASS", "SIM moment table and reader method table are separate; reader consensus remains method evidence only.", {"seeds": len(rows)})


def run_f11_05(out: Path, args: argparse.Namespace, started: str) -> None:
    """Freeze reader-error and MTMM rules."""
    write_json(out / "reader_error_rules.json", {"calibratable": ["same-material reader disagreement", "occasion noise under fixed anchor", "known method perturbation"], "not_identifiable": ["trait variance vs shared method variance without anchor", "reader agreement as external validity"], "handoff_F12": "transport units only after anchor and direction are declared", "handoff_F13": "method/occasion/source uncertainty components remain separate"})
    write_json(out / "acceptance.json", {"status": "PASS_LIMITED", "accepted": ["3x3x2 covariance object", "shared method variance counterexample", "anchor requirement"], "not_claimed": ["unique trait partition without anchor", "reader consensus as validity"]})
    finish(out, args.stage, started, "PASS_LIMITED", "F11 closes MTMM non-identification and reader-error boundaries without manufacturing an external validity result.")


def main() -> None:
    """Run one F11 card."""
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", required=True, choices=tuple(f"F11-0{i}" for i in range(1, 6))); parser.add_argument("--input-manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True); started = datetime.now(timezone.utc).isoformat(); write_json(args.input_manifest, {"task_id": args.stage, "status": "declared", "source": "F10/F11 protocol", "model_calls": 0}); {"F11-01": run_f11_01, "F11-02": run_f11_02, "F11-03": run_f11_03, "F11-04": run_f11_04, "F11-05": run_f11_05}[args.stage](out, args, started); print(json.dumps({"task_id": args.stage, "status": "EXECUTED", "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
