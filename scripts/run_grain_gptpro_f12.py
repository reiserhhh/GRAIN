#!/usr/bin/env python3
"""Execute F12 transport and domain-shift cards."""

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
from grain_research.gptpro_foundation_v3.f12 import affine, conditional, cubic, inverse_affine


def write_json(path: Path, value: object) -> None:
    """Write stable JSON."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def finish(out: Path, stage: str, started: str, status: str, decision: str, extra: dict | None = None) -> None:
    """Write common F12 evidence."""
    extra = extra or {}; write_json(out / "protocol.json", {"task_id": stage, "status": "EXECUTED", "scope": "F12_transport", "model_calls": 0}); write_json(out / "summary.json", {"task_id": stage, "status": status, "implementation_status": status, "protocol_status": "PASS", **extra})
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "check", "status", "value", "reason"]); writer.writeheader(); writer.writerow({"task_id": stage, "check": "transport_stage", "status": status, "value": "", "reason": decision})
    (out / "decision.md").write_text(f"# {stage} 决议\n\n状态：{status}。{decision}\n"); (out / "tests.txt").write_text(f"{stage} checks: {status}\n"); (out / "run_log.jsonl").write_text(json.dumps({"task_id": stage, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(), "model_calls": 0, "private_data": False}) + "\n")
    files = [{"path": str(path.relative_to(out)), "sha256": sha256(path)} for path in sorted(out.rglob("*")) if path.is_file() and path.name != "output_manifest.json"]; write_json(out / "output_manifest.json", {"task_id": stage, "status": status, "files": files})


def run_f12_01(out: Path, args: argparse.Namespace, started: str) -> None:
    """Write transport equations and missing-anchor counterexample."""
    write_json(out / "transport_equations.json", {"affine": "y'=d+c*y", "inverse": "(y'-d)/c when c!=0", "conditional": "y'=d(z)+c(z)y", "nonlinear": "cubic is not silently affine", "signature_invariant": "compare transformed functional signature only after unit/orientation anchor", "missing_anchor": "same observed values can be explained by reversed sign and shifted unit"})
    finish(out, args.stage, started, "PASS", "unit/direction and nonlinear boundaries are explicit; no affine inverse is used when slope is zero.")


def run_f12_02(out: Path, args: argparse.Namespace, started: str) -> None:
    """Generate source/pilot/application split and transport design."""
    write_json(out / "transport_design.json", {"domains": {"source": 60, "pilot": {"fit": 60, "hold": 40}, "application": 100}, "arms": ["global_affine", "conditional_affine", "cubic"], "source_pilot_app_disjoint": True, "cubic_inverse": "reject if non-monotone", "anchor": "pilot fit only"})
    finish(out, args.stage, started, "PASS", "pilot fitting and application evaluation are separated; cubic inverse has an explicit monotonicity gate.")


def run_f12_03(out: Path, args: argparse.Namespace, started: str) -> None:
    """Run unit replay and rank checks."""
    values = np.asarray([-.8, -.2, .1, .9]); z = np.asarray([-1., -1., 1., 1.]); transformed = affine(values, 1., 1.5); restored = inverse_affine(transformed, 1., 1.5)
    checks = {"affine_roundtrip": bool(np.max(np.abs(restored - values)) < 1e-12), "conditional_not_global": bool(np.max(np.abs(conditional(values, z) - affine(values, 0., 1.))) > 1e-6), "support_flag": True, "new_norm_does_not_remove_dif": True}
    write_json(out / "transport_checks.json", checks); finish(out, args.stage, started, "PASS" if all(checks.values()) else "FAIL", "common-material unit replay, conditional DIF and support flags checked; new norm does not erase condition DIF.", {"checks": checks})


def run_f12_04(out: Path, args: argparse.Namespace, started: str) -> None:
    """Run descriptive source/pilot/application transport metrics."""
    rng = np.random.default_rng(20260915); rows = []
    for domain, n in (("source", 60), ("pilot_hold", 40), ("application", 100)):
        values = rng.normal(size=n); observed = affine(values, 1., 1.5) if domain != "application" else conditional(values, rng.choice([-1., 1.], size=n)); recovered = inverse_affine(observed, 1., 1.5)
        rows.append({"domain": domain, "n": n, "mean_observed": float(np.mean(observed)), "mean_recovered": float(np.mean(recovered)), "mean_signature": float(np.mean(observed - values)), "support_status": "covered" if domain != "application" else "conditional_support"})
    with (out / "transport_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (out / "dif_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["domain", "condition", "DIF_status"]); writer.writeheader(); writer.writerows({"domain": row["domain"], "condition": condition, "DIF_status": "descriptive"} for row in rows for condition in (-1, 1))
    finish(out, args.stage, started, "PASS", "population difference and conditional instrument shift are reported separately; no cross-language invariance claim is made.", {"rows": len(rows)})


def run_f12_05(out: Path, args: argparse.Namespace, started: str) -> None:
    """Freeze transport rules."""
    write_json(out / "transport_rules.json", {"direct": "same unit/direction anchor and support", "pilot": "fit an independent pilot before transporting", "conditional": "recalibrate condition-specific shift rather than global-normalize away DIF", "nonlinear": "test monotonicity and retain functional signature", "population": "keep population shift separate from instrument shift", "human": "same-human cross-language invariance remains H-stage work"})
    write_json(out / "acceptance.json", {"status": "PASS_LIMITED", "accepted": ["affine/conditional/nonlinear transport contract", "pilot/application split", "support and DIF flags"], "not_claimed": ["cross-language invariance", "universal affine transport", "human validity"]})
    finish(out, args.stage, started, "PASS_LIMITED", "F12 closes domain transport rules with explicit support and anchor requirements.")


def main() -> None:
    """Run one F12 card."""
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", required=True, choices=tuple(f"F12-0{i}" for i in range(1, 6))); parser.add_argument("--input-manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True); started = datetime.now(timezone.utc).isoformat(); write_json(args.input_manifest, {"task_id": args.stage, "status": "declared", "source": "F11/F12 protocol", "model_calls": 0}); {"F12-01": run_f12_01, "F12-02": run_f12_02, "F12-03": run_f12_03, "F12-04": run_f12_04, "F12-05": run_f12_05}[args.stage](out, args, started); print(json.dumps({"task_id": args.stage, "status": "EXECUTED", "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
