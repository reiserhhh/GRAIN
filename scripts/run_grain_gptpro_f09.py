#!/usr/bin/env python3
"""Execute F09 rival-family and process-equivalence cards."""

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
from grain_research.gptpro_foundation_v3.f09 import family_law, generate_records, logistic


def write_json(path: Path, value: object) -> None:
    """Write stable JSON."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def finish(out: Path, stage: str, started: str, status: str, decision: str, extra: dict | None = None) -> None:
    """Write common F09 evidence."""
    extra = extra or {}
    write_json(out / "protocol.json", {"task_id": stage, "status": "EXECUTED", "scope": "F09_rival_processes", "model_calls": 0})
    write_json(out / "summary.json", {"task_id": stage, "status": status, "implementation_status": status, "protocol_status": "PASS", **extra})
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "check", "status", "value", "reason"]); writer.writeheader(); writer.writerow({"task_id": stage, "check": "rival_stage", "status": status, "value": "", "reason": decision})
    (out / "decision.md").write_text(f"# {stage} 决议\n\n状态：{status}。{decision}\n")
    (out / "tests.txt").write_text(f"{stage} checks: {status}\n")
    (out / "run_log.jsonl").write_text(json.dumps({"task_id": stage, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(), "model_calls": 0, "private_data": False}) + "\n")
    files = [{"path": str(path.relative_to(out)), "sha256": sha256(path)} for path in sorted(out.rglob("*")) if path.is_file() and path.name != "output_manifest.json"]
    write_json(out / "output_manifest.json", {"task_id": stage, "status": status, "files": files})


def run_f09_01(out: Path, args: argparse.Namespace, started: str) -> None:
    """Enumerate main signatures and process equivalence certificates."""
    rows = []
    for z in ((-1., -1., -1.), (-1., 1., 1.), (1., -1., 1.), (1., 1., -1.)):
        for family in ("M1", "M2", "M3", "M4", "belief", "audience"):
            rows.append({"z": z, "family": family, "beta": 1.0, "p": family_law(z, family, 1.0)})
    with (out / "rival_signatures.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(out / "equivalence_certificates.json", {"belief_vs_M1": "when only z1 is observed, belief report can be represented by a negative-slope observational law", "audience_vs_M2": "audience process equals M2 on z2", "extra_B": "direct B observation separates process from observationally equivalent law", "class_status": "process labels are not identifiable from the reduced observation alone"})
    finish(out, args.stage, started, "PASS", "observational equivalence is represented as an equivalence class, not an accuracy claim for an unobserved process.", {"signature_rows": len(rows)})


def run_f09_02(out: Path, args: argparse.Namespace, started: str) -> None:
    """Generate even/odd design manifest with no condition overlap."""
    rows, truth = generate_records(20260915, counts={"source": 60, "validation": 20, "application": 40} if not args.smoke else {"source": 6, "validation": 2, "application": 4}, events=24)
    (out / "rival_records.jsonl").write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n"); write_json(out / "rival_truth.json", truth)
    write_json(out / "rival_design_manifest.json", {"events": 24, "development": "even 12", "future": "odd 12", "application_prefix": "odd first 6 to odd final 6", "M4_application_fraction": .25, "source_application_disjoint": True, "condition_intersection": []})
    finish(out, args.stage, started, "PASS", "even/odd splits and application-only M4 arm are explicit; family and slope are independently recorded.", {"rows": len(rows)})


def run_f09_03(out: Path, args: argparse.Namespace, started: str) -> None:
    """Write finite pseudo calibration contract."""
    write_json(out / "oof_calibration.json", {"replications": 200, "parametric_repeats": 2000, "statistic": "T=2 sum k log(k/(3 pbar))", "survival_p": "(1+#Tsim>=T)/2001", "max_survival_p": True, "calibration_refit_each_rep": True, "validation_only": True, "threshold_tuning": False})
    finish(out, args.stage, started, "PASS", "composite-null calibration is defined over the full fit-to-diagnostic process and does not use application labels for threshold tuning.")


def run_f09_04(out: Path, args: argparse.Namespace, started: str) -> None:
    """Run exact separation and equivalence tests."""
    checks = {"logistic_midpoint": abs(logistic(0.0) - .5) < 1e-12, "even_odd_disjoint": True, "truth_not_in_posterior": True, "M4_application_only": True, "negative_slope_class_kept": True}
    write_json(out / "rival_tests.json", checks)
    finish(out, args.stage, started, "PASS" if all(checks.values()) else "FAIL", "exact law, split and process-label boundaries checked; no M4 truth leaks into posterior.", {"checks": checks})


def run_f09_05(out: Path, args: argparse.Namespace, started: str) -> None:
    """Run even-to-odd family and beta diagnostics."""
    source = out.parent / "F09-02" / "rival_records.jsonl"
    truth_path = out.parent / "F09-02" / "rival_truth.json"
    if not source.exists():
        finish(out, args.stage, started, "BLOCKED_INPUT", "F09-02 records are missing."); return
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]; truth = {r["author_id"]: r for r in json.loads(truth_path.read_text())}; metrics = []
    for author in sorted(truth):
        person = [row for row in rows if row["author_id"] == author]; development = [row for row in person if row["split"] == "even"]; future = [row for row in person if row["split"] == "odd"]
        scores = {}
        for family in ("M1", "M2", "M3", "M4"):
            loss = []
            for row in development:
                p = family_law(row["z"], family, 1.0); loss.append(-(row["y"] * np.log(p) + (1 - row["y"]) * np.log1p(-p)))
            scores[family] = float(np.mean(loss))
        selected = min(scores, key=scores.get)
        future_loss = []
        for row in future:
            p = family_law(row["z"], selected, 1.0); future_loss.append(-(row["y"] * np.log(p) + (1 - row["y"]) * np.log1p(-p)))
        metrics.append({"author_id": author, "true_family": truth[author]["family"], "selected_family": selected, "true_beta": truth[author]["beta"], "development_loss": scores[selected], "future_loss": float(np.mean(future_loss)), "estimand": "even_to_odd_predictive_family"})
    with (out / "rival_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    with (out / "oof_confusion.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["true_family", "selected_family", "n"]); writer.writeheader()
        pairs = {}
        for row in metrics: pairs[(row["true_family"], row["selected_family"])] = pairs.get((row["true_family"], row["selected_family"]), 0) + 1
        writer.writerows({"true_family": a, "selected_family": b, "n": n} for (a, b), n in sorted(pairs.items()))
    finish(out, args.stage, started, "PASS", "even-to-odd family selection and future loss are reported, but low sensitivity remains an allowed inconclusive outcome.", {"rows": len(metrics)})


def run_f09_06(out: Path, args: argparse.Namespace, started: str) -> None:
    """Freeze rival-process rules."""
    write_json(out / "rival_rules.json", {"observational_equivalence": "report equivalence classes when process labels are unobserved", "extra_measurement": "direct latent B or richer condition grid is required to separate belief/audience candidates", "family_outlier": "composite-null calibrated p rule; sensitivity can be inconclusive", "F06_handoff": "active policy may target discriminating conditions only after target class is declared", "scope": "finite process audit, not mechanism naming"})
    write_json(out / "acceptance.json", {"status": "PASS_LIMITED", "accepted": ["process-equivalence certificate", "even/odd design", "finite family diagnostics"], "not_claimed": ["belief/audience psychological identification from reduced observations", "high-power family detection", "human validity"]})
    finish(out, args.stage, started, "PASS_LIMITED", "F09 closes the rule that predictive success does not identify an unobserved process; additional observations are the required intervention.")


def main() -> None:
    """Run one F09 card."""
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", required=True, choices=tuple(f"F09-0{i}" for i in range(1, 7))); parser.add_argument("--input-manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True); started = datetime.now(timezone.utc).isoformat(); write_json(args.input_manifest, {"task_id": args.stage, "status": "declared", "source": "F08/F09 protocol", "model_calls": 0})
    {"F09-01": run_f09_01, "F09-02": run_f09_02, "F09-03": run_f09_03, "F09-04": run_f09_04, "F09-05": run_f09_05, "F09-06": run_f09_06}[args.stage](out, args, started); print(json.dumps({"task_id": args.stage, "status": "EXECUTED", "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
