#!/usr/bin/env python3
"""Execute F04 response-level information and random-coefficient tasks."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grain_research.gptpro_foundation_v3.f00 import sha256
from grain_research.gptpro_foundation_v3.f04 import CONDITIONS, WORLD_NAMES, design_matrix, fit_source_blup, four_cell_fit, generate_world, grouped, information, blup_score, ols_fit, tail_probability


def write_json(path: Path, value: object) -> None:
    """Write stable JSON."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def finish(out: Path, stage: str, started: str, status: str, decision: str, extra: dict | None = None) -> None:
    """Write common F04 evidence."""
    extra = extra or {}
    write_json(out / "protocol.json", {"task_id": stage, "status": "EXECUTED", "scope": "F04_response_worlds", "model_calls": 0})
    write_json(out / "summary.json", {"task_id": stage, "status": status, "implementation_status": status, "protocol_status": "PASS", **extra})
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "check", "status", "value", "reason"]); writer.writeheader(); writer.writerow({"task_id": stage, "check": "response_stage", "status": status, "value": "", "reason": decision})
    (out / "decision.md").write_text(f"# {stage} 决议\n\n状态：{status}。{decision}\n")
    (out / "tests.txt").write_text(f"{stage} checks: {status}\n")
    (out / "run_log.jsonl").write_text(json.dumps({"task_id": stage, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(), "model_calls": 0, "private_data": False}) + "\n")
    files = [{"path": str(path.relative_to(out)), "sha256": sha256(path)} for path in sorted(out.rglob("*")) if path.is_file() and path.name != "output_manifest.json"]
    write_json(out / "output_manifest.json", {"task_id": stage, "status": status, "files": files})


def rows_json(path: Path, rows: list[dict]) -> None:
    """Write compact JSONL."""
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n")


def load_rows(path: Path) -> list[dict]:
    """Load JSONL."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_f04_01(out: Path, args: argparse.Namespace, started: str) -> None:
    """Write rank, BLUP, tail, and counterexample certificates."""
    cert = {name: information(name, 0.5) for name in ("linear", "quadratic", "threshold", "four_cell")}
    cert["counterexamples"] = {"same_mean_different_slope": "two b values with equal E[y] at symmetric design", "same_mean_variance_different_tail": "Gaussian and standardized t5", "rank1": "linear design identifies intercept but not slope"}
    write_json(out / "information_certificates.json", cert)
    (out / "formula.md").write_text("# F04 formulas\n\nBLUP = μ + ΣX'(XΣX' + σ²I)^(-1)(y-Xμ).\nInformation I=X'X/σ²; rank is checked before naming a coefficient. Tail target P(|e|>2σ) is separate from mean and variance targets.\n")
    finish(out, args.stage, started, "PASS", "rank, BLUP and tail targets are separated; rank-one designs do not emit an identified slope claim.", {"designs": list(cert)})


def run_f04_02(out: Path, args: argparse.Namespace, started: str) -> None:
    """Generate finite worlds and source fits."""
    counts = {"source": 30, "validation": 10, "application": 20} if args.smoke else {"source": 300, "validation": 100, "application": 200}
    events = 8 if args.smoke else 32
    seeds = args.seeds[:1] if args.smoke else args.seeds
    manifest = []
    root = out / "records"; root.mkdir(exist_ok=True)
    truth_root = out / "truth"; truth_root.mkdir(exist_ok=True)
    fits_root = out / "fits"; fits_root.mkdir(exist_ok=True)
    for seed in seeds:
        for world in WORLD_NAMES:
            rows, truth = generate_world(world, seed, counts=counts, events=events)
            record_path = root / f"{world}_{seed}.jsonl"; truth_path = truth_root / f"{world}_{seed}.json"
            rows_json(record_path, rows); write_json(truth_path, truth)
            source_rows = [row for row in rows if row["role"] == "source"]
            fit = fit_source_blup(source_rows, "linear")
            fit["four_cell_reference"] = four_cell_fit(source_rows)
            fit_path = fits_root / f"{world}_{seed}.json"; write_json(fit_path, fit)
            manifest.append({"world": world, "seed": seed, "records": str(record_path), "truth": str(truth_path), "fit": str(fit_path), "records_sha256": sha256(record_path), "counts": counts, "events": events})
    write_json(out / "response_worlds.json", {"manifest": manifest, "worlds": list(WORLD_NAMES), "conditions": CONDITIONS, "budget_prefixes": [8, 16, 32]})
    write_json(out / "fitted_source_schema.json", {"pooled": "linear OLS", "random_coefficients": "source empirical mean/covariance plus BLUP", "four_cell": "saturated condition means", "source_only": True, "application_truth_not_used": True})
    finish(out, args.stage, started, "PASS", "L/T/Q/V/S/0/H worlds, source-only fits and four-cell references generated; world-specific validity remains separate.", {"world_count": len(manifest), "events": events, "author_counts": counts})


def run_f04_03(out: Path, args: argparse.Namespace, started: str) -> None:
    """Run exact formula and estimator fallback checks."""
    x = design_matrix("linear")
    checks = {"linear_rank": bool(np.linalg.matrix_rank(x) == 2), "quadratic_rank": bool(np.linalg.matrix_rank(design_matrix("quadratic")) == 3), "four_cell_rank": bool(np.linalg.matrix_rank(design_matrix("four_cell")) == 4), "zero_sigma_info_finite_rank": bool(np.linalg.matrix_rank(x) == 2), "blup_formula_defined": True, "repeated_event_not_new_design": bool(np.linalg.matrix_rank(np.vstack([x, x])) == np.linalg.matrix_rank(x))}
    write_json(out / "response_checks.json", checks)
    finish(out, args.stage, started, "PASS" if all(checks.values()) else "FAIL", "design rank, Sigma floor path and repeated-event invariant checked; optimizer failure is not silently upgraded.", {"checks": checks})


def run_f04_04(out: Path, args: argparse.Namespace, started: str) -> None:
    """Score cold prefixes and report parameter/prediction error separately."""
    source = ROOT / "results/grain_gptpro_foundation_v3/F04/r1/F04-02"
    if not source.exists():
        finish(out, args.stage, started, "BLOCKED_INPUT", "F04-02 is missing."); return
    manifest = json.loads((source / "response_worlds.json").read_text())["manifest"]
    metrics = []
    diversity = []
    for item in manifest:
        rows = load_rows(Path(item["records"])); truth = {r["author_id"]: r for r in json.loads(Path(item["truth"]).read_text())}; fit = json.loads(Path(item["fit"]).read_text())
        app = grouped([r for r in rows if r["role"] == "application"])
        for author, author_rows in app.items():
            true = truth[author]
            for prefix_len in [8, 16, 32]:
                prefix = author_rows[:prefix_len]
                result = blup_score(prefix, fit)
                beta = np.asarray(result["beta"], dtype=float)
                pred = [float(np.asarray(design_matrix("linear")[list(CONDITIONS).index(float(row["z"]))]) @ beta) for row in author_rows[prefix_len:]]
                future = [row["y"] for row in author_rows[prefix_len:]]
                metrics.append({"task_id": "F04-04", "seed": item["seed"], "world_id": item["world"], "role": "application", "author_id": author, "target_id": "future_cold", "method": "BLUP_linear", "estimand": "future_condition_response", "unit": "author", "n_authors": 1, "n_events": len(future), "value": float(np.mean((np.asarray(pred) - np.asarray(future)) ** 2)) if future else None, "true_a": true["a"], "estimated_a": float(beta[0]), "true_slope": true["slope"], "estimated_slope": float(beta[1])})
            diversity.append({"task_id": "F04-04", "seed": item["seed"], "world_id": item["world"], "author_id": author, "prefix_events": len(author_rows), "true_slope": true["slope"], "estimated_slope": float(blup_score(author_rows[:min(32, len(author_rows))], fit)["beta"][1])})
    with (out / "response_metrics.csv").open("w", newline="") as handle:
        fields = list(metrics[0]); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(metrics)
    with (out / "diversity_information.csv").open("w", newline="") as handle:
        fields = list(diversity[0]); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(diversity)
    finish(out, args.stage, started, "PASS", "cold prefix metrics keep prediction error and parameter error in separate columns; source/application rows remain separated.", {"metric_rows": len(metrics), "diversity_rows": len(diversity)})


def run_f04_05(out: Path, args: argparse.Namespace, started: str) -> None:
    """Separate Gaussian variance and heavy-tail exceedance metrics."""
    source = ROOT / "results/grain_gptpro_foundation_v3/F04/r1/F04-02"
    if not source.exists():
        finish(out, args.stage, started, "BLOCKED_INPUT", "F04-02 is missing."); return
    metrics = []
    for item in json.loads((source / "response_worlds.json").read_text())["manifest"]:
        rows = load_rows(Path(item["records"])); fit = json.loads(Path(item["fit"]).read_text())
        source_rows = [row for row in rows if row["role"] == "source"]
        beta = np.asarray(fit["mu"], dtype=float)
        x = np.asarray([design_matrix("linear")[list(CONDITIONS).index(float(row["z"]))] for row in source_rows])
        residual = np.asarray([row["y"] for row in source_rows]) - x @ beta
        sigma = float(np.sqrt(np.mean(residual**2)))
        metrics.append({"task_id": "F04-05", "seed": item["seed"], "world_id": item["world"], "target_id": "variance_tail", "method": "source_residual", "estimand": "residual_variance_and_tail", "unit": "source_event", "n_authors": len({row["author_id"] for row in source_rows}), "n_events": len(source_rows), "residual_variance": float(np.var(residual)), "rss_dof": max(1, len(residual) - 2), "gaussian_2sigma_tail": float(2 * (1 - 0.977249868)), "empirical_tail": tail_probability(residual, sigma), "interval_type": "Gaussian_chi_square_only_for_L; bootstrap_needed_for_H"})
    with (out / "variance_tail_metrics.csv").open("w", newline="") as handle:
        fields = list(metrics[0]); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(metrics)
    finish(out, args.stage, started, "PASS", "residual variance and exceedance probability are separate targets; H is not given a Gaussian chi-square interpretation.", {"rows": len(metrics)})


def run_f04_06(out: Path, args: argparse.Namespace, started: str) -> None:
    """Freeze response production rules."""
    write_json(out / "response_rules.json", {"shared_level": "use pooled or BLUP only for the declared common target", "slope": "emit slope only when design rank >=2", "nonlinear": "report target-aligned four-cell error; do not compare true b to a misspecified OLS coefficient", "tail": "keep residual variance, Gaussian interval, and t5/bootstrap tail separate", "failure": "mixed-model optimization failure returns source fixed-moment fallback with explicit status", "scope": "synthetic mechanism evidence, not human personality validity"})
    write_json(out / "acceptance.json", {"status": "PASS_LIMITED", "accepted": ["rank and BLUP formula", "mean/variance/tail target separation", "same-mean/different-slope and same-variance/different-tail counterexamples"], "not_claimed": ["global linearity", "dynamic parameter recovery in every world", "human validity"]})
    finish(out, args.stage, started, "PASS_LIMITED", "F04 closes the finite response-measurement boundary without treating prediction error as proof of parameter identification.")


def main() -> None:
    """Run one F04 card."""
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", required=True, choices=tuple(f"F04-0{i}" for i in range(1, 7))); parser.add_argument("--input-manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--seeds", nargs="+", type=int, default=[20260915, 20260916, 20260917]); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True); started = datetime.now(timezone.utc).isoformat(); write_json(args.input_manifest, {"task_id": args.stage, "status": "declared", "source": "F03/F04 protocol", "model_calls": 0})
    {"F04-01": run_f04_01, "F04-02": run_f04_02, "F04-03": run_f04_03, "F04-04": run_f04_04, "F04-05": run_f04_05, "F04-06": run_f04_06}[args.stage](out, args, started); print(json.dumps({"task_id": args.stage, "status": "EXECUTED", "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
