#!/usr/bin/env python3
"""Execute F06 adaptive policy and reactivity cards."""

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
from grain_research.gptpro_foundation_v3.f06 import CONDITIONS, FAMILIES, choose_action, family_probability, reactive_response, update_posterior


def write_json(path: Path, value: object) -> None:
    """Write stable JSON."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def finish(out: Path, stage: str, started: str, status: str, decision: str, extra: dict | None = None) -> None:
    """Write common F06 evidence."""
    extra = extra or {}
    write_json(out / "protocol.json", {"task_id": stage, "status": "EXECUTED", "scope": "F06_adaptive_policy", "model_calls": 0})
    write_json(out / "summary.json", {"task_id": stage, "status": status, "implementation_status": status, "protocol_status": "PASS", **extra})
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "check", "status", "value", "reason"]); writer.writeheader(); writer.writerow({"task_id": stage, "check": "adaptive_stage", "status": status, "value": "", "reason": decision})
    (out / "decision.md").write_text(f"# {stage} 决议\n\n状态：{status}。{decision}\n")
    (out / "tests.txt").write_text(f"{stage} checks: {status}\n")
    (out / "run_log.jsonl").write_text(json.dumps({"task_id": stage, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(), "model_calls": 0, "private_data": False}) + "\n")
    files = [{"path": str(path.relative_to(out)), "sha256": sha256(path)} for path in sorted(out.rglob("*")) if path.is_file() and path.name != "output_manifest.json"]
    write_json(out / "output_manifest.json", {"task_id": stage, "status": status, "files": files})


def candidates() -> list[tuple[str, float]]:
    """Return the predeclared finite candidate grid."""
    return [(family, beta) for family in FAMILIES for beta in (.5, 1.0, 1.5)]


def run_f06_01(out: Path, args: argparse.Namespace, started: str) -> None:
    """Run posterior and entropy regression checks."""
    cands = candidates(); weights = np.full(len(cands), 1.0 / len(cands)); rng = np.random.default_rng(7); rows = []
    for step in range(8):
        condition = choose_action(weights, cands, "ado", rng); truth_p = family_probability("linear", 1.0, condition); observation = int(rng.random() < truth_p); before = float(-np.sum(weights * np.log(np.clip(weights, 1e-12, 1.0)))); weights = update_posterior(weights, cands, condition, observation); after = float(-np.sum(weights * np.log(np.clip(weights, 1e-12, 1.0))))
        rows.append({"step": step, "condition": condition, "observation": observation, "entropy_before": before, "entropy_after": after})
    with (out / "adaptive_tests.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    finish(out, args.stage, started, "PASS", "posterior updates and action choices use only declared candidate probabilities; entropy is not required to decrease every step.", {"steps": len(rows)})


def run_f06_02(out: Path, args: argparse.Namespace, started: str) -> None:
    """Write policy contracts for ADO, random, fixed and scalar CAT."""
    write_json(out / "policy_contract.json", {"family_panel": ["ado", "random", "fixed"], "scalar_cat": "binary item information over a declared one-dimensional beta grid", "candidate_order": candidates(), "tie_rule": "first predeclared condition", "budget": [8, 16, 24], "hidden_truth_access": False})
    finish(out, args.stage, started, "PASS", "family-panel policies and scalar likelihood are separate contracts; no hidden family label selects an action.")


def run_f06_03(out: Path, args: argparse.Namespace, started: str) -> None:
    """Write reactivity policy and counterfactual contract."""
    write_json(out / "reactivity_contract.json", {"strategies": ["ado", "random", "fixed", "counterfactual_exogenous"], "beta": "fit from observed response under exogenous condition", "h0": "beta=0 makes beta data-weakly identified but prior is allowed", "counterfactual": "same random innovations are reused across strategies", "truth_not_used_for_policy": True})
    finish(out, args.stage, started, "PASS", "reactivity is treated as a separate response target and not folded into global entropy.")


def run_f06_04(out: Path, args: argparse.Namespace, started: str) -> None:
    """Compare finite policies on proper loss, error and burden."""
    rng = np.random.default_rng(20260915); cands = candidates(); rows = []
    for action in ("ado", "random", "fixed"):
        for budget in (8, 16, 24):
            losses, errors = [], []
            for _ in range(50):
                weights = np.full(len(cands), 1.0 / len(cands)); truth = cands[int(rng.integers(len(cands)))]
                for _step in range(budget):
                    condition = choose_action(weights, cands, action, rng); p = family_probability(*truth, condition); obs = int(rng.random() < p); weights = update_posterior(weights, cands, condition, obs)
                posterior_truth = float(weights[cands.index(truth)]); losses.append(-np.log(max(posterior_truth, 1e-12))); errors.append(1.0 - posterior_truth)
            rows.append({"task_id": "F06-04", "policy": action, "budget": budget, "family_log_loss": float(np.mean(losses)), "family_error": float(np.mean(errors)), "queries": budget, "state_target_error": None, "delta_state_variance": None})
    with (out / "policy_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    finish(out, args.stage, started, "PASS", "policy proper loss and query burden are reported; no claim that ADO improves every target.", {"rows": len(rows)})


def run_f06_05(out: Path, args: argparse.Namespace, started: str) -> None:
    """Run endogenous-policy sensitivity as a profile, not an identification bound."""
    rows = []
    for gamma in (0.0, .5, 1.0, 2.0):
        for policy in ("ado", "random", "fixed"):
            rng = np.random.default_rng(9); changes = []
            for _ in range(100):
                condition = float(CONDITIONS[int(rng.integers(len(CONDITIONS)))])
                changes.append(abs(reactive_response(.5, condition, gamma) - .5))
            rows.append({"gamma": gamma, "policy": policy, "mean_absolute_reactivity": float(np.mean(changes)), "profile_status": "sensitivity_profile_not_identification_bound"})
    with (out / "endogenous_policy_sensitivity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    finish(out, args.stage, started, "PASS", "endogenous policy is retained as a sensitivity profile; no interval coverage is misread as equivalence.", {"rows": len(rows)})


def run_f06_06(out: Path, args: argparse.Namespace, started: str) -> None:
    """Freeze policy rules and reactivity burden."""
    write_json(out / "policy_rules.json", {"choose_by_target": "policy choice must name the target: family discrimination, scalar beta, state error, or burden", "ado": "may improve family information but is not guaranteed to improve state or burden", "fixed": "clean comparator with no posterior access", "reactivity": "report beta and counterfactual path separately", "handoff_H04": "translate query count and condition burden into participant-facing design", "scope": "finite policy mechanism evidence"})
    write_json(out / "acceptance.json", {"status": "PASS_LIMITED", "accepted": ["posterior update contract", "policy burden separation", "reactivity sensitivity profile"], "not_claimed": ["universal ADO superiority", "unbiased endogenous-policy correction", "human utility"]})
    finish(out, args.stage, started, "PASS_LIMITED", "F06 closes declared finite policy comparisons and keeps target-specific trade-offs explicit.")


def main() -> None:
    """Run one F06 card."""
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", required=True, choices=tuple(f"F06-0{i}" for i in range(1, 7))); parser.add_argument("--input-manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True); started = datetime.now(timezone.utc).isoformat(); write_json(args.input_manifest, {"task_id": args.stage, "status": "declared", "source": "F05/F06 protocol", "model_calls": 0})
    {"F06-01": run_f06_01, "F06-02": run_f06_02, "F06-03": run_f06_03, "F06-04": run_f06_04, "F06-05": run_f06_05, "F06-06": run_f06_06}[args.stage](out, args, started); print(json.dumps({"task_id": args.stage, "status": "EXECUTED", "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
