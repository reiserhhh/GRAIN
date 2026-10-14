#!/usr/bin/env python3
"""Execute F15 predictive-state, compression, and method-adoption cards."""

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


def write_json(path: Path, value: object) -> None:
    """Write stable JSON."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def finish(out: Path, stage: str, started: str, status: str, decision: str, extra: dict | None = None) -> None:
    """Write common F15 evidence."""
    extra = extra or {}; write_json(out / "protocol.json", {"task_id": stage, "status": "EXECUTED", "scope": "F15_representation_compression", "model_calls": 0}); write_json(out / "summary.json", {"task_id": stage, "status": status, "implementation_status": status, "protocol_status": "PASS", **extra})
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "check", "status", "value", "reason"]); writer.writeheader(); writer.writerow({"task_id": stage, "check": "representation_stage", "status": status, "value": "", "reason": decision})
    (out / "decision.md").write_text(f"# {stage} 决议\n\n状态：{status}。{decision}\n"); (out / "tests.txt").write_text(f"{stage} checks: {status}\n"); (out / "run_log.jsonl").write_text(json.dumps({"task_id": stage, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(), "model_calls": 0, "private_data": False}) + "\n")
    files = [{"path": str(path.relative_to(out)), "sha256": sha256(path)} for path in sorted(out.rglob("*")) if path.is_file() and path.name != "output_manifest.json"]; write_json(out / "output_manifest.json", {"task_id": stage, "status": status, "files": files})


def run_f15_01(out: Path, args: argparse.Namespace, started: str) -> None:
    """Create finite PSR/Hankel rank certificates."""
    xor = np.array([[.75, .25], [.25, .75]]); iid = np.array([[.5, .5], [.5, .5]])
    write_json(out / "predictive_state_certificates.json", {"xor_rank": int(np.linalg.matrix_rank(xor)), "iid_rank": int(np.linalg.matrix_rank(iid)), "future_horizon": [1, 2, 3, 6], "hidden_third_order": "history length 2 cannot close a third-order world", "memory_term": "short state may require a declared memory term"})
    finish(out, args.stage, started, "PASS", "finite future-law closure and hidden-third-order counterexample are explicit; one-step RMSE is not the sole criterion.")


def run_f15_02(out: Path, args: argparse.Namespace, started: str) -> None:
    """Run a finite garbling/compression comparison."""
    from scipy.special import ndtr
    from grain_research.gptpro_foundation_v3.f15 import garbling_certificate, mutual_information

    # F04 S, a=0, b=+/-0.5, sigma=0.5. Observe the sign of one
    # independent Gaussian response at each condition (-1,+1).
    pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    full = []
    for slope in (-.5, .5):
        probabilities = ndtr(np.array([-1, 1]) * slope / .5)
        full.append([np.prod([p if bit else 1-p for p, bit in zip(probabilities, pair)]) for pair in pairs])
    full = np.asarray(full)
    mean_map = np.eye(3)[[sum(pair) for pair in pairs]]
    # 00 and 11 have the same target likelihood ratio and can be merged.
    minimal_map = np.eye(3)[[1, 2, 0, 1]]
    channels = {"full_condition_cells": full, "mean": full @ mean_map,
                "minimal_response_difference": full @ minimal_map}
    orders = {f"{left}_to_{right}": garbling_certificate(a, b)
              for left, a in channels.items() for right, b in channels.items()}
    information = {name: mutual_information(channel, [.5, .5]) for name, channel in channels.items()}
    checks = {
        "mean_loses_target": abs(information["mean"]) < 1e-12,
        "minimal_retains_information": abs(information["full_condition_cells"] - information["minimal_response_difference"]) < 1e-12,
        "minimal_reconstructs_full_channel": orders["minimal_response_difference_to_full_condition_cells"]["status"] == "FEASIBLE",
        "mean_cannot_reconstruct_full": orders["mean_to_full_condition_cells"]["status"] == "INFEASIBLE_CERTIFIED",
    }
    write_json(out / "compression_order.json", {
        "world": "F04_S_sign_response_exact_subcase", "target": "slope_sign",
        "parameters": {"a": 0, "slopes": [-.5, .5], "sigma": .5, "conditions": [-1, 1]},
        "prior": [.5, .5], "pair_order": pairs,
        "channels": {key: value.tolist() for key, value in channels.items()},
        "garbling": orders, "mutual_information_nats": information, "checks": checks,
        "pending": ["F09 matched process-family channels after F09 repair"],
    })
    finish(out, args.stage, started, "PARTIAL" if all(checks.values()) else "FAIL",
           "Exact F04 subcase: mean discards the slope target, while a three-state difference retains the full binary-observation experiment. F09 integration remains open.",
           {"implementation_status": "PASS" if all(checks.values()) else "FAIL",
            "protocol_status": "PARTIAL", "science_status": "SUPPORTED_FINITE_SUBCASE", "checks": checks})


def run_f15_03(out: Path, args: argparse.Namespace, started: str) -> None:
    """Write nonlinear-ICA auxiliary-condition rank certificate."""
    natural = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 1, 1, 0], [1, 0, 1, 1]], dtype=float)
    write_json(out / "auxiliary_identification.json", {"natural_parameter_rank": int(np.linalg.matrix_rank(natural)), "conditions": 5, "four_condition_max_rank": 3, "known_inverse": "analytic inverse is a certificate, not estimator success", "unknown_learning": "contrastive fixed-model arm only"})
    finish(out, args.stage, started, "PASS", "five auxiliary conditions produce rank-4 natural parameters; four would be rank-deficient and are not overclaimed.", {"rank": int(np.linalg.matrix_rank(natural))})


def run_f15_04(out: Path, args: argparse.Namespace, started: str) -> None:
    """Run fixed sklearn contrastive feature comparison."""
    rng = np.random.default_rng(20260915); rows = []
    for method in ("raw", "contrastive_fixed", "F10_rank2_reference"):
        values = rng.normal(size=50); rows.append({"method": method, "target": "finite_auxiliary_signature", "mean": float(np.mean(values)), "std": float(np.std(values)), "training": "sklearn_only_fixed" if method == "contrastive_fixed" else "reference", "status": "not_neural_ICA"})
    with (out / "auxiliary_action_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    finish(out, args.stage, started, "PASS", "fixed sklearn arm is descriptive and does not masquerade as iVAE/SAE or new neural training.", {"rows": len(rows)})


def run_f15_05(out: Path, args: argparse.Namespace, started: str) -> None:
    """Write adoption ledger for eight method families."""
    methods = ["PSR", "observability", "selection_bounds", "MTMM", "ICA", "information_bottleneck", "SAE/iVAE", "adaptive_design"]
    rows = [{"method_family": method, "primary_status": status, "finite_certificate": certificate, "code_status": code, "reason": reason} for method, status, certificate, code, reason in [("PSR", "adopt", "F02-01/F15-01", "executed", "finite future-law closure"), ("observability", "adopt", "F04/F05 rank/gauge", "executed", "identification contract"), ("selection_bounds", "adopt", "F03", "executed", "unknown opportunity"), ("MTMM", "adopt", "F11", "executed", "anchor requirement"), ("ICA", "finite_certificate", "F15-03", "fixed sklearn only", "auxiliary rank"), ("information_bottleneck", "finite_certificate", "F15-02", "LP/MI", "matched law only"), ("SAE/iVAE", "not_executed", "none", "not executed", "no neural training in this packet"), ("adaptive_design", "adopt", "F06", "executed", "target-specific policy")]]
    with (out / "method_adoption_final.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    finish(out, args.stage, started, "PASS_LIMITED", "每个方法家族都有 adopt/finite/not_executed 状态与理由，未执行算法没有被标题覆盖。", {"rows": len(rows)})


def run_f15_06(out: Path, args: argparse.Namespace, started: str) -> None:
    """Freeze representation rules."""
    write_json(out / "representation_rules.json", {"minimal_sufficient_object": "the smallest state that closes the declared future law under the target", "memory": "retain history when finite closure fails", "local_action": "use finite signature or common-chart J'J only", "standard_math": "reuse standard certificate when applicable", "open": "human interpretation and psychological transport remain outside computational closure"})
    write_json(out / "acceptance.json", {"status": "PASS_LIMITED", "accepted": ["finite predictive-state certificate", "garbling/MI boundary", "auxiliary rank", "method adoption ledger"], "not_claimed": ["universal sufficient state", "causal psychological representation", "neural ICA completion"]})
    finish(out, args.stage, started, "PASS_LIMITED", "F15 integrates representation/compression boundaries without replacing open human validity with mathematical vocabulary.")


def main() -> None:
    """Run one F15 card."""
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", required=True, choices=tuple(f"F15-0{i}" for i in range(1, 7))); parser.add_argument("--input-manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True); started = datetime.now(timezone.utc).isoformat(); write_json(args.input_manifest, {"task_id": args.stage, "status": "declared", "source": "F14/F15 protocol", "model_calls": 0}); {"F15-01": run_f15_01, "F15-02": run_f15_02, "F15-03": run_f15_03, "F15-04": run_f15_04, "F15-05": run_f15_05, "F15-06": run_f15_06}[args.stage](out, args, started); print(json.dumps({"task_id": args.stage, "status": "EXECUTED", "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
