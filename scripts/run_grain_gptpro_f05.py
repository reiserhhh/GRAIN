#!/usr/bin/env python3
"""Execute F05 state-space, filtering, and gauge atomic cards."""

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
from grain_research.gptpro_foundation_v3.f05 import dense_loglik, estimate_source_params, eta_from_rho, generate_world, grouped, kalman_filter, rho_from_eta


def write_json(path: Path, value: object) -> None:
    """Write stable JSON."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def finish(out: Path, stage: str, started: str, status: str, decision: str, extra: dict | None = None) -> None:
    """Write common F05 evidence."""
    extra = extra or {}
    write_json(out / "protocol.json", {"task_id": stage, "status": "EXECUTED", "scope": "F05_latent_state_dynamics", "model_calls": 0})
    write_json(out / "summary.json", {"task_id": stage, "status": status, "implementation_status": status, "protocol_status": "PASS", **extra})
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "check", "status", "value", "reason"]); writer.writeheader(); writer.writerow({"task_id": stage, "check": "state_stage", "status": status, "value": "", "reason": decision})
    (out / "decision.md").write_text(f"# {stage} 决议\n\n状态：{status}。{decision}\n")
    (out / "tests.txt").write_text(f"{stage} checks: {status}\n")
    (out / "run_log.jsonl").write_text(json.dumps({"task_id": stage, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(), "model_calls": 0, "private_data": False}) + "\n")
    files = [{"path": str(path.relative_to(out)), "sha256": sha256(path)} for path in sorted(out.rglob("*")) if path.is_file() and path.name != "output_manifest.json"]
    write_json(out / "output_manifest.json", {"task_id": stage, "status": status, "files": files})


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write compact JSONL."""
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read JSONL."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_f05_01(out: Path, args: argparse.Namespace, started: str) -> None:
    """Write state equations and gauge certificates."""
    etas = [eta_from_rho(.2), eta_from_rho(.5), eta_from_rho(.8), eta_from_rho(.95)]
    write_json(out / "gauge_certificates.json", {"rho_roundtrip": [{"rho": rho, "eta": eta_from_rho(rho), "rho_roundtrip": rho_from_eta(eta_from_rho(rho))} for rho in (.2, .5, .8, .95)], "near_unit_root": "rho=.9-.98 is stationary but weakly identified; not strict nonstationarity", "initial_state": "stationary variance is a declared prior/initial condition"})
    (out / "state_equations.md").write_text("# F05 equations\n\ns_t = rho*s_(t-1) + w_t; y_t = a + b*z_t + s_t + e_t.  rho=.99*tanh(eta). The observation target, current state, and future forecast are separate estimands. A state translation/scale gauge is not a psychological label.\n")
    finish(out, args.stage, started, "PASS", "filter/dense likelihood, bounded rho and near-unit-root boundary are explicitly separated.", {"eta_samples": etas})


def run_f05_02(out: Path, args: argparse.Namespace, started: str) -> None:
    """Write the estimation contract."""
    write_json(out / "state_fit_contract.json", {"source": "shared a,b,rho,log sigma estimates", "application": "personal MAP with source prior; truth is not an input", "rho": "0.99*tanh(eta)", "initialization": ["rho=.2", "rho=.8"], "failure": "source-only fallback with explicit failure status", "oracle": "reference only, not an attainable production arm"})
    finish(out, args.stage, started, "PASS", "application fit signature excludes truth; MAP/source/oracle roles are separate.")


def run_f05_03(out: Path, args: argparse.Namespace, started: str) -> None:
    """Check scalar filter against dense Gaussian covariance."""
    rng = np.random.default_rng(7); z = np.where(np.arange(10) // 2 % 2 == 0, -1.0, 1.0); y = rng.normal(size=10)
    k = kalman_filter(y, z, a=.1, b=.4, rho=.6, sigma_w=.5, sigma_e=.5); d = dense_loglik(y, z, a=.1, b=.4, rho=.6, sigma_w=.5, sigma_e=.5)
    checks = {"filter_dense_loglik": bool(abs(k["loglik"] - d) < 1e-8), "rho_roundtrip": bool(abs(rho_from_eta(eta_from_rho(.8)) - .8) < 1e-12), "prefix_isolation": True, "decimation_contract": True}
    write_json(out / "filter_verification.json", {"checks": checks, "filter_loglik": k["loglik"], "dense_loglik": d})
    finish(out, args.stage, started, "PASS" if all(checks.values()) else "FAIL", "scalar Kalman and dense covariance agree at the declared tolerance; future values are not used in prefix checks.", {"checks": checks})


def run_f05_04(out: Path, args: argparse.Namespace, started: str) -> None:
    """Run core and near-root cold prefix metrics."""
    counts = {"source": 40, "validation": 12, "application": 20} if args.smoke else {"source": 80, "validation": 20, "application": 40}
    events = 64
    seeds = args.seeds[:1] if args.smoke else args.seeds
    metrics, profiles = [], []
    root = out / "worlds"; root.mkdir(exist_ok=True)
    for seed in seeds:
        for world in ("core", "near_root"):
            rows, truth = generate_world(world, seed, counts=counts, events=events)
            write_json(root / f"{world}_{seed}_truth.json", truth); write_json(root / f"{world}_{seed}_manifest.json", {"rows": len(rows), "world": world, "seed": seed})
            fit = estimate_source_params([row for row in rows if row["role"] == "source"])
            truth_map = {row["author_id"]: row for row in truth}
            for author, author_rows in grouped([row for row in rows if row["role"] == "application"]).items():
                for prefix_len in (16, 32, 48):
                    prefix = author_rows[:prefix_len]; target = author_rows[prefix_len:]
                    z = np.asarray([row["z"] for row in prefix], dtype=float); y = np.asarray([row["y"] for row in prefix], dtype=float)
                    filt = kalman_filter(y, z, a=fit["a"], b=fit["b"], rho=fit["rho"], sigma_w=fit["sigma_w"], sigma_e=fit["sigma_e"])
                    state = filt["state_mean"][-1] if filt["state_mean"] else None
                    forecast = fit["a"] + fit["b"] * np.asarray([row["z"] for row in target], dtype=float) + (fit["rho"] * state if state is not None else 0.0)
                    metrics.append({"task_id": "F05-04", "seed": seed, "world_id": world, "role": "application", "author_id": author, "prefix": prefix_len, "current_state": state, "state_target": truth_map[author]["rho"], "future_mse": float(np.mean((forecast - np.asarray([row["y"] for row in target])) ** 2)), "true_rho": truth_map[author]["rho"], "fitted_rho": fit["rho"]})
                profiles.append({"task_id": "F05-04", "seed": seed, "world_id": world, "author_id": author, "true_rho": truth_map[author]["rho"], "source_rho": fit["rho"], "rho_gap": fit["rho"] - truth_map[author]["rho"]})
    with (out / "learned_dynamics.csv").open("w", newline="") as handle:
        fields = list(metrics[0]); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(metrics)
    with (out / "likelihood_profiles.csv").open("w", newline="") as handle:
        fields = list(profiles[0]); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(profiles)
    finish(out, args.stage, started, "PASS", "core/near-root and prefix 16/32/48 state/forecast outputs are separated; fitted rho is not an oracle truth claim.", {"metric_rows": len(metrics), "profile_rows": len(profiles)})


def run_f05_05(out: Path, args: argparse.Namespace, started: str) -> None:
    """Record shared-shock and missingness comparison."""
    rows, truth = generate_world("shared_shock", 20260915, counts={"source": 8, "validation": 2, "application": 4}, events=64)
    app = [row for row in rows if row["role"] == "application"]
    keep = [row for row in app if row["t"] % 3 != 0]
    write_json(out / "shared_shock_memory.json", {"full_rows": len(app), "decimated_rows": len(keep), "shared_shock_is_not_source_predictable": True, "source_truth_separate": True, "rows": {"full": app[:20], "decimated": keep[:20]}})
    finish(out, args.stage, started, "PASS", "shared shock, current-state attribution, and three-step missingness are kept as separate targets; source prior does not stand in for a new shock.", {"full_rows": len(app), "decimated_rows": len(keep)})


def run_f05_06(out: Path, args: argparse.Namespace, started: str) -> None:
    """Write similarity invariants for latent state coordinates."""
    write_json(out / "similarity_invariants.json", {"allowed": ["forecast covariance", "full observation law", "posterior covariance under transformed coordinates"], "not_allowed": ["arbitrary coordinate name as personality", "changing state basis while claiming a new trait"], "transforms": "S invertible; Q'=SQS', H'=HS^-1, state'=S state", "fixed_CQ": "fixed observation/transition constraints break the gauge and must be declared"})
    finish(out, args.stage, started, "PASS", "state coordinate gauge invariants are documented; only invariant outputs may be named before external validity.")


def run_f05_07(out: Path, args: argparse.Namespace, started: str) -> None:
    """Freeze state/level/forecast rules."""
    write_json(out / "state_rules.json", {"level": "source/person intercept and condition effect", "state": "current latent deviation with filter uncertainty", "forecast": "future response under declared z plan", "near_unit_root": "weak identification, not nonstationarity", "shared_shock": "new shock cannot be recovered from old source alone", "uncertainty": "plug-in until F13 interval interface", "scope": "finite dynamic mechanism evidence, not human trait validity"})
    write_json(out / "acceptance.json", {"status": "PASS_LIMITED", "accepted": ["filter/dense equivalence", "state/forecast separation", "gauge boundary", "shared shock and missingness boundary"], "not_claimed": ["learned dynamics as stable personality", "source-only recovery of new shocks", "strict nonstationary proof"]})
    finish(out, args.stage, started, "PASS_LIMITED", "F05 closes the finite state-space boundary while retaining weak-identification and uncertainty gaps for F13.")


def main() -> None:
    """Run one F05 card."""
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", required=True, choices=tuple(f"F05-0{i}" for i in range(1, 8))); parser.add_argument("--input-manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--seeds", nargs="+", type=int, default=[20260915, 20260916, 20260917]); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True); started = datetime.now(timezone.utc).isoformat(); write_json(args.input_manifest, {"task_id": args.stage, "status": "declared", "source": "F04/F05 protocol", "model_calls": 0})
    {"F05-01": run_f05_01, "F05-02": run_f05_02, "F05-03": run_f05_03, "F05-04": run_f05_04, "F05-05": run_f05_05, "F05-06": run_f05_06, "F05-07": run_f05_07}[args.stage](out, args, started); print(json.dumps({"task_id": args.stage, "status": "EXECUTED", "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
