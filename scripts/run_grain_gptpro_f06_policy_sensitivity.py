#!/usr/bin/env python3
"""Frozen F06 hidden-policy counterexamples; not the full panel-B comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f00 import sha256
from grain_research.gptpro_foundation_v3.f06_policy_state import (
    fit_reactive_prefix,
    gaussian_policy_filter,
    policy_sensitivity,
    simulate_policy_pair,
)


def dump(path, value):
    """Persist immutable packet inputs and outputs."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def worker(case, smoke):
    """Observed-only subprocess never opens latent or counterfactual truth."""
    rows = json.loads((case / "observed.json").read_text())
    source = json.loads((case / "source_model.json").read_text())
    specification = json.loads((case / "specification.json").read_text())
    fit = fit_reactive_prefix(rows, source)
    dump(case / "prefix_fit.json", fit)
    a, b = fit["psi"][:2]
    q, r = np.exp(2 * np.asarray(source["log_sigma"]))
    betas = [-0.5, 0.4, 1.0] if smoke else np.round(np.arange(-0.5, 1.001, 0.05), 2)
    rhos = [0.0, 0.5, 0.95] if smoke else np.round(np.arange(0, 0.951, 0.05), 2)
    profile = []
    for rho in rhos:
        for beta in betas:
            if specification["policy"] == "hidden":
                result = policy_sensitivity(
                    rows, a, b, float(rho), float(beta), q=q, r=r
                )
                for j, gamma in enumerate((0.0, 1.0, 3.0)):
                    profile.append(
                        {
                            "rho": float(rho),
                            "beta": float(beta),
                            "gamma": gamma,
                            "loglik": result["loglik"][j]
                            if result["status"] == "OK"
                            else None,
                            "status": result["status"],
                            "final_state": result["state_mean"][-1][j]
                            if result["status"] == "OK"
                            else None,
                            "max_outside_mass": result.get("max_outside_mass"),
                            "grid_attempts": len(result["grid_attempts"]),
                            "grid_limit": result["limit"],
                            "quadrature_mass_error": result.get(
                                "max_quadrature_mass_error"
                            ),
                        }
                    )
            else:
                result = gaussian_policy_filter(
                    rows,
                    a,
                    b,
                    float(rho),
                    float(beta),
                    q=q,
                    r=r,
                    policy="recorded"
                    if specification["policy"] == "recorded"
                    else "exogenous",
                )
                profile.append(
                    {
                        "rho": float(rho),
                        "beta": float(beta),
                        "gamma": None,
                        "loglik": result["loglik"],
                        "status": result["status"],
                        "final_state": result["state_mean"][-1],
                        "max_outside_mass": None,
                        "grid_attempts": 0,
                        "grid_limit": None,
                        "quadrature_mass_error": None,
                    }
                )
    pd.DataFrame(profile).to_csv(case / "profile.csv", index=False)
    dump(
        case / "worker_summary.json",
        {
            "profile_rows": len(profile),
            "success": fit["success"],
            "source_hash": sha256(case / "source_model.json"),
            "observed_hash": sha256(case / "observed.json"),
            "method": "conditional likelihood with prefix working-MAP a,b fixed; not identification set or CI",
            "truth_access": False,
        },
    )


def main():
    """Create paired counterexamples, freeze inputs, infer, then evaluate truth."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.smoke)
        return
    if args.output is None or args.source is None:
        parser.error("--output and --source required")
    if args.output.exists():
        raise FileExistsError("preserve earlier packet")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    source_manifest = json.loads((args.source / "output_manifest.json").read_text())
    for item in source_manifest["files"]:
        assert sha256(args.source / item["path"]) == item["sha256"]
    dump(
        args.output / "protocol.json",
        {
            "kind": "frozen-policy counterexample, not population performance panel",
            "scope": "one predeclared exogenous instance per seed, three policies, beta zero/distributed, full conditional beta/rho grid; permitted counterexample path F06-05",
            "seeds": [20260915] if args.smoke else [20260915, 20260916, 20260917],
            "events": 8 if args.smoke else 24,
            "gamma_truth": 3,
            "gamma_sensitivity": [0, 1, 3],
            "grid": [[-8, 8, 401], [-12, 12, 601]],
            "max_tail": 0.001,
            "source": str(args.source.resolve()),
            "source_manifest_hash": sha256(args.source / "output_manifest.json"),
        },
    )
    cases, truths = [], {}
    for seed in [20260915] if args.smoke else [20260915, 20260916, 20260917]:
        params = np.random.default_rng(np.random.SeedSequence([seed, 6, 5, 0]))
        a, b, rho, beta = (
            params.normal(),
            params.normal(0.5, 0.5),
            params.uniform(0.2, 0.8),
            params.normal(0.4, 0.2),
        )
        events = 8 if args.smoke else 24
        initial = np.random.default_rng(np.random.SeedSequence([seed, 6, 5, 1])).normal(
            0, 0.5 / np.sqrt(1 - rho**2)
        )
        innovation = np.random.default_rng(
            np.random.SeedSequence([seed, 6, 5, 2])
        ).normal(0, 0.5, events)
        noise = np.random.default_rng(np.random.SeedSequence([seed, 6, 5, 3])).normal(
            0, 0.5, events
        )
        uniforms = np.random.default_rng(
            np.random.SeedSequence([seed, 6, 5, 4])
        ).random(events)
        author = (
            "u_"
            + hashlib.sha256(f"F06-policy-counterexample/{seed}".encode()).hexdigest()[
                :16
            ]
        )
        for reactivity, effect in (("zero", 0.0), ("distributed", beta)):
            for policy in ("zero", "recorded", "hidden"):
                case = args.output / f"{seed}_{reactivity}_{policy}"
                case.mkdir()
                result = simulate_policy_pair(
                    a, b, rho, effect, initial, innovation, noise, uniforms, policy
                )
                for row in result["observed"]:
                    row.update(
                        {
                            "author_id": author,
                            "event_id": f"{author}:{row['t']}",
                            "role": "application",
                        }
                    )
                dump(case / "observed.json", result.pop("observed"))
                (case / "source_model.json").write_bytes(
                    (args.source / f"core_{seed}" / "source_model.json").read_bytes()
                )
                dump(
                    case / "specification.json",
                    {"policy": policy, "seed": seed, "counterexample": True},
                )
                truths[case.name] = {
                    **result,
                    "a": float(a),
                    "b": float(b),
                    "rho": float(rho),
                    "beta": float(effect),
                    "initial": float(initial),
                    "innovations": innovation.tolist(),
                    "measurement": noise.tolist(),
                    "uniforms": uniforms.tolist(),
                    "seed": seed,
                    "reactivity": reactivity,
                    "policy": policy,
                }
                cases.append(case)
    dump(args.output / "truth.json", truths)
    dump(
        args.output / "input_manifest.json",
        {
            "files": [
                {"path": str(p.relative_to(args.output)), "sha256": sha256(p)}
                for p in args.output.rglob("*.json")
                if not p.name.startswith("._") and p.name != "truth.json"
            ]
        },
    )
    for case in cases:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            str(case.resolve()),
        ]
        if args.smoke:
            command.append("--smoke")
        with (case / "worker.log").open("w") as log:
            subprocess.run(
                command,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "OPENBLAS_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                },
            )
        print(
            json.dumps(
                {"case": case.name, "elapsed_seconds": time.monotonic() - started}
            ),
            flush=True,
        )
    best, profiles, perturbations = [], [], []
    for case in cases:
        truth = truths[case.name]
        data = pd.read_csv(case / "profile.csv")
        data["case"] = case.name
        profiles.append(data)
        perturbations.append(
            {
                "case": case.name,
                "seed": truth["seed"],
                "reactivity": truth["reactivity"],
                "policy": truth["policy"],
                "sum_squared_perturbation": truth["sum_squared_perturbation"],
                "sum_h": truth["sum_h"],
            }
        )
        for gamma, subset in data.groupby("gamma", dropna=False):
            valid = subset[subset.status == "OK"]
            if valid.empty:
                best.append(
                    {
                        "case": case.name,
                        "gamma": None if pd.isna(gamma) else gamma,
                        "status": "NO_VALID_GRID",
                        "failed_cells": len(subset),
                    }
                )
                continue
            maximum = valid.loglik.max()
            chosen = (
                valid[np.isclose(valid.loglik, maximum, atol=1e-10, rtol=0)]
                .sort_values(["rho", "beta"])
                .iloc[0]
            )
            envelope = valid[valid.loglik >= maximum - 2]
            beta_flat = bool(
                valid.groupby("rho").loglik.apply(lambda s: s.max() - s.min()).max()
                < 1e-10
            )
            best.append(
                {
                    "case": case.name,
                    "gamma": None if pd.isna(gamma) else gamma,
                    "status": "OK",
                    "beta_status": "prior_only_no_data_information"
                    if beta_flat
                    else "conditional_grid",
                    "rho": chosen.rho,
                    "beta": None if beta_flat else chosen.beta,
                    "loglik": maximum,
                    "rho_true": truth["rho"],
                    "beta_true": truth["beta"],
                    "rho_error": chosen.rho - truth["rho"],
                    "beta_error": None if beta_flat else chosen.beta - truth["beta"],
                    "final_state": chosen.final_state,
                    "state_error": chosen.final_state - truth["states"][-1],
                    "beta_drop2_min": envelope.beta.min(),
                    "beta_drop2_max": envelope.beta.max(),
                    "rho_drop2_min": envelope.rho.min(),
                    "rho_drop2_max": envelope.rho.max(),
                    "failed_cells": int((subset.status != "OK").sum()),
                    "max_tail": valid.max_outside_mass.max()
                    if valid.max_outside_mass.notna().any()
                    else None,
                }
            )
    pd.concat(profiles).to_csv(args.output / "all_profiles.csv", index=False)
    pd.DataFrame(best).to_csv(args.output / "conditional_optima.csv", index=False)
    pd.DataFrame(perturbations).to_csv(
        args.output / "paired_perturbations.csv", index=False
    )
    dump(
        args.output / "summary.json",
        {
            "status": "CONDITIONAL_POLICY_COUNTEREXAMPLES_EXECUTED_F06_OPEN",
            "smoke": args.smoke,
            "cases": len(cases),
            "independent_exogenous_instances": 1 if args.smoke else 3,
            "profile_rows": sum(len(p) for p in profiles),
            "elapsed_seconds": time.monotonic() - started,
            "remaining": [
                "independent profile and timing replay",
                "full B target policies and population errors",
                "IRT",
                "task contracts",
            ],
            "interpretation": "conditional sensitivity only, not full parameter uncertainty or human performance",
        },
    )
    (args.output / "source_archive").mkdir()
    for path in (
        Path(__file__),
        ROOT / "grain_research/gptpro_foundation_v3/f06_policy_state.py",
    ):
        (args.output / "source_archive" / path.name).write_bytes(path.read_bytes())
    dump(
        args.output / "output_manifest.json",
        {
            "files": [
                {"path": str(p.relative_to(args.output)), "sha256": sha256(p)}
                for p in sorted(args.output.rglob("*"))
                if p.is_file() and not p.name.startswith("._")
            ]
        },
    )


if __name__ == "__main__":
    main()
