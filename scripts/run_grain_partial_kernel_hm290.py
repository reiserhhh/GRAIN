#!/usr/bin/env python3
"""One partial-observation shared-kernel construction,not a forecast-accuracy search."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_belief as B
from grain_research.construction import math_conditional_memory as C
from grain_research.construction import math_partial_kernel as M
from scripts import run_grain_conditional_memory_hm282 as H
from scripts import run_grain_observed_family_hm170 as R
from scripts.score_grain_author_belief_cold import load_bundle

CONFIG = "configs/grain_partial_kernel_hm290.json"
SPEC = R.read(ROOT / CONFIG)
DEFAULT = ROOT / SPEC["output"]
TRACE_KEYS = ("predictions", "logp", "profiles", "marginals", "effective_sources")
SCIENCE = (
    CONFIG,
    "docs/GRAIN_PARTIAL_KERNEL_PLAN_HM290.md",
    "grain_research/construction/math_partial_kernel.py",
    "scripts/run_grain_partial_kernel_hm290.py",
    "scripts/score_grain_partial_kernel_cold.py",
    "tests/test_grain_partial_kernel.py",
    "grain_research/construction/math_author_belief.py",
    "grain_research/construction/math_conditional_memory.py",
    "scripts/run_grain_conditional_memory_hm282.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/score_grain_author_belief_cold.py",
)


def prepare(out):
    """Freeze exact old sampling support and SOURCE parameters,never fit a new prior."""
    if out.exists():
        raise FileExistsError("No overwrite or scientific retry")
    parent, integrated = ROOT / SPEC["parent"], ROOT / SPEC["integrated_parent"]
    R.check(
        parent, ("computed_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    R.check(integrated, ("computed_seal.json", "analyzed_seal.json"))
    z = load_bundle(parent / "observations.npz")
    if (
        z["source_paths"].shape != (386, 2, 18)
        or z["held_paths"].shape != (120, 2, 18)
        or np.intersect1d(z["source_indices"], z["held"]).size
        or np.any(z["donors"] < 0)
        or not np.array_equal(np.sort(z["donors"]), np.arange(120))
        or np.any(z["donors"] == np.arange(120))
        or np.any(z["held_paths"][:, 0, 0] != z["held_paths"][z["donors"], 0, 0])
    ):
        raise ValueError("Changed existing author/event/control scope")
    if (
        SPEC["contracts"] != list(C.CONTRACTS)
        or len(SPEC["seeds"]) != 2
        or SPEC["particles"] != 64
    ):
        raise ValueError("Changed fixed observation/numerical contract")
    inputs = [
        parent / p
        for p in (
            "manifest.json",
            "observations.npz",
            "example_record.json",
            "trajectories.npz",
            "verification_seal.json",
        )
    ]
    inputs += [parent / f"source_{ref}.npz" for ref in SPEC["references"]]
    inputs += [
        integrated / p
        for p in (
            "manifest.json",
            "trajectories.npz",
            "computed_seal.json",
            "analyzed_seal.json",
        )
    ]
    out.mkdir()
    R.write(
        out / "manifest.json",
        {
            "config": SPEC,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "original_scope_complete": False,
        },
    )
    R.archive(out / "observations.npz", **z)
    R.write(out / "example_record.json", R.read(parent / "example_record.json"))
    for reference in SPEC["references"]:
        bundle = load_bundle(parent / f"source_{reference}.npz")
        M.parameters(bundle)
        R.archive(out / f"source_{reference}.npz", **bundle)
    R.write(
        out / "input_quality.json",
        {
            "source_authors": 386,
            "held_authors": 120,
            "events_per_episode": 18,
            "source_held_overlap": 0,
            "first_state_matched_strangers": 120,
            "held_labels_neural_API_private_training_fit": 0,
            "source_kernels_prior_states_donors_unchanged": True,
            "partial_integration_approximate_not_exact": True,
            "historical_not_untouched_validation": True,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "example_record.json", "input_quality.json")
        + tuple(f"source_{r}.npz" for r in SPEC["references"]),
    )
    H.status(out, {"state": "PREPARED", "pid": None})
    print(json.dumps(R.read(out / "input_quality.json")), flush=True)


def expired(_signal, _frame):
    """Stop at the original finite budget without renewing it or increasing particles."""
    raise TimeoutError("Original360-second new compute cap exhausted")


def compute(out):
    """Score every old natural episode under both fixed numerical realizations."""
    cfg = R.check(out)["config"]
    if R.read(out / "runtime.json")["state"] != "PREPARED":
        raise ValueError("No repeat or automatic recovery of scientific computation")
    z = load_bundle(out / "observations.npz")
    started = time.monotonic()
    H.status(
        out,
        {"state": "RUNNING", "pid": os.getpid(), "completed_realization_references": 0},
    )
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(cfg["compute_seconds_cap"])
    try:
        result, completed = {}, 0
        with threadpool_limits(limits=1):
            for ref, reference in enumerate(cfg["references"]):
                bundle = load_bundle(out / f"source_{reference}.npz")
                for rep, base_seed in enumerate(cfg["seeds"]):
                    seed = base_seed + 10 * ref
                    early_fields, early_weights, early_violations, early_profiles = (
                        [],
                        [],
                        [],
                        [],
                    )
                    traces = {
                        kind: {key: [None] * 120 for key in TRACE_KEYS}
                        for kind in ("self", "stranger")
                    }
                    recipients = np.argsort(z["donors"])
                    for origin, path in enumerate(z["held_paths"]):
                        current = M.early_contracts(
                            bundle, path[0], cfg["particles"], seed
                        )
                        early_fields.append(current["fields"])
                        early_weights.append(current["logweights"])
                        early_violations.append(
                            current["approximate_coarse_likelihood_exceeds_exact_full"]
                        )
                        early_profiles.append(
                            [
                                B.profile(np.exp(w), bundle["codes"])
                                for w in current["logweights"]
                            ]
                        )
                        # The old donor map is bijective: one filter serves its owner and recipient.
                        for kind, u in (
                            ("self", origin),
                            ("stranger", int(recipients[origin])),
                        ):
                            rows = [
                                M.late(bundle, z["held_paths"][u, 1], w, state)
                                for w, state in zip(
                                    current["logweights"],
                                    current["states"],
                                    strict=True,
                                )
                            ]
                            for key in TRACE_KEYS:
                                traces[kind][key][u] = [row[key] for row in rows]
                        if (origin + 1) % 40 == 0:
                            print(
                                f"HM290 {reference}/rep{rep} {origin + 1}/120 authors",
                                flush=True,
                            )
                    result[f"{reference}/rep{rep}/early_fields"] = np.asarray(
                        early_fields
                    )
                    result[f"{reference}/rep{rep}/early_logweights"] = np.asarray(
                        early_weights
                    )
                    result[f"{reference}/rep{rep}/monotonicity_violations"] = (
                        np.asarray(early_violations)
                    )
                    result[f"{reference}/rep{rep}/early_profiles"] = np.asarray(
                        early_profiles
                    )
                    for kind, arrays in traces.items():
                        for key, value in arrays.items():
                            result[f"{reference}/rep{rep}/{kind}/{key}"] = np.asarray(
                                value
                            )
                    completed += 1
                    H.status(
                        out,
                        {
                            "state": "RUNNING",
                            "pid": os.getpid(),
                            "completed_realization_references": completed,
                            "seconds": time.monotonic() - started,
                        },
                    )
                    print(
                        f"HM290 {completed}/6 reference-realizations complete",
                        flush=True,
                    )
        R.archive(out / "trajectories.npz", **result)
        runtime = {
            "state": "COMPUTED",
            "pid": None,
            "seconds": time.monotonic() - started,
            "partial_early_filters": 3 * 2 * 120 * 2,
            "late_programs": 3 * 2 * 2 * 120 * 6,
            "held_people_not_programs": 120,
            "particles_per_SOURCE": 64,
            "new_neural_API_private_training": 0,
        }
        H.status(out, runtime)
        R.seal(out, "computed_seal.json", ("trajectories.npz", "runtime.json"))
        print(json.dumps(runtime), flush=True)
    except BaseException as error:
        H.status(
            out,
            {
                "state": "FAILED",
                "pid": None,
                "seconds": time.monotonic() - started,
                "error": repr(error),
                "no_retry": True,
            },
        )
        raise
    finally:
        signal.alarm(0)


def analyze(out):
    """Retain paired scientific differences and sampling diagnostics without selecting models."""
    R.check(out, ("computed_seal.json",))
    data = load_bundle(out / "trajectories.npz")
    old = load_bundle(ROOT / SPEC["parent"] / "trajectories.npz")
    sample = np.random.default_rng(SPEC["bootstrap_seed"]).integers(
        120, size=(SPEC["bootstrap_authors"], 120)
    )
    rows, individuals, diagnostics = [], [], []
    for reference in SPEC["references"]:
        previous = old[f"{reference}/self/logp"][..., 1:].mean(-1)
        for rep in (0, 1):
            own = data[f"{reference}/rep{rep}/self/logp"][..., 1:].mean(-1)
            stranger = data[f"{reference}/rep{rep}/stranger/logp"][..., 1:].mean(-1)
            for j, contract in enumerate(C.CONTRACTS):
                contrasts = {
                    "own_minus_fresh": own[:, j] - own[:, 0],
                    "own_minus_stranger": own[:, j] - stranger[:, j],
                    "integrated_minus_old_fixed_same_contract": own[:, j]
                    - previous[:, j],
                }
                for name, delta in contrasts.items():
                    lo, hi = np.quantile(delta[sample].mean(1), [0.025, 0.975])
                    rows.append(
                        {
                            "reference": reference,
                            "realization": rep,
                            "contract": contract,
                            "contrast": name,
                            "mean_nats_per_transition": float(delta.mean()),
                            "conditional_exploratory_low": float(lo),
                            "conditional_exploratory_high": float(hi),
                            "positive_authors": int((delta > 0).sum()),
                        }
                    )
                    individuals.extend(
                        {
                            "reference": reference,
                            "realization": rep,
                            "contract": contract,
                            "contrast": name,
                            "author_index": u,
                            "delta": float(d),
                        }
                        for u, d in enumerate(delta)
                    )
            violations = data[f"{reference}/rep{rep}/monotonicity_violations"]
            diagnostics.append(
                {
                    "reference": reference,
                    "realization": rep,
                    "estimated_coarse_below_exact_full_prefix_cells": int(
                        violations.sum()
                    ),
                    "total_prefix_cells": int(violations.size),
                }
            )
    variation = []
    for reference in SPEC["references"]:
        a, b = (data[f"{reference}/rep{r}/self/logp"][..., 1:].mean(-1) for r in (0, 1))
        pa, pb = (data[f"{reference}/rep{r}/early_profiles"] for r in (0, 1))
        for j, contract in enumerate(C.CONTRACTS):
            variation.append(
                {
                    "reference": reference,
                    "contract": contract,
                    "late_logscore_mean_abs_difference": float(
                        np.mean(abs(a[:, j] - b[:, j]))
                    ),
                    "late_logscore_max_abs_difference": float(
                        np.max(abs(a[:, j] - b[:, j]))
                    ),
                    "early_profile_total_variation_mean": float(
                        np.mean(abs(pa[:, j] - pb[:, j]).sum(-1) / 2)
                    ),
                }
            )
    R.table(out / "contrasts.csv", rows)
    R.table(out / "individual_contrasts.csv", individuals)
    R.table(out / "integration_diagnostics.csv", diagnostics)
    R.table(out / "realization_variation.csv", variation)
    R.write(
        out / "summary.json",
        {
            "primary_realization": 0,
            "all_contrasts": rows,
            "integration_diagnostics": diagnostics,
            "numerical_realization_variation": variation,
            "runtime": R.read(out / "runtime.json"),
            "old_fixed_baseline_not_refitted": True,
            "approximate_not_exact_coarse_integration": True,
            "not_label_accuracy_or_human_trait_change": True,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "contrasts.csv",
            "individual_contrasts.csv",
            "integration_diagnostics.csv",
            "realization_variation.csv",
            "summary.json",
        ),
    )
    print(
        json.dumps(
            {
                "primary_pooled": [
                    r
                    for r in rows
                    if r["reference"] == "pooled" and r["realization"] == 0
                ],
                "integration_diagnostics": diagnostics,
            }
        ),
        flush=True,
    )


def verify(out):
    """Audit exact full cases against old beta law and all returned numerical programs."""
    R.check(out, ("computed_seal.json", "analyzed_seal.json"))
    z = load_bundle(out / "observations.npz")
    data = load_bundle(out / "trajectories.npz")
    old = load_bundle(ROOT / SPEC["integrated_parent"] / "trajectories.npz")
    errors = {
        "full_exact_source_likelihood": 0.0,
        "full_carry_original": 0.0,
        "fresh_original": 0.0,
        "two_realizations_full_exact": 0.0,
        "normalization": 0.0,
    }
    with threadpool_limits(limits=1):
        for ref, reference in enumerate(SPEC["references"]):
            bundle = load_bundle(out / f"source_{reference}.npz")
            for rep in (0, 1):
                fields = data[f"{reference}/rep{rep}/early_fields"]
                for u, episodes in enumerate(z["held_paths"]):
                    likelihood = B.component_log_likelihood(bundle, [episodes[0]], True)
                    errors["full_exact_source_likelihood"] = max(
                        errors["full_exact_source_likelihood"],
                        float(abs(fields[u, 0, -1] - likelihood).max()),
                    )
                for key in TRACE_KEYS:
                    current = data[f"{reference}/rep{rep}/self/{key}"]
                    errors["full_carry_original"] = max(
                        errors["full_carry_original"],
                        float(
                            abs(
                                current[:, 1] - old[f"{reference}/carry/{key}"][:, 1]
                            ).max()
                        ),
                    )
                    errors["fresh_original"] = max(
                        errors["fresh_original"],
                        float(
                            abs(
                                current[:, 0] - old[f"{reference}/fresh/{key}"][:, 3, 1]
                            ).max()
                        ),
                    )
                for kind in ("self", "stranger"):
                    prediction = data[f"{reference}/rep{rep}/{kind}/predictions"]
                    profiles = data[f"{reference}/rep{rep}/{kind}/profiles"]
                    errors["normalization"] = max(
                        errors["normalization"],
                        float(abs(prediction.sum(-1) - 1).max()),
                        float(abs(profiles.sum(-1) - 1).max()),
                    )
                    if np.any(prediction <= 0) or not np.isfinite(prediction).all():
                        raise ArithmeticError("Missing numerical prediction support")
            for key in TRACE_KEYS:
                a, b = (data[f"{reference}/rep{r}/self/{key}"][:, :2] for r in (0, 1))
                errors["two_realizations_full_exact"] = max(
                    errors["two_realizations_full_exact"], float(abs(a - b).max())
                )
    reply = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_partial_kernel_cold.py"),
            "--bundle",
            str(out / "source_pooled.npz"),
            "--input",
            str(out / "example_record.json"),
            "--seed",
            str(SPEC["seeds"][0] + 20),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cold = json.loads(reply.stdout)
    for j, name in enumerate(C.CONTRACTS):
        for key in TRACE_KEYS:
            errors[f"cold/{name}/{key}"] = float(
                abs(
                    np.asarray(cold["contracts"][name][key])
                    - data[f"pooled/rep0/self/{key}"][0, j]
                ).max()
            )
    if max(errors.values()) > 1e-10 or cold["neural_modules_loaded"]:
        raise ArithmeticError("Exact-control or cold replay failed;no precision rescue")
    R.write(out / "cold_readback.json", cold)
    R.write(
        out / "verification.json",
        {
            "PASS": True,
            "errors": errors,
            "exact_FULL_controls_not_coarse_exactness": True,
            "same_agent_not_blind": True,
            "new_neural_calls": 0,
        },
    )
    R.seal(out, "verification_seal.json", ("cold_readback.json", "verification.json"))
    print(json.dumps(R.read(out / "verification.json")), flush=True)


def main():
    """Execute bounded immutable stages without restarting old or failed computations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "compute", "analyze", "verify", "check", "run")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "run":
        compute(args.out)
        analyze(args.out)
        verify(args.out)
        R.check(
            args.out,
            ("computed_seal.json", "analyzed_seal.json", "verification_seal.json"),
        )
        print("HM290 construction/scientific stages/seals PASS", flush=True)
    elif args.stage == "check":
        R.check(
            args.out,
            ("computed_seal.json", "analyzed_seal.json", "verification_seal.json"),
        )
        print("HM290 scientific seals PASS")
    else:
        {"prepare": prepare, "compute": compute, "analyze": analyze, "verify": verify}[
            args.stage
        ](args.out)


if __name__ == "__main__":
    main()
