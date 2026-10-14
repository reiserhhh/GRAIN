#!/usr/bin/env python3
"""One finite conditional author-memory discriminator; no accuracy/decay search."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_belief as B
from grain_research.construction import math_conditional_memory as M
from scripts import run_grain_observed_family_hm170 as R
from scripts.score_grain_author_belief_cold import load_bundle

CONFIG = "configs/grain_conditional_memory_hm282.json"
DEFAULT = ROOT / "results/grain_math_response_v1/conditional_memory_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_CONDITIONAL_MEMORY_PLAN_HM282.md",
    "grain_research/construction/math_conditional_memory.py",
    "grain_research/construction/math_author_belief.py",
    "scripts/run_grain_conditional_memory_hm282.py",
    "scripts/score_grain_conditional_memory_cold.py",
    "scripts/score_grain_author_belief_cold.py",
    "tests/test_grain_conditional_memory.py",
)
TRACE_KEYS = ("predictions", "logp", "profiles", "marginals", "effective_sources")


def status(out, value):
    """Atomically replace runtime state only; scientific artifacts remain immutable."""
    temporary = out / "runtime.tmp"
    R.write(temporary, value)
    os.replace(temporary, out / "runtime.json")


def prepare(out):
    """Freeze the SAME old natural episodes and unchanged SOURCE laws."""
    if out.exists():
        raise FileExistsError("No overwrite or scientific packet retry")
    cfg = R.read(ROOT / CONFIG)
    if cfg["contracts"] != list(M.CONTRACTS) or cfg["kernel_learning"]:
        raise ValueError("Changed fixed observation contracts")
    parent = ROOT / cfg["parent"]
    R.check(parent, ("computed_seal.json", "analyzed_seal.json"))
    z = load_bundle(parent / "observations.npz")
    if (
        z["source_paths"].shape != (386, 2, 18)
        or z["held_paths"].shape != (120, 2, 18)
        or z["targets"].shape != (2, 386, 5)
        or np.intersect1d(z["source_indices"], z["held"]).size
        or np.any(z["donors"] < 0)
        or np.any(z["donors"] == np.arange(120))
        or np.any(z["held_paths"][:, 0, 0] != z["held_paths"][z["donors"], 0, 0])
    ):
        raise ValueError("Changed event/source/stranger scope")
    records = R.read(parent / "records.json")
    for record in records:
        B.validate_record(record, 8)
    files = (
        "manifest.json",
        "prepared_seal.json",
        "computed_seal.json",
        "analyzed_seal.json",
        "observations.npz",
        "records.json",
        "trajectories.npz",
    ) + tuple(f"source_{r}.npz" for r in cfg["references"])
    out.mkdir(parents=True)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {
                str((parent / p).relative_to(ROOT)): R.digest(parent / p) for p in files
            },
            "original_scope_complete": False,
        },
    )
    R.archive(out / "observations.npz", **z)
    R.write(out / "example_record.json", records[0])
    for reference in cfg["references"]:
        bundle = load_bundle(parent / f"source_{reference}.npz")
        M.fixed_rows(bundle)
        R.archive(out / f"source_{reference}.npz", **bundle)
    R.write(
        out / "input_quality.json",
        {
            "source_authors": 386,
            "held_authors": 120,
            "source_held_overlap": 0,
            "first_state_matched_strangers": 120,
            "condition_bit": "TOP20 venue, not full topic/situation",
            "expression_bits": "two inherited frozen GRAIN coordinate cuts",
            "hidden_states_marginalized_not_given_as_covariates": True,
            "held_traits_not_read": True,
            "historical_not_untouched_validation": True,
            "no_new_neural_API_private_training": True,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "example_record.json", "input_quality.json")
        + tuple(f"source_{r}.npz" for r in cfg["references"]),
    )
    status(out, {"state": "PREPARED", "pid": None})
    print("HM282 prepared: exact coarsening on SAME386/120 natural histories")


def expired(_signal, _frame):
    """Stop without scientific retry at the declared finite wall/CPU-run cap."""
    raise TimeoutError("Original180-second finite compute cap reached")


def compute(out):
    """Observe all early contracts; reuse identical late natural events once per arm."""
    cfg = R.check(out, ("prepared_seal.json",))["config"]
    if R.read(out / "runtime.json")["state"] != "PREPARED":
        raise ValueError("No scientific rerun or automatic recovery")
    z = load_bundle(out / "observations.npz")
    started = time.monotonic()
    status(out, {"state": "RUNNING", "pid": os.getpid()})
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(cfg["compute_seconds_cap"])
    try:
        result = {}
        for reference in cfg["references"]:
            bundle = load_bundle(out / f"source_{reference}.npz")
            fields = np.array(
                [M.evidence_fields(bundle, x[0]) for x in z["held_paths"]]
            )
            ell = np.array([M.contract_log_likelihoods(f) for f in fields])
            weights = M.posterior_logweights(ell[:, :, -1])
            result[f"{reference}/early_evidence"] = fields
            result[f"{reference}/early_logweights"] = weights
            result[f"{reference}/early_profiles"] = np.array(
                [
                    [B.profile(np.exp(w), bundle["codes"]) for w in author]
                    for author in weights
                ]
            )
            for kind in ("self", "stranger"):
                traces = {k: [] for k in TRACE_KEYS}
                for u, path in enumerate(z["held_paths"][:, 1]):
                    origin = u if kind == "self" else int(z["donors"][u])
                    rows = [M.observe_late(bundle, path, w) for w in weights[origin]]
                    for key in TRACE_KEYS:
                        traces[key].append([row[key] for row in rows])
                for key in TRACE_KEYS:
                    result[f"{reference}/{kind}/{key}"] = np.array(traces[key])
        R.archive(out / "trajectories.npz", **result)
        runtime = {
            "state": "COMPUTED",
            "pid": None,
            "seconds": time.monotonic() - started,
            "early_forward_views": 3 * 3 * 120,
            "late_program_paths": 3 * 2 * 6 * 120,
            "held_people_not_program_paths": 120,
            "new_neural_API_private_training": 0,
            "original_scope_complete": False,
        }
        status(out, runtime)
        R.seal(out, "computed_seal.json", ("trajectories.npz", "runtime.json"))
        print(json.dumps(runtime))
    except BaseException as error:
        status(
            out,
            {
                "state": "FAILED",
                "pid": None,
                "error": repr(error),
                "seconds": time.monotonic() - started,
                "no_retry": True,
            },
        )
        raise
    finally:
        signal.alarm(0)


def analyze(out):
    """Retain signed mechanism differences, without selecting a winning predictor."""
    cfg = R.check(out, ("computed_seal.json",))["config"]
    arrays = load_bundle(out / "trajectories.npz")
    rng = np.random.default_rng(cfg["seed"])
    sample = rng.integers(120, size=(cfg["bootstrap_authors"], 120))
    rows, author_rows, descriptions = [], [], {}
    for reference in cfg["references"]:
        self_score = arrays[f"{reference}/self/logp"][..., 1:].mean(-1)
        stranger = arrays[f"{reference}/stranger/logp"][..., 1:].mean(-1)
        contrasts = {}
        for k, name in enumerate(M.CONTRACTS[1:], start=1):
            contrasts[f"{name}_minus_fresh"] = self_score[:, k] - self_score[:, 0]
            contrasts[f"{name}_self_minus_stranger"] = self_score[:, k] - stranger[:, k]
        for name, left, right in (
            ("expression_minus_condition", 3, 2),
            ("condition_minus_full", 2, 1),
            ("expression_minus_full", 3, 1),
            ("conditional_expression_minus_natural_expression", 4, 3),
            ("conditional_expression_minus_full", 4, 1),
            ("conditional_condition_minus_full", 5, 1),
        ):
            contrasts[name] = self_score[:, left] - self_score[:, right]
        for name, delta in contrasts.items():
            lo, hi = np.quantile(delta[sample].mean(1), [0.025, 0.975])
            rows.append(
                {
                    "reference": reference,
                    "contrast": name,
                    "mean_nats_per_transition": float(delta.mean()),
                    "conditional_percentile_low": float(lo),
                    "conditional_percentile_high": float(hi),
                    "positive_authors": int((delta > 0).sum()),
                    "authors": 120,
                    "not_independent_donor_population_CI": "stranger" in name,
                }
            )
            author_rows.extend(
                {
                    "reference": reference,
                    "contrast": name,
                    "author_index": u,
                    "delta": float(v),
                }
                for u, v in enumerate(delta)
            )
        fields = arrays[f"{reference}/early_evidence"][:, :, -1]
        residual = fields[:, 0] - fields[:, 1] - fields[:, 2]
        residual -= residual.mean(-1, keepdims=True)
        weights = arrays[f"{reference}/early_logweights"]
        product = M.posterior_logweights(fields[:, 1] + fields[:, 2])
        gap = abs(np.exp(product) - np.exp(weights[:, 1])).sum(-1) / 2
        descriptions[reference] = {
            "self_logscore_by_contract": self_score.mean(0).tolist(),
            "stranger_logscore_by_contract": stranger.mean(0).tolist(),
            "start_logscore_by_contract": arrays[f"{reference}/self/logp"][:, :, 0]
            .mean(0)
            .tolist(),
            "early_effective_sources_by_contract": (1 / np.exp(2 * weights).sum(-1))
            .mean(0)
            .tolist(),
            "joint_minus_marginal_logevidence_centered_RMS": float(
                np.sqrt(np.mean(residual**2))
            ),
            "invalid_product_vs_full_posterior_TV_mean": float(gap.mean()),
            "invalid_product_vs_full_posterior_TV_max": float(gap.max()),
            "early_all32profiles_both_assignments_preserved": True,
        }
    R.table(out / "contrasts.csv", rows)
    R.table(out / "author_contrasts.csv", author_rows)
    R.write(
        out / "summary.json",
        {
            "references": descriptions,
            "contract_order": list(M.CONTRACTS),
            "conditional_design_changes_prior_not_denoising_or_do_identification": True,
            "no_arm_selection_or_accuracy_fit": True,
            "not_human_trait_posteriors": True,
            "fixed_kernel_base_not_integrated_coarsening": True,
            "runtime": R.read(out / "runtime.json"),
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "analyzed_seal.json",
        ("contrasts.csv", "author_contrasts.csv", "summary.json"),
    )
    print(json.dumps(R.read(out / "summary.json")))


def independent_forward(bundle, path, view):
    """Independent LOG-space sum-product, not the primary scaled forward function."""
    a0, a = bundle["alpha0"], bundle["alpha"]
    mu = np.log(a0) - np.log(a0.sum(1, keepdims=True))
    transition = np.log(a) - np.log(a.sum(2, keepdims=True))
    alphabet = np.arange(8)
    output, v = [np.zeros(len(a0))], mu
    for t, cell in enumerate(path):
        if t:
            v = logsumexp(v[:, :, None] + transition, axis=1)
        mask = (
            alphabet == cell
            if view == "full"
            else alphabet // 4 == cell // 4
            if view == "condition"
            else alphabet % 4 == cell % 4
        )
        v = np.where(mask, v, -np.inf)
        output.append(logsumexp(v, axis=1))
    return np.array(output)


def verify(out):
    """Audit every arm, natural-law replay and cold scoring, without new observations."""
    cfg = R.check(out, ("computed_seal.json", "analyzed_seal.json"))["config"]
    z = load_bundle(out / "observations.npz")
    arrays = load_bundle(out / "trajectories.npz")
    old = load_bundle(ROOT / cfg["parent"] / "trajectories.npz")
    errors = {
        key: 0.0
        for key in (
            "hidden_forward",
            "late_likelihood",
            "profile",
            "old_replay",
            "chain",
        )
    }
    endpoints = 0
    for reference in cfg["references"]:
        bundle = load_bundle(out / f"source_{reference}.npz")
        early = []
        for u, path in enumerate(z["held_paths"][:, 0]):
            fields = np.array([independent_forward(bundle, path, v) for v in M.VIEWS])
            errors["hidden_forward"] = max(
                errors["hidden_forward"],
                float(abs(fields - arrays[f"{reference}/early_evidence"][u]).max()),
            )
            early.append(fields[:, -1])
            full, condition, expression = fields
            for reconstructed in (
                condition + full - condition,
                expression + full - expression,
            ):
                errors["chain"] = max(
                    errors["chain"], float(abs(reconstructed - full).max())
                )
        early = np.array(early)
        # Direct complete-path products at every prefix, not a second observe() call.
        for u, path in enumerate(z["held_paths"][:, 1]):
            late = independent_forward(bundle, path, "full")
            for kind in ("self", "stranger"):
                origin = u if kind == "self" else int(z["donors"][u])
                f, c, s = early[origin]
                es = (np.zeros_like(f), f, c, s, f - c, f - s)
                for k, ell in enumerate(es):
                    joint = late + ell
                    mixture = logsumexp(joint, axis=-1)
                    logs = np.diff(mixture)
                    stored = arrays[f"{reference}/{kind}/logp"][u, k]
                    errors["late_likelihood"] = max(
                        errors["late_likelihood"], float(abs(logs - stored).max())
                    )
                    weights = np.exp(joint - mixture[:, None])
                    expected = np.array(
                        [B.profile(w, bundle["codes"]) for w in weights]
                    ).transpose(1, 0, 2)
                    errors["profile"] = max(
                        errors["profile"],
                        float(
                            abs(
                                expected - arrays[f"{reference}/{kind}/profiles"][u, k]
                            ).max()
                        ),
                    )
                    endpoints += 1
        for kind, old_key in (("self", "carry"), ("stranger", "stranger")):
            for key in ("logp", "profiles", "predictions"):
                errors["old_replay"] = max(
                    errors["old_replay"],
                    float(
                        abs(
                            arrays[f"{reference}/{kind}/{key}"][:, 1]
                            - old[f"{reference}/{old_key}/{key}"][:, 0]
                        ).max()
                    ),
                )
        for key in ("logp", "profiles", "predictions"):
            fresh = old[f"{reference}/fresh/{key}"][:, 1, 1]
            errors["old_replay"] = max(
                errors["old_replay"],
                float(abs(arrays[f"{reference}/self/{key}"][:, 0] - fresh).max()),
            )
    if (
        not all(np.isfinite(v).all() for v in arrays.values())
        or max(errors.values()) > 1e-10
    ):
        raise ArithmeticError(json.dumps(errors))
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_conditional_memory_cold.py"),
            "--bundle",
            str(out / "source_pooled.npz"),
            "--input",
            str(out / "example_record.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cold = json.loads(run.stdout)
    error = 0.0
    for k, name in enumerate(M.CONTRACTS):
        row = cold["contracts"][name]
        error = max(
            error,
            float(
                abs(np.array(row["late_logp"]) - arrays["pooled/self/logp"][0, k]).max()
            ),
            float(
                abs(
                    np.array(row["late_profiles"])
                    - arrays["pooled/self/profiles"][0, k]
                ).max()
            ),
        )
    if error > 1e-12:
        raise ArithmeticError("Cold contract readback changed")
    report = {
        "PASS": True,
        "independent_logspace_early_views": 1080,
        "all_late_arm_endpoints": endpoints,
        "errors": errors,
        "cold_error": error,
        "no_new_observations_or_fitting": True,
        "same_agent_not_blind": True,
        "original_scope_complete": False,
    }
    R.write(out / "cold_readback.json", cold)
    R.write(out / "verification.json", report)
    R.seal(out, "verification_seal.json", ("cold_readback.json", "verification.json"))
    print(json.dumps(report))


def main():
    """Run finite stages or verify existing seals; no automatic experiment queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "compute", "analyze", "verify", "check")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(
            args.output,
            ("computed_seal.json", "analyzed_seal.json", "verification_seal.json"),
        )
        print("HM282 all seals PASS")
    else:
        globals()[args.stage](args.output)


if __name__ == "__main__":
    main()
