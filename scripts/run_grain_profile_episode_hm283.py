#!/usr/bin/env python3
"""Test shared psychological-anchor memory with episode-specific expression modes."""

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
from grain_research.construction import math_conditional_memory as C
from grain_research.construction import math_profile_episode as M
from scripts import run_grain_observed_family_hm170 as R
from scripts.run_grain_conditional_memory_hm282 import status
from scripts.score_grain_author_belief_cold import load_bundle

CONFIG = "configs/grain_profile_episode_hm283.json"
DEFAULT = ROOT / "results/grain_math_response_v1/profile_episode_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_PROFILE_EPISODE_PLAN_HM283.md",
    "grain_research/construction/math_profile_episode.py",
    "grain_research/construction/math_author_belief.py",
    "grain_research/construction/math_conditional_memory.py",
    "grain_research/construction/math_vector_memory.py",
    "scripts/run_grain_profile_episode_hm283.py",
    "scripts/run_grain_conditional_memory_hm282.py",
    "scripts/score_grain_profile_episode_cold.py",
    "scripts/score_grain_author_belief_cold.py",
    "tests/test_grain_profile_episode.py",
    "tests/test_grain_conditional_memory.py",
)
TRACE_KEYS = ("predictions", "logp", "profiles", "marginals", "effective_sources")


def prepare(out):
    """Freeze unchanged SOURCE kernels, full histories and actual/shuffled anchors."""
    if out.exists():
        raise FileExistsError("No overwrite or scientific retry")
    cfg = R.read(ROOT / CONFIG)
    if cfg["models"] != list(M.MODELS) or cfg["primary_reference"] != "pooled":
        raise ValueError("Changed scientific contracts")
    parent = ROOT / cfg["parent"]
    R.check(
        parent, ("computed_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    z = load_bundle(parent / "observations.npz")
    if (
        z["held_paths"].shape != (120, 2, 18)
        or np.intersect1d(z["held"], z["source_indices"]).size
    ):
        raise ValueError("Changed same-author/source sampling")
    files = (
        "manifest.json",
        "prepared_seal.json",
        "computed_seal.json",
        "analyzed_seal.json",
        "verification_seal.json",
        "observations.npz",
        "trajectories.npz",
        "example_record.json",
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
    R.write(out / "example_record.json", R.read(parent / "example_record.json"))
    support = {}
    for reference in cfg["references"]:
        b = load_bundle(parent / f"source_{reference}.npz")
        counts = [M.profile_bank(b, a)["counts"] for a in (0, 1)]
        support[reference] = {
            "counts_actual_permuted": [c.tolist() for c in counts],
            "supported_profile_masks": [(c > 0).tolist() for c in counts],
            "counts_match": bool(np.array_equal(*counts)),
            "singleton_profiles": [int((c == 1).sum()) for c in counts],
            "missing_profiles": [np.flatnonzero(c == 0).tolist() for c in counts],
        }
        R.archive(out / f"source_{reference}.npz", **b)
    if not support["pooled"]["counts_match"]:
        raise ValueError("Primary old-label control no longer count matched")
    R.write(out / "support.json", support)
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "example_record.json", "support.json")
        + tuple(f"source_{r}.npz" for r in cfg["references"]),
    )
    status(out, {"state": "PREPARED", "pid": None})
    print(
        "HM283 prepared: complete-path profile sharing; pooled count-matched old control"
    )


def expired(_signal, _frame):
    """Stop without restoring a scientific queue at the original finite cap."""
    raise TimeoutError("Original120-second compute cap")


def compute(out):
    """Evaluate only two new complete-episode laws; full/fresh data are reused."""
    cfg = R.check(out, ("prepared_seal.json",))["config"]
    if R.read(out / "runtime.json")["state"] != "PREPARED":
        raise ValueError("No scientific retry")
    z = load_bundle(out / "observations.npz")
    old = load_bundle(ROOT / cfg["parent"] / "trajectories.npz")
    started = time.monotonic()
    status(out, {"state": "RUNNING", "pid": os.getpid()})
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(cfg["compute_seconds_cap"])
    try:
        result = {}
        for reference in cfg["references"]:
            b = load_bundle(out / f"source_{reference}.npz")
            fine = old[f"{reference}/early_logweights"][:, 1]
            projected = np.array(
                [M.episode_logweights(b, fine, a) for a in (0, 1)]
            ).transpose(1, 0, 2)
            prior = np.full_like(fine, -np.log(len(b["alpha0"])))
            weights = np.concatenate([prior[:, None], fine[:, None], projected], axis=1)
            result[f"{reference}/boundary_logweights"] = weights
            result[f"{reference}/boundary_profiles"] = np.array(
                [[B.profile(np.exp(w), b["codes"]) for w in u] for u in weights]
            )
            for kind, previous in (("self", "self"), ("stranger", "stranger")):
                traces = {key: [] for key in TRACE_KEYS}
                for u, late in enumerate(z["held_paths"][:, 1]):
                    origin = u if kind == "self" else int(z["donors"][u])
                    rows = [C.observe_late(b, late, w) for w in projected[origin]]
                    for key in TRACE_KEYS:
                        retained = old[f"{reference}/{previous}/{key}"][u, :2]
                        traces[key].append(
                            np.concatenate([retained, np.array([r[key] for r in rows])])
                        )
                for key in TRACE_KEYS:
                    result[f"{reference}/{kind}/{key}"] = np.array(traces[key])
        R.archive(out / "trajectories.npz", **result)
        runtime = {
            "state": "COMPUTED",
            "pid": None,
            "seconds": time.monotonic() - started,
            "new_complete_path_evaluations": 1440,
            "reused_fresh_full_path_descriptions": 1440,
            "held_people": 120,
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
    """Read actual-label specificity separately from a generic memory reset benefit."""
    cfg = R.check(out, ("computed_seal.json",))["config"]
    arrays = load_bundle(out / "trajectories.npz")
    rng = np.random.default_rng(cfg["seed"])
    sample = rng.integers(120, size=(cfg["bootstrap_authors"], 120))
    rows, author_rows, descriptions = [], [], {}
    for reference in cfg["references"]:
        scores = arrays[f"{reference}/self/logp"][..., 1:].mean(-1)
        stranger = arrays[f"{reference}/stranger/logp"][..., 1:].mean(-1)
        contrasts = {
            "actual_profile_minus_permuted_profile": scores[:, 2] - scores[:, 3]
        }
        for index, name in enumerate(M.MODELS[2:], start=2):
            for comparator, j in (("full", 1), ("fresh", 0)):
                contrasts[f"{name}_minus_{comparator}"] = (
                    scores[:, index] - scores[:, j]
                )
            contrasts[f"{name}_self_minus_stranger"] = (
                scores[:, index] - stranger[:, index]
            )
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
                    "primary_count_matched_reference": reference == "pooled",
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
        profile = arrays[f"{reference}/boundary_profiles"]
        marginal = arrays[f"{reference}/self/marginals"]
        selected_errors, next_target = [], []
        for assignment in (0, 1):
            selected_errors.append(
                float(
                    abs(
                        profile[:, assignment + 2, assignment]
                        - profile[:, 1, assignment]
                    ).max()
                )
            )
            delta = marginal[:, assignment + 2, assignment] - marginal[:, 1, assignment]
            next_target.append(np.sqrt(np.mean(delta**2, axis=0)).tolist())
        w = arrays[f"{reference}/boundary_logweights"]
        descriptions[reference] = {
            "self_transition_logscore_by_model": scores.mean(0).tolist(),
            "stranger_transition_logscore_by_model": stranger.mean(0).tolist(),
            "start_logscore_by_model": arrays[f"{reference}/self/logp"][:, :, 0]
            .mean(0)
            .tolist(),
            "selected_boundary32profile_error_actual_permuted": selected_errors,
            "selected_future_target_change_RMS_assignment_prefix_trait": next_target,
            "boundary_effective_source_modes_by_model": (1 / np.exp(2 * w).sum(-1))
            .mean(0)
            .tolist(),
            "reference_support": R.read(out / "support.json")[reference],
        }
    R.table(out / "contrasts.csv", rows)
    R.table(out / "author_contrasts.csv", author_rows)
    R.write(
        out / "summary.json",
        {
            "model_order": list(M.MODELS),
            "references": descriptions,
            "runtime": R.read(out / "runtime.json"),
            "primary": "pooled actual-minus-old-permuted profile; one historical control not a randomization test",
            "reset_formula_already_HM211_not_rediscovered": True,
            "future_prediction_not_human_trait_accuracy": True,
            "model_selection_or_new_parameter_fit": False,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "analyzed_seal.json",
        ("contrasts.csv", "author_contrasts.csv", "summary.json"),
    )
    print(json.dumps({"rows": rows, "runtime": R.read(out / "runtime.json")}))


def literal_path(bundle, path):
    """Direct indexed complete-path products, independent of observe/forward."""
    a0, a = bundle["alpha0"], bundle["alpha"]
    initial = a0 / a0.sum(1, keepdims=True)
    transition = a / a.sum(2, keepdims=True)
    ell = np.zeros(len(a0))
    output = [ell.copy()]
    for t, value in enumerate(path):
        ell += np.log(
            initial[:, value] if t == 0 else transition[:, path[t - 1], value]
        )
        output.append(ell.copy())
    return np.array(output)


def verify(out):
    """Check all new laws by explicit per-profile episode sums and prefix products."""
    cfg = R.check(out, ("computed_seal.json", "analyzed_seal.json"))["config"]
    z, arrays = (
        load_bundle(out / "observations.npz"),
        load_bundle(out / "trajectories.npz"),
    )
    old = load_bundle(ROOT / cfg["parent"] / "trajectories.npz")
    errors = {
        key: 0.0
        for key in ("projection", "likelihood", "profile", "boundary", "old_replay")
    }
    endpoints = 0
    for reference in cfg["references"]:
        b = load_bundle(out / f"source_{reference}.npz")
        early = np.array([literal_path(b, x)[-1] for x in z["held_paths"][:, 0]])
        for u, path in enumerate(z["held_paths"][:, 1]):
            late = literal_path(b, path)
            for kind in ("self", "stranger"):
                origin = u if kind == "self" else int(z["donors"][u])
                for assignment in (0, 1):
                    codes = b["codes"][assignment]
                    ell = np.empty(len(codes))
                    for g in np.unique(codes):
                        ix = codes == g
                        ell[ix] = (
                            logsumexp(early[origin, ix])
                            - np.log(ix.sum())
                            - logsumexp(early[origin])
                        )
                    if kind == "self":
                        errors["projection"] = max(
                            errors["projection"],
                            float(
                                abs(
                                    ell
                                    - arrays[f"{reference}/boundary_logweights"][
                                        u, assignment + 2
                                    ]
                                ).max()
                            ),
                        )
                    joint = late + ell
                    denominator = logsumexp(joint, axis=-1)
                    logs = np.diff(denominator)
                    errors["likelihood"] = max(
                        errors["likelihood"],
                        float(
                            abs(
                                logs
                                - arrays[f"{reference}/{kind}/logp"][u, assignment + 2]
                            ).max()
                        ),
                    )
                    probabilities = np.exp(joint - denominator[:, None])
                    profiles = np.array(
                        [B.profile(w, b["codes"]) for w in probabilities]
                    ).transpose(1, 0, 2)
                    errors["profile"] = max(
                        errors["profile"],
                        float(
                            abs(
                                profiles
                                - arrays[f"{reference}/{kind}/profiles"][
                                    u, assignment + 2
                                ]
                            ).max()
                        ),
                    )
                    endpoints += 1
        profiles = arrays[f"{reference}/boundary_profiles"]
        for assignment in (0, 1):
            errors["boundary"] = max(
                errors["boundary"],
                float(
                    abs(
                        profiles[:, assignment + 2, assignment]
                        - profiles[:, 1, assignment]
                    ).max()
                ),
            )
        for kind in ("self", "stranger"):
            for key in TRACE_KEYS:
                errors["old_replay"] = max(
                    errors["old_replay"],
                    float(
                        abs(
                            arrays[f"{reference}/{kind}/{key}"][:, :2]
                            - old[f"{reference}/{kind}/{key}"][:, :2]
                        ).max()
                    ),
                )
    if max(errors.values()) > 1e-10 or not all(
        np.isfinite(v).all() for v in arrays.values()
    ):
        raise ArithmeticError(json.dumps(errors))
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_profile_episode_cold.py"),
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
    cold_error = 0.0
    for k, name in enumerate(M.MODELS):
        row = cold["models"][name]
        cold_error = max(
            cold_error,
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
    if cold_error > 1e-12:
        raise ArithmeticError("Cold episode model differs")
    report = {
        "PASS": True,
        "all_new_complete_path_endpoints": endpoints,
        "errors": errors,
        "cold_error": cold_error,
        "same_agent_not_blind": True,
        "no_scientific_rerun": True,
        "original_scope_complete": False,
    }
    R.write(out / "verification.json", report)
    R.write(out / "cold_readback.json", cold)
    R.seal(out, "verification_seal.json", ("verification.json", "cold_readback.json"))
    print(json.dumps(report))


def main():
    """Execute one immutable packet's stages; never queue a follow-up experiment."""
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
        print("HM283 seals PASS")
    else:
        globals()[args.stage](args.output)


if __name__ == "__main__":
    main()
