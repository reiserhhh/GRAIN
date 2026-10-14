#!/usr/bin/env python3
"""One cross-fitted real-author criterion test, with no optimization grid."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_pair_criterion as M
from grain_research.construction import math_source_event_coupling as P
from grain_research.construction.math_relational_diagnostic_update import TARGETS
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_author_pair_criterion_hm277.json"
DEFAULT = ROOT / "results/grain_math_response_v1/author_pair_criterion_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_AUTHOR_PAIR_CRITERION_PLAN_HM277.md",
    "grain_research/construction/math_author_pair_criterion.py",
    "grain_research/construction/math_source_event_coupling.py",
    "grain_research/construction/math_relational_diagnostic_update.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_author_pair_criterion_hm277.py",
    "scripts/score_grain_author_pair_criterion_cold.py",
    "scripts/verify_grain_author_pair_criterion_hm277.py",
    "tests/test_grain_author_pair_criterion.py",
)


def number(v):
    """Represent undefined metrics explicitly rather than as JSON NaN or zero."""
    return float(v) if np.isfinite(v) else None


def load(path):
    """Read this packet's explicitly qualified public numerical artifact."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def prepare(out):
    """Freeze source identity, complete author scope and method before any scoring."""
    if out.exists():
        raise FileExistsError("Never replace a frozen or failed run")
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["source_parent"]
    R.check(parent, ("computed_seal.json", "verification_seal.json"))
    data = load(parent / "source_pair_bank.npz")
    if (
        data["prototypes"].shape != (386, 22, 2, 1024)
        or data["projection"].shape != (1024, 15)
        or data["codes"].shape != (386,)
        or len(np.unique(data["source_indices"])) != 386
        or set(np.unique(data["folds"])) != {0, 1}
        or float(data["variance"]) != cfg["variance"]
    ):
        raise ValueError("Changed frozen source scope or operator")
    for bank in (0, 1):
        if np.bincount(data["codes"][data["folds"] == bank], minlength=32).min() < 1:
            raise ValueError("All32 profiles need source support in both banks")
    left, right, relink = P.pair_indices(data["times"], data["conditions"])
    for k, value in (("left", left), ("right", right), ("relink", relink)):
        np.testing.assert_array_equal(data[k], value)
    out.mkdir(parents=True)
    inputs = (
        "source_pair_bank.npz",
        "manifest.json",
        "prepared_seal.json",
        "computed_seal.json",
        "verification_seal.json",
    )
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {
                str((parent / p).relative_to(ROOT)): R.digest(parent / p)
                for p in inputs
            },
            "cohort": "historical386_SOURCE,exploratory_not_lockbox",
            "original_scope_complete": False,
        },
    )
    R.write(
        out / "input_quality.json",
        {
            "authors": 386,
            "comments": 9264,
            "overlapping_pairs": 8492,
            "banks": [int(np.sum(data["folds"] == k)) for k in (0, 1)],
            "frozen_geometry_previously_explored": True,
            "criterion": "historical_binary_Big5_profile_not_paired_behavior",
            "criteria_used_for_score": "opposite_bank_only",
            "native24_panel_used": False,
            "no_new_neural_API_private_training": True,
        },
    )
    u = int(np.flatnonzero(data["folds"] == 0)[0])
    R.write(
        out / "example_author.json",
        {
            "author_cache_id": int(data["source_indices"][u]),
            "sampling_regime": "next_cached_record_within_registered_window",
            "pairs": data["prototypes"][u].tolist(),
        },
    )
    R.seal(out, "prepared_seal.json", ("input_quality.json", "example_author.json"))
    print("PREPARED HM277: 386 authors, two opposite-bank scorers", flush=True)


def expired(_signum, _frame):
    """Stop the finite computation rather than automatically extending its budget."""
    raise TimeoutError("Original240s finite computation budget reached")


def compute(out):
    """Score every held author once before computing criterion associations."""
    cfg = R.check(out)["config"]
    data = load(ROOT / cfg["source_parent"] / "source_pair_bank.npz")
    start = time.monotonic()
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(cfg["max_seconds"])
    profiles = np.empty((2, 386, 22, 4, 32))
    weights = np.zeros((2, 386, 2, 3, 386))
    for space, name in enumerate(cfg["spaces"]):
        z = data["prototypes"]
        if name == "GRAIN15":
            z = z @ data["projection"]
        for bank in (0, 1):
            app = np.flatnonzero(data["folds"] == bank)
            src = np.flatnonzero(data["folds"] != bank)
            for begin in range(0, len(app), cfg["author_block"]):
                ix = app[begin : begin + cfg["author_block"]]
                logs = P.event_logs(
                    z[ix].reshape(-1, 2, z.shape[-1]), z[src], cfg["variance"]
                )
                reading = P.profile_readings(
                    logs, data["codes"][src], data["relink"][src]
                )
                profiles[space, ix] = reading["profile"].reshape(len(ix), 22, 4, 32)
                w = M.source_author_weights(logs, data["relink"][src])
                w = w.reshape(len(ix), 2, 11, 3, len(src)).mean(2)
                for i, u in enumerate(ix):
                    weights[space, u][..., src] = w[i]
            print(f"SCORED {name} held bank{bank}: {len(app)} authors", flush=True)
    aggregate = M.aggregate_profiles(profiles)
    series = M.all_series(aggregate @ P.BITS)
    all_weights = weights.mean(2)
    difference = all_weights[:, :, 0] - all_weights[:, :, 1]
    labels = P.BITS[data["codes"]]
    np.testing.assert_allclose(
        difference @ labels, series[:, :, 2, 4], rtol=0, atol=1e-12
    )
    observed, null, nominal, family, permutations = M.permutation_inference(
        difference, data["codes"], data["folds"], cfg["permutations"], cfg["seed"]
    )
    ci, boot, samples = M.bootstrap_primary(
        series[:, :, 2, 4], labels, data["folds"], cfg["bootstrap"], cfg["seed"] + 1
    )
    match, matching_features, matching_distance = M.matched_strangers(
        data["conditions"], data["folds"]
    )
    rows, stability = [], []
    for sp, name in enumerate(cfg["spaces"]):
        for win, window in enumerate(M.WINDOWS):
            for s, method in enumerate(M.SERIES):
                for group in ("pooled", "within_banks", "bank0", "bank1"):
                    ix = (
                        np.ones(386, bool)
                        if group in ("pooled", "within_banks")
                        else data["folds"] == int(group[-1])
                    )
                    x, y = series[sp, ix, win, s], labels[ix]
                    banks = data["folds"][ix] if group == "within_banks" else None
                    r, rho = M.corr(x, y, banks), M.corr(x, y, banks, ranks=True)
                    for q, target in enumerate(TARGETS):
                        err = x[:, q] - y[:, q]
                        rows.append(
                            {
                                "space": name,
                                "window": window,
                                "series": method,
                                "group": group,
                                "target": target,
                                "authors": int(ix.sum()),
                                "pearson": number(r[q]),
                                "spearman": number(rho[q]),
                                "mean": float(x[:, q].mean()),
                                "rms": float(np.sqrt(np.mean(x[:, q] ** 2))),
                                "mae": float(np.mean(abs(err))) if s < 4 else None,
                                "rmse": float(np.sqrt(np.mean(err**2)))
                                if s < 4
                                else None,
                                "brier": float(np.mean(err**2)) if s < 4 else None,
                            }
                        )
        for s, method in enumerate(M.SERIES):
            a, b = series[sp, :, 0, s], series[sp, :, 1, s]
            same = M.corr(a, b, data["folds"])
            stranger = M.corr(a, b[match], data["folds"])
            for q, target in enumerate(TARGETS):
                stability.append(
                    {
                        "space": name,
                        "series": method,
                        "target": target,
                        "same_author_r": number(same[q]),
                        "coarse_matched_stranger_r": number(stranger[q]),
                        "difference": number(same[q] - stranger[q]),
                    }
                )
    primary = []
    for sp, name in enumerate(cfg["spaces"]):
        for q, target in enumerate(TARGETS):
            primary.append(
                {
                    "space": name,
                    "target": target,
                    "within_bank_r": float(observed[sp, q]),
                    "conditional_author_ci_low": number(ci[0, sp, q]),
                    "conditional_author_ci_high": number(ci[1, sp, q]),
                    "permutation_p_two_sided": float(nominal[sp, q]),
                    "maxabs_10cell_p": float(family[sp, q]),
                    "bank0_r": number(
                        M.corr(
                            series[sp, data["folds"] == 0, 2, 4],
                            labels[data["folds"] == 0],
                        )[q]
                    ),
                    "bank1_r": number(
                        M.corr(
                            series[sp, data["folds"] == 1, 2, 4],
                            labels[data["folds"] == 1],
                        )[q]
                    ),
                }
            )
    R.archive(
        out / "readings.npz",
        pair_profiles=profiles,
        aggregate_profiles=aggregate,
        series=series,
        source_weights=weights,
        codes=data["codes"],
        banks=data["folds"],
        source_indices=data["source_indices"],
        matching=match,
        matching_features=matching_features,
        matching_distance=matching_distance,
    )
    R.archive(
        out / "inference.npz",
        null=null,
        profile_permutations=permutations,
        bootstrap=boot,
        bootstrap_samples=samples,
        observed=observed,
        ci=ci,
    )
    R.table(out / "all_associations.csv", rows)
    R.table(out / "primary_association.csv", primary)
    R.table(out / "window_stability.csv", stability)
    score_rows = [
        {
            "space": name,
            "author_cache_id": int(data["source_indices"][u]),
            "evaluation_bank": int(data["folds"][u]),
            "window": window,
            "series": method,
            **{
                target: float(series[sp, u, win, s, q])
                for q, target in enumerate(TARGETS)
            },
        }
        for sp, name in enumerate(cfg["spaces"])
        for u in range(386)
        for win, window in enumerate(M.WINDOWS)
        for s, method in enumerate(M.SERIES)
    ]
    R.table(out / "author_scores.csv", score_rows)
    cold = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                str(ROOT / "scripts/score_grain_author_pair_criterion_cold.py"),
                "--bundle",
                str(ROOT / cfg["source_parent"] / "source_pair_bank.npz"),
                "--record",
                str(out / "example_author.json"),
                "--source-bank",
                "1",
                "--space",
                "GRAIN15",
            ],
            text=True,
        )
    )
    np.testing.assert_allclose(
        cold["series"],
        series[1, int(np.flatnonzero(data["folds"] == 0)[0])],
        atol=1e-12,
        rtol=0,
    )
    R.write(out / "cold_readback.json", cold)
    own = max(
        float(np.max(abs(weights[:, u, :, :, data["folds"] == data["folds"][u]])))
        for u in range(386)
    )
    summary = {
        "packet": "HM277",
        "authors": 386,
        "primary": primary,
        "pair_readings": int(profiles.shape[0] * profiles.shape[1] * profiles.shape[2]),
        "wall_seconds": time.monotonic() - start,
        "no_hyperparameter_or_sign_search": True,
        "intervals_conditional_on_frozen_scores": True,
        "permutation_unit": "whole_author_profile_within_historical_bank_source_and_application",
        "primary_family": "two_spaces_x_five_traits_all22pair_FminusJ",
        "undefined_bootstrap_cells": int(np.sum(~np.isfinite(boot))),
        "own_bank_weight_max": own,
        "new_neural_API_private_training_calls": 0,
        "original_scope_complete": False,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "computed_seal.json",
        (
            "readings.npz",
            "inference.npz",
            "all_associations.csv",
            "primary_association.csv",
            "window_stability.csv",
            "author_scores.csv",
            "cold_readback.json",
            "summary.json",
        ),
    )
    signal.alarm(0)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    """Perform registered immutable stages, preserving any failure for inspection."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "compute", "verify", "check"))
    p.add_argument("--output", type=Path, default=DEFAULT)
    a = p.parse_args()
    try:
        if a.mode == "prepare":
            prepare(a.output)
        elif a.mode == "compute":
            compute(a.output)
        elif a.mode == "verify":
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/verify_grain_author_pair_criterion_hm277.py"),
                    "--output",
                    str(a.output),
                ],
                check=True,
            )
        else:
            R.check(a.output, ("computed_seal.json", "verification_seal.json"))
            print("CHECKED HM277", flush=True)
    except Exception as e:
        if a.output.exists() and not (a.output / "failure.json").exists():
            R.write(
                a.output / "failure.json",
                {"mode": a.mode, "exception": type(e).__name__, "message": str(e)},
            )
        raise


if __name__ == "__main__":
    main()
