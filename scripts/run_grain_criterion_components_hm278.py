#!/usr/bin/env python3
"""One finite target-meaning discriminator on all frozen HM277 author readings."""

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
from grain_research.construction import math_author_pair_criterion as A
from grain_research.construction import math_criterion_components as M
from grain_research.construction import math_source_event_coupling as P
from scripts import run_grain_author_pair_criterion_hm277 as OLD
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_criterion_components_hm278.json"
DEFAULT = ROOT / "results/grain_math_response_v1/criterion_components_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_CRITERION_COMPONENTS_PLAN_HM278.md",
    "grain_research/construction/math_criterion_components.py",
    "grain_research/construction/math_source_anchor_exchange.py",
    "grain_research/construction/math_profile_reader.py",
    "grain_research/construction/math_author_pair_criterion.py",
    "grain_research/construction/math_source_event_coupling.py",
    "grain_research/construction/math_relational_diagnostic_update.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_author_pair_criterion_hm277.py",
    "scripts/run_grain_criterion_components_hm278.py",
    "scripts/score_grain_criterion_components_cold.py",
    "tests/test_grain_criterion_components.py",
)


def pair_weights(data):
    """Retain the two old window weight differences and their all22 mean."""
    w = data["source_weights"]
    d = w[:, :, :, 0] - w[:, :, :, 1]
    return np.concatenate((d, d.mean(2, keepdims=True)), 2)


def prepare(out):
    """Bind the existing full scope and new criterion question before scoring."""
    if out.exists():
        raise FileExistsError("Preserve any prior result or failure")
    cfg = R.read(ROOT / CONFIG)
    for parent in (cfg["parent"], cfg["source_parent"]):
        R.check(ROOT / parent, ("computed_seal.json", "verification_seal.json"))
    data = OLD.load(ROOT / cfg["parent"] / "readings.npz")
    if (
        data["source_weights"].shape != (2, 386, 2, 3, 386)
        or data["codes"].shape != (386,)
        or len(np.unique(data["source_indices"])) != 386
        or [int(np.sum(data["banks"] == b)) for b in (0, 1)] != [203, 183]
    ):
        raise ValueError("Changed historical author scope")
    M.criterion_design(P.BITS[data["codes"]], data["banks"])
    out.mkdir(parents=True)
    inputs = [
        f"{cfg['parent']}/{p}"
        for p in (
            "readings.npz",
            "manifest.json",
            "computed_seal.json",
            "verification_seal.json",
            "example_author.json",
        )
    ] + [f"{cfg['source_parent']}/source_pair_bank.npz"]
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {p: R.digest(ROOT / p) for p in inputs},
            "original_scope_complete": False,
        },
    )
    R.write(
        out / "input_quality.json",
        {
            "authors": 386,
            "overlapping_pairs": 8492,
            "all32_source_profiles_per_bank": True,
            "no_new_embeddings_or_kernel_parameters": True,
            "no_new_neural_API_private_training": True,
            "historical_exploratory_not_lockbox": True,
            "conditional_target_is_not_causal_target": True,
        },
    )
    R.seal(out, "prepared_seal.json", ("input_quality.json",))
    print("PREPARED HM278: all386 authors, conditional criterion question", flush=True)


def expired(_signum, _frame):
    """Stop at the original cap without scheduling a larger follow-up."""
    raise TimeoutError("Original120s cap reached")


def compute(out):
    """Keep shared/conditional target meaning and test the full declared family."""
    cfg = R.check(out)["config"]
    data = OLD.load(ROOT / cfg["parent"] / "readings.npz")
    start = time.monotonic()
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(cfg["max_seconds"])
    d = pair_weights(data)
    labels, banks = P.BITS[data["codes"]], data["banks"]
    scores, means, departure, groups = M.score_components(d, labels, banks)
    np.testing.assert_allclose(
        scores[:, :, :, 0], data["series"][:, :, :, 4], atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        scores[:, :, :, 0], scores[:, :, :, 1] + scores[:, :, :, 2], atol=1e-12, rtol=0
    )
    observed, null, variance, nominal, holm, assignments = M.assignment_reference(
        d[:, :, 2], labels, banks, cfg["permutations_per_target"], cfg["seed"]
    )
    rows = []
    for sp, name in enumerate(cfg["spaces"]):
        for win, window in enumerate(A.WINDOWS):
            for ch, channel in enumerate(M.CHANNELS):
                for group in ("within_banks", "bank0", "bank1"):
                    ix = (
                        np.ones(386, bool)
                        if group == "within_banks"
                        else banks == int(group[-1])
                    )
                    x, y, e, g, b = (
                        scores[sp, ix, win, ch],
                        labels[ix],
                        departure[ix],
                        groups[ix],
                        banks[ix],
                    )
                    raw_r = A.corr(x, y, b)
                    conditional_r = M.conditional_r(x, e, g)
                    raw_covariance = np.empty(5)
                    for q in range(5):
                        xx, yy = x[:, q].copy(), y[:, q].copy()
                        for bank in np.unique(b):
                            mask = b == bank
                            xx[mask] -= xx[mask].mean()
                            yy[mask] -= yy[mask].mean()
                        raw_covariance[q] = np.mean(xx * yy)
                    conditional_covariance = np.mean(x * e, 0)
                    for q, target in enumerate(cfg["targets"]):
                        rows.append(
                            {
                                "space": name,
                                "window": window,
                                "channel": channel,
                                "group": group,
                                "target": target,
                                "authors": int(ix.sum()),
                                "raw_within_bank_r": OLD.number(raw_r[q]),
                                "within_other_four_r": OLD.number(conditional_r[q]),
                                "raw_covariance": float(raw_covariance[q]),
                                "conditional_covariance": float(
                                    conditional_covariance[q]
                                ),
                                "score_mean": float(x[:, q].mean()),
                                "score_sd": float(np.std(x[:, q])),
                            }
                        )
    primary = []
    for sp, name in enumerate(cfg["spaces"]):
        r = M.conditional_r(scores[sp, :, 2, 2], departure, groups)
        for q, target in enumerate(cfg["targets"]):
            if variance[sp, q] < 0:
                raise ValueError("Negative exact variance; no clipping rescue")
            sd = float(np.sqrt(variance[sp, q]))
            primary.append(
                {
                    "space": name,
                    "target": target,
                    "conditional_covariance": float(observed[sp, q]),
                    "within_other_four_r": OLD.number(r[q]),
                    "exact_orbit_sd": sd,
                    "orbit_sd_ratio_not_Gaussian_z": float(observed[sp, q] / sd)
                    if sd
                    else None,
                    "conditional_assignment_p": float(nominal[sp, q]),
                    "Holm_10cell_p": float(holm[sp, q]),
                }
            )
    R.archive(
        out / "scores.npz",
        scores=scores,
        source_means=means,
        criterion_departure=departure,
        groups=groups,
        labels=labels,
        banks=banks,
        source_indices=data["source_indices"],
    )
    R.archive(
        out / "inference.npz",
        observed=observed,
        null=null,
        variance=variance,
        nominal=nominal,
        holm=holm,
        assignments=assignments,
    )
    R.table(out / "all_accounts.csv", rows)
    R.table(out / "primary.csv", primary)
    cold = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                str(ROOT / "scripts/score_grain_criterion_components_cold.py"),
                "--bundle",
                str(ROOT / cfg["source_parent"] / "source_pair_bank.npz"),
                "--record",
                str(ROOT / cfg["parent"] / "example_author.json"),
                "--source-bank",
                "1",
                "--space",
                "GRAIN15",
            ],
            text=True,
        )
    )
    u = int(np.flatnonzero(banks == 0)[0])
    np.testing.assert_allclose(cold["values"], scores[1, u], atol=1e-12, rtol=0)
    R.write(out / "cold_readback.json", cold)
    summary = {
        "packet": "HM278",
        "authors": 386,
        "primary": primary,
        "wall_seconds": time.monotonic() - start,
        "reconstruction_max": float(
            np.max(abs(scores[:, :, :, 0] - scores[:, :, :, 1] - scores[:, :, :, 2]))
        ),
        "old_total_max_error": float(
            np.max(abs(scores[:, :, :, 0] - data["series"][:, :, :, 4]))
        ),
        "no_components_deleted": True,
        "no_weight_sign_or_precision_search": True,
        "historical_exploratory_reference": True,
        "new_neural_API_private_training_calls": 0,
        "original_scope_complete": False,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "computed_seal.json",
        (
            "scores.npz",
            "inference.npz",
            "all_accounts.csv",
            "primary.csv",
            "cold_readback.json",
            "summary.json",
        ),
    )
    signal.alarm(0)
    print(json.dumps(summary, indent=2), flush=True)


def verify(out):
    """Recompute direct author sums and orbit variance by stratum block centering."""
    cfg = R.check(out, ("computed_seal.json",))["config"]
    old = OLD.load(ROOT / cfg["parent"] / "readings.npz")
    result, inference = OLD.load(out / "scores.npz"), OLD.load(out / "inference.npz")
    d, y, b = pair_weights(old), result["labels"], result["banks"]
    means, groups = np.empty_like(y), np.empty(y.shape, int)
    for bank in (0, 1):
        ix = np.flatnonzero(b == bank)
        for q in range(5):
            other = [k for k in range(5) if k != q]
            codes = sum(y[ix, k].astype(int) * 2**j for j, k in enumerate(other))
            groups[ix, q] = codes + 16 * bank
            for cell in range(16):
                jj = ix[codes == cell]
                means[jj, q] = sum(y[jj, q]) / len(jj)
    e = y - means
    np.testing.assert_allclose(means, result["source_means"], atol=1e-15, rtol=0)
    errors = {"source_sums": 0.0, "orbit_variance": 0.0, "assignment_statistics": 0.0}
    for sp in range(2):
        for u in range(386):
            src = np.flatnonzero(b != b[u])
            np.testing.assert_array_equal(d[sp, u, :, b == b[u]], 0)
            for win in range(3):
                for q in range(5):
                    direct = [
                        sum(d[sp, u, win, j] * v[j, q] for j in src)
                        for v in (y, means, e)
                    ]
                    errors["source_sums"] = max(
                        errors["source_sums"],
                        float(np.max(abs(direct - result["scores"][sp, u, win, :, q]))),
                    )
        ix, jx = np.flatnonzero(b == 0), np.flatnonzero(b == 1)
        h = d[sp, ix, 2][:, jx] + d[sp, jx, 2][:, ix].T
        for q in range(5):
            variance = 0.0
            for c in range(16):
                ii = np.flatnonzero(groups[ix, q] == c)
                p0, n0 = y[ix[ii], q].mean(), len(ii)
                for cc in range(16):
                    jj = np.flatnonzero(groups[jx, q] == cc + 16)
                    p1, n1 = y[jx[jj], q].mean(), len(jj)
                    block = h[np.ix_(ii, jj)]
                    centered = (
                        block - block.mean(0) - block.mean(1)[:, None] + block.mean()
                    )
                    variance += (
                        p0
                        * (1 - p0)
                        * n0
                        / (n0 - 1)
                        * p1
                        * (1 - p1)
                        * n1
                        / (n1 - 1)
                        * np.sum(centered**2)
                        / 386**2
                    )
            errors["orbit_variance"] = max(
                errors["orbit_variance"],
                abs(float(variance) - inference["variance"][sp, q]),
            )
            for draw, permutation in enumerate(inference["assignments"][q]):
                np.testing.assert_array_equal(np.sort(permutation), np.arange(386))
                np.testing.assert_array_equal(b[permutation], b)
                np.testing.assert_array_equal(
                    y[permutation][:, [k for k in range(5) if k != q]],
                    y[:, [k for k in range(5) if k != q]],
                )
                np.testing.assert_array_equal(
                    np.bincount(old["codes"][permutation], minlength=32),
                    np.bincount(old["codes"], minlength=32),
                )
                pe = e[permutation, q]
                value = sum(pe[u] * np.dot(d[sp, u, 2], pe) for u in range(386)) / 386
                errors["assignment_statistics"] = max(
                    errors["assignment_statistics"],
                    abs(float(value) - inference["null"][draw, sp, q]),
                )
    if max(errors.values()) > 1e-12:
        raise ValueError("Alternate arithmetic failed; retain output")
    R.write(
        out / "verification.json",
        {
            "alternate_errors": errors,
            "authors": 386,
            "conditional_target_assignments": 995,
            "all_components_retained": True,
            "same_agent_not_blind": True,
        },
    )
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(R.read(out / "verification.json"), indent=2), flush=True)


def main():
    """Run only immutable finite stages, preserving a failure instead of rewriting it."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "compute", "verify", "check"))
    p.add_argument("--output", type=Path, default=DEFAULT)
    a = p.parse_args()
    try:
        if a.mode == "check":
            R.check(a.output, ("computed_seal.json", "verification_seal.json"))
            print("CHECKED HM278", flush=True)
        else:
            globals()[a.mode](a.output)
    except Exception as exc:
        if a.output.exists() and not (a.output / "failure.json").exists():
            R.write(a.output / "failure.json", {"stage": a.mode, "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
