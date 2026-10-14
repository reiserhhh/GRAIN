#!/usr/bin/env python3
"""One finite-bank/generating-law comparison, with no precision or rank grid."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_block_observation as M
from grain_research.construction.math_natural_proxy import auc, correlation
from scripts import run_grain_natural_proxy_hm253 as P
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_block_observation_hm254.json"
DEFAULT = ROOT / "results/grain_math_response_v1/block_observation_law_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_BLOCK_OBSERVATION_PLAN_HM254.md",
    "grain_research/construction/math_block_observation.py",
    "scripts/run_grain_block_observation_hm254.py",
    "scripts/score_grain_block_observation_cold.py",
    "scripts/verify_grain_block_observation_hm254.py",
    "tests/test_grain_block_observation.py",
    "grain_research/construction/math_natural_proxy.py",
    "scripts/run_grain_natural_proxy_hm253.py",
    "scripts/run_grain_observed_family_hm170.py",
)


def prepare(out):
    """Bind all same public events and freeze the observation action before scoring."""
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    P.check(
        parent,
        (
            "compiled_seal.json",
            "cold_seal.json",
            "analyzed_seal.json",
            "verification_seal.json",
        ),
    )
    d = P.arrays(parent / "observations.npz")
    if d["coordinates"].shape != (cfg["authors"], cfg["events_per_author"], 15):
        raise ValueError("Changed source public-event scope")
    q = M.event_features(d["coordinates"], d["conditions"], d["units"])
    mu, gamma = M.orbit_moments(q)
    views, assignments = M.partition_views(q, cfg["partitions"], cfg["partition_seed"])
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {
                str((parent / p).relative_to(ROOT)): R.digest(parent / p)
                for p in (
                    "observations.npz",
                    "bundle.json",
                    "prepared_seal.json",
                    "compiled_seal.json",
                )
            },
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        q=q,
        mu=mu,
        gamma=gamma,
        views=views,
        assignments=assignments,
        folds=d["folds"],
        labels=d["labels"],
        units=d["units"],
    )
    R.archive(
        out / "cold_inputs.npz", mu=mu, gamma=gamma, views=views, folds=d["folds"]
    )
    quality = {
        "authors": len(q),
        "same_cached_events": int(q.shape[0] * q.shape[1]),
        "blocks_per_author": q.shape[1] // 3,
        "events_per_view": q.shape[1] // 3,
        "partition_draws": cfg["partitions"],
        "new_human_events": 0,
        "choice_and_conditional_mass_preserved": True,
        "view_sum_error": float(np.max(np.abs(views.sum(2) - 3 * mu[:, None]))),
        "qualification": "conditional on fixed bank; new assignments NOT new authors/events or established iid generating law",
    }
    R.write(out / "quality.json", quality)
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "cold_inputs.npz", "quality.json"),
    )
    print(json.dumps(quality))


def compile_stage(out):
    """Freeze three SOURCE operators for each author fold, before held-link analysis."""
    cfg = R.check(out)["config"]
    d = P.arrays(out / "observations.npz")
    start = time.perf_counter()
    programs = []
    for fold in (0, 1):
        train = d["folds"] == fold
        p = M.construct(
            d["mu"][train],
            d["gamma"][train],
            d["labels"][train],
            cfg["relative_rank_tolerance"],
        )
        p.update({"source_fold": fold, "source_authors": int(train.sum())})
        programs.append(p)
    R.write(
        out / "bundle.json",
        {
            "programs": programs,
            "methods": cfg["methods"],
            "targets": cfg["targets"],
            "contract": "finite bank orbit versus stipulated generating-law; no natural latent identification certified",
        },
    )
    R.write(out / "generating_world.json", M.generating_world())
    R.write(
        out / "compile_runtime.json",
        {"seconds": time.perf_counter() - start, "application_labels_used": False},
    )
    R.seal(
        out,
        "compiled_seal.json",
        ("bundle.json", "generating_world.json", "compile_runtime.json"),
    )
    print("Compiled two SOURCE programs x three fixed observation contracts")


def allocation_accounts(out, programs):
    """Keep signed protocol/allocation/original-equation accounts on old coefficients."""
    cfg = R.read(ROOT / CONFIG)
    old = R.read(ROOT / cfg["parent"] / "bundle.json")["programs"]
    accounts = []
    for prior in old:
        p = programs[prior["source_fold"]]
        mtx, g, rhs = map(np.asarray, (p["M"], p["G"], p["m"]))
        oldk, oldrhs, beta = map(
            np.asarray, (prior["kernel"], prior["moment"], prior["bridge"])
        )
        dk, dm = oldk - (mtx - g), oldrhs - rhs
        terms = np.stack([g @ beta, dm, -dk @ beta, oldk @ beta - oldrhs])
        residual = mtx @ beta - rhs
        gram = np.einsum("idq,jdq->qij", terms, terms)
        accounts.append(
            {
                "source_fold": prior["source_fold"],
                "cycle": prior["w"],
                "terms": terms.tolist(),
                "residual": residual.tolist(),
                "signed_gram": gram.tolist(),
                "closure_error": float(np.max(np.abs(terms.sum(0) - residual))),
                "qualification": "old fixed interleave not a randomized draw; no attribution of all target-link weakness",
            }
        )
    return accounts


def analyze(out):
    """Report every fixed method/target/partition, not an optimized model leaderboard."""
    cfg = R.check(out, ("compiled_seal.json", "cold_seal.json"))["config"]
    d, cold = (P.arrays(out / p) for p in ("observations.npz", "cold_scores.npz"))
    bundle = R.read(out / "bundle.json")
    start = time.perf_counter()
    target_rows, variance_rows, method_results = [], [], {}
    for i, method in enumerate(M.METHODS):
        mean_r, average_variance, variance_ratios = [], [], []
        for q, target in enumerate(cfg["targets"]):
            y, mean_score = d["labels"][:, q], cold["mean_scores"][i, :, q]
            r = correlation(y, mean_score)
            mean_r.append(r)
            target_rows.append(
                {
                    "method": method,
                    "target": target,
                    "partition": "bank",
                    "view": "bank",
                    "r": r,
                    "AUC": auc(y, mean_score),
                }
            )
            for draw in range(cfg["partitions"]):
                for view in range(3):
                    s = cold["view_scores"][i, :, draw, view, q]
                    target_rows.append(
                        {
                            "method": method,
                            "target": target,
                            "partition": draw,
                            "view": view,
                            "r": correlation(y, s),
                            "AUC": auc(y, s),
                        }
                    )
            actual = cold["partition_variance"][i, :, q]
            empirical = cold["view_scores"][i, :, :, 0, q].var(axis=1, ddof=1)
            differences = (
                cold["view_scores"][i, :, :, 0, q] - cold["view_scores"][i, :, :, 1, q]
            )
            average_variance.append(float(actual.mean()))
            variance_ratios.append(
                float(empirical.mean() / actual.mean()) if actual.mean() > 0 else None
            )
            for u in range(len(y)):
                variance_rows.append(
                    {
                        "method": method,
                        "author_row": u,
                        "target": target,
                        "exact_view_variance": actual[u],
                        "empirical_16_view_variance": empirical[u],
                        "exact_difference_variance": 3 * actual[u],
                        "empirical_16_difference_variance": differences[u].var(ddof=1),
                    }
                )
        method_results[method] = {
            "bank_r_ENACO": mean_r,
            "exact_partition_variance_mean_ENACO": average_variance,
            "empirical_over_exact_mean_variance_ENACO": variance_ratios,
            "score_min_max": [
                float(cold["mean_scores"][i].min()),
                float(cold["mean_scores"][i].max()),
            ],
        }
    accounts = allocation_accounts(out, bundle["programs"])
    R.table(out / "target_links.csv", target_rows)
    R.table(out / "sampling_uncertainty.csv", variance_rows)
    R.write(
        out / "allocation_accounts.json",
        {
            "names": [
                "finite_partition_covariance",
                "criterion_allocation",
                "cross_moment_allocation",
                "old_equation_residual",
            ],
            "accounts": accounts,
        },
    )
    R.write(
        out / "summary.json",
        {
            "authors": len(d["mu"]),
            "methods": method_results,
            "orbit_proxy_negative_eigenvalue_counts_by_source": [
                int(
                    np.sum(np.asarray(p["operators"]["orbit_proxy"]["eigenvalues"]) < 0)
                )
                for p in bundle["programs"]
            ],
            "bank_correction_is_mean_regression_by_algebra": True,
            "allocation_account_closure_max": max(a["closure_error"] for a in accounts),
            "generating_world": R.read(out / "generating_world.json"),
            "analysis_seconds": time.perf_counter() - start,
            "new_model_API_private_training": 0,
            "original_scope_complete": False,
            "decision": "close fixed-bank mean correction branch; no inference that natural latent propensity is identified or all proxy bridges are invalid",
        },
    )
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "target_links.csv",
            "sampling_uncertainty.csv",
            "allocation_accounts.json",
            "summary.json",
        ),
    )
    print(json.dumps(R.read(out / "summary.json")))


def main():
    """Execute one immutable stage or read back all scientific identities."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "compile", "analyze", "check"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    args = p.parse_args()
    if args.stage == "check":
        R.check(
            args.out,
            (
                "compiled_seal.json",
                "cold_seal.json",
                "analyzed_seal.json",
                "verification_seal.json",
            ),
        )
        print("All HM254 registered method/input/result identities unchanged")
    else:
        {"prepare": prepare, "compile": compile_stage, "analyze": analyze}[args.stage](
            args.out
        )


if __name__ == "__main__":
    main()
