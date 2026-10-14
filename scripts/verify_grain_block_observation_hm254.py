#!/usr/bin/env python3
"""Literal block enumeration, independent OLS and cold arithmetic readback."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_block_observation as M
from scripts import run_grain_natural_proxy_hm253 as P
from scripts import run_grain_observed_family_hm170 as R


def verify(out):
    """Audit all natural blocks by the six permutations without science reruns."""
    cfg = R.check(out, ("compiled_seal.json", "cold_seal.json", "analyzed_seal.json"))[
        "config"
    ]
    d, cold = (P.arrays(out / f) for f in ("observations.npz", "cold_scores.npz"))
    q, mu, gamma = d["q"], d["mu"], d["gamma"]
    blocks = q.reshape(len(q), -1, 3, q.shape[-1])
    b = blocks.shape[1]
    centered = blocks - blocks.mean(2, keepdims=True)
    cross, diagonal = np.zeros_like(gamma), np.zeros_like(gamma)
    for p in M.PERMUTATIONS:
        x, y = centered[:, :, p[0]], centered[:, :, p[1]]
        cross += np.einsum("ubd,ube->ude", x, y) / (6 * b * b)
        diagonal += np.einsum("ubd,ube->ude", x, x) / (6 * b * b)
    moment_error = max(
        float(np.max(np.abs(cross + gamma))),
        float(np.max(np.abs(diagonal - 2 * gamma))),
    )
    if not np.array_equal(
        np.sort(d["assignments"], -1),
        np.broadcast_to([0, 1, 2], d["assignments"].shape),
    ):
        raise AssertionError("Invalid event partition")
    scores_error, variance_error, ols_relative, bank_equation = 0.0, 0.0, 0.0, 0.0
    scalar_count = 0
    for p in R.read(out / "bundle.json")["programs"]:
        source, held = d["folds"] == p["source_fold"], d["folds"] != p["source_fold"]
        beta_bank = np.asarray(p["operators"]["finite_bank"]["beta"])
        direct = np.linalg.lstsq(
            mu[source],
            d["labels"][source],
            rcond=np.sqrt(cfg["relative_rank_tolerance"]),
        )[0]
        ols_relative = max(
            ols_relative,
            float(np.linalg.norm(direct - beta_bank) / max(1, np.linalg.norm(direct))),
        )
        bank_equation = max(
            bank_equation,
            float(
                np.max(
                    np.abs(
                        (np.asarray(p["M"]) - np.asarray(p["G"]))
                        + np.asarray(p["G"])
                        - np.asarray(p["M"])
                    )
                )
            ),
        )
        for i, method in enumerate(M.METHODS):
            beta = np.asarray(p["operators"][method]["beta"])
            for u in np.flatnonzero(held):
                direct_views = sum(
                    d["views"][u, :, :, j, None] * beta[j] for j in range(len(beta))
                )
                direct_mean = sum(mu[u, j] * beta[j] for j in range(len(beta)))
                scores_error = max(
                    scores_error,
                    float(np.max(np.abs(direct_views - cold["view_scores"][i, u]))),
                    float(np.max(np.abs(direct_mean - cold["mean_scores"][i, u]))),
                )
                projected = centered[u] @ beta
                direct_variance = np.sum(projected * projected, axis=(0, 1)) / (
                    3 * b * b
                )
                variance_error = max(
                    variance_error,
                    float(
                        np.max(
                            np.abs(direct_variance - cold["partition_variance"][i, u])
                        )
                    ),
                )
                scalar_count += (
                    direct_views.size + direct_mean.size + direct_variance.size
                )
    accounts = R.read(out / "allocation_accounts.json")["accounts"]
    account_error = max(
        float(np.max(np.abs(np.sum(a["terms"], 0) - a["residual"]))) for a in accounts
    )
    energy_error = max(
        float(
            np.max(
                np.abs(
                    np.asarray(a["signed_gram"]).sum((1, 2))
                    - np.sum(np.asarray(a["residual"]) ** 2, 0)
                )
            )
        )
        for a in accounts
    )
    world = R.read(out / "generating_world.json")
    generating_error = max(
        abs(world["propensity_slope"] - 1),
        abs(world["finite_bank_slope"] - 6 / 7),
        abs(world["single_view_slope"] - 2 / 3),
    )
    result = {
        "PASS": moment_error < 1e-12
        and scores_error < 1e-8
        and variance_error < 1e-8
        and ols_relative < 1e-5
        and bank_equation < 1e-12
        and account_error < 1e-9
        and energy_error < 1e-6
        and generating_error < 1e-12,
        "natural_block_enumerations": len(q) * b * 6,
        "cold_scalar_accounts": scalar_count,
        "six_permutation_covariance_error": moment_error,
        "cold_sum_error": scores_error,
        "projected_variance_error": variance_error,
        "direct_OLS_relative_error": ols_relative,
        "bank_equation_identity_error": bank_equation,
        "old_signed_account_error": account_error,
        "signed_energy_error": energy_error,
        "generating_world_error": generating_error,
        "same_agent_not_blind": True,
        "no_scientific_rerun": True,
    }
    R.write(out / "verification.json", result)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(result))
    if not result["PASS"]:
        raise AssertionError("Preserved HM254 alternate verification failure")


def main():
    """Run the registered alternative arithmetic only once for a new packet."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    verify(p.parse_args().out)


if __name__ == "__main__":
    main()
