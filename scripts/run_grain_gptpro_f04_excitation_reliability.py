#!/usr/bin/env python3
"""F04 excitation / reliability / within-between packet (R06, R12, P02 p.5 omission).

Frozen before running (2026-09-26, Claude Opus 5.5). Linear response world
y = a + b z + e (a~N(0,1), b~N(.5,.5), e~N(0,.5^2)), the F04 L-world inputs.
1. Persistent excitation (R06, P02 pp.27-28): rank of sum phi phi' and slope
   MSE for 1,000 readings over 4 conditions vs 1,000,000 readings at one
   condition (exact posterior with the source prior b~N(.5,.25)).
2. Reliability curves (R12, P01 p.21, P02 pp.29-30): split-half reliability of
   the person slope (block halves) versus tokens (readings 8..64 at 4 conditions) and versus
   context diversity at fixed tokens (32 readings over K=1,2,4,8 conditions),
   next to single-reading reliability.
3. Within vs between structure (omission, P02 p.5): a world where two response
   features correlate +.6 between persons and -.6 within persons; the check
   compares the correlation of person means with the pooled within-person
   correlation; a matched world where both are +.6 is the control.
Synthetic only; seeds SeedSequence([20260926, 4, k]).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def slope_posterior_mse(z, sigma2=0.25, prior_var=0.25, reps=2000, rng=None):
    """Exact Gaussian posterior MSE for b with a known (conditioning on a) is optimistic; use joint (a,b)."""
    x = np.column_stack([np.ones_like(z), z])
    prior_prec = np.diag([1.0, 1 / prior_var])
    post_cov = np.linalg.inv(prior_prec + x.T @ x / sigma2)
    return float(post_cov[1, 1]), int(np.linalg.matrix_rank(x.T @ x))


def reliability(n, k, persons=400, sims=60, rng=None):
    """Split-half reliability of OLS slope (k conditions evenly over [-1,1], n readings)."""
    zs = np.linspace(-1, 1, k) if k > 1 else np.array([1.0])
    z = np.resize(zs, n)
    rs, single = [], []
    for _ in range(sims):
        a = rng.normal(0, 1, persons)
        b = rng.normal(0.5, 0.5, persons)
        y = a[:, None] + b[:, None] * z[None, :] + rng.normal(0, 0.5, (persons, n))
        if k == 1:
            rs.append(np.nan)
        else:
            half = [np.arange(0, n // 2), np.arange(n // 2, n)]  # block halves keep every condition in each half
            sl = []
            for h in half:
                zc = z[h] - z[h].mean()
                sl.append(((y[:, h] - y[:, h].mean(1, keepdims=True)) * zc).sum(1) / (zc**2).sum())
            rs.append(np.corrcoef(sl[0], sl[1])[0, 1])
        single.append(np.corrcoef(y[:, 0], y[:, k if n > k else 1])[0, 1])
    r = np.nanmean(rs) if k > 1 else float("nan")
    return {"readings": n, "conditions": k, "slope_split_half_r": float(r), "slope_spearman_brown": float(2 * r / (1 + r)) if k > 1 else None,
            "single_reading_r": float(np.mean(single))}


def within_between(rho_between, rho_within, persons=300, per_person=40, sims=100, rng=None):
    """Correlation of person means vs pooled within-person correlation."""
    out_b, out_w = [], []
    for _ in range(sims):
        mb = rng.multivariate_normal([0, 0], [[1, rho_between], [rho_between, 1]], persons)
        dev = rng.multivariate_normal([0, 0], [[1, rho_within], [rho_within, 1]], (persons, per_person))
        y = mb[:, None, :] + dev
        means = y.mean(1)
        out_b.append(np.corrcoef(means[:, 0], means[:, 1])[0, 1])
        centered = (y - means[:, None, :]).reshape(-1, 2)
        out_w.append(np.corrcoef(centered[:, 0], centered[:, 1])[0, 1])
    return {"true_between": rho_between, "true_within": rho_within, "between_estimate": float(np.mean(out_b)), "within_estimate": float(np.mean(out_w)),
            "structures_differ": bool(abs(np.mean(out_b) - np.mean(out_w)) > 0.2)}


def main():
    """Write excitation_reliability.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(np.random.SeedSequence([20260926, 4, 1]))
    diverse_var, diverse_rank = slope_posterior_mse(np.resize(np.array([-1, -1 / 3, 1 / 3, 1.0]), 1000))
    homog_var, homog_rank = slope_posterior_mse(np.ones(1_000_000))
    excitation = {"diverse_1000_over_4_conditions": {"posterior_var_b": diverse_var, "rank": diverse_rank},
                  "homogeneous_1e6_one_condition": {"posterior_var_b": homog_var, "rank": homog_rank},
                  "prior_var_b": 0.25, "note": "with one condition the slope and level are confounded: posterior variance of b stays near the prior-level trade-off regardless of N"}
    tokens = [reliability(n, 4, rng=rng) for n in (8, 16, 32, 64)]
    diversity = [reliability(32, k, rng=rng) for k in (1, 2, 4, 8)]
    wb = [within_between(0.6, -0.6, rng=rng), within_between(0.6, 0.6, rng=rng)]
    out = {"R06_persistent_excitation": excitation, "R12_reliability_vs_tokens": tokens, "R12_reliability_vs_diversity_fixed_32": diversity, "within_between_check": wb}
    (args.output / "excitation_reliability.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
