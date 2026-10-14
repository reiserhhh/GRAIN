#!/usr/bin/env python3
"""R04 combined world: selection, response, state with X->s feedback, reader sensor, nu target."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_R04_COMBINED_WORLD_PROTOCOL_V1.md"
PROTOCOL_SHA = "51b08926507ca6393e5656aa37ca101a17c67042e4966637cf36628066a27c81"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F01/combined_world_r1"
BETA = np.array([-1.0, 0.0, 1.0])
RHO, ALPHA, SW, SE, SR = 0.6, 0.3, 0.4, 0.5, 0.3
N, T, SEED = 400, 40, 20260926
NUS = {"nu1_uniform": np.array([1 / 3, 1 / 3, 1 / 3]), "nu2_provoking": np.array([0.1, 0.2, 0.7])}


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def linear_map(c, a, xi):
    """r = M z + d, z = [theta, kappa, w_1..w_T, eps_1..eps_T, e_1..e_T]."""
    nz = 2 + 3 * T
    s_coef, s_const = np.zeros(nz), 0.0
    M, d = np.zeros((T, nz)), np.zeros(T)
    for t in range(T):
        x_coef = s_coef.copy()
        x_coef[0] += 1
        x_coef[1] += BETA[c[t]]
        x_coef[2 + T + t] += 1
        x_const = s_const
        M[t] = x_coef
        M[t, 2 + 2 * T + t] += 1
        d[t] = x_const
        s_coef = RHO * s_coef + xi * x_coef
        s_coef[2 + t] += 1
        s_const = RHO * s_const + ALPHA * a[t] + xi * x_const
    return M, d


def prior():
    mu = np.zeros(2 + 3 * T)
    mu[1] = 1.0
    var = np.r_[1.0, 0.16, np.full(T, SW ** 2), np.full(T, SE ** 2), np.full(T, SR ** 2)]
    return mu, var


def posterior_theta_kappa(r, c, a, xi):
    M, d = linear_map(c, a, xi)
    mu, var = prior()
    S = (M * var) @ M.T
    cross = (var[:2, None] * M[:, :2].T)
    gain = np.linalg.solve(S, cross.T).T
    return mu[:2] + gain @ (r - M @ mu - d)


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for wi, xi in enumerate((0.0, 0.25)):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, wi]))
        theta, kappa = rng.standard_normal(N), 1 + 0.4 * rng.standard_normal(N)
        truth = {k: theta + kappa * (nu @ BETA) for k, nu in NUS.items()}
        est = {e: {k: [] for k in NUS} for e in ("E1_naive", "E2_a_to_s_only", "E3_full")}
        for u in range(N):
            q = np.exp(0.8 * theta[u] * BETA)
            q /= q.sum()
            c = rng.choice(3, T, p=q)
            a = (rng.random(T) < 0.5).astype(float)
            M, d = linear_map(c, a, xi)
            z = np.r_[theta[u], kappa[u], SW * rng.standard_normal(T), SE * rng.standard_normal(T),
                      SR * rng.standard_normal(T)]
            r = M @ z + d
            for k in NUS:
                est["E1_naive"][k].append(r.mean())
            for name, assumed in (("E2_a_to_s_only", 0.0), ("E3_full", xi)):
                th, ka = posterior_theta_kappa(r, c, a, assumed)
                for k, nu in NUS.items():
                    est[name][k].append(th + ka * (nu @ BETA))
        wres = {"spearman_true_nu1_vs_nu2": float(spearmanr(truth["nu1_uniform"], truth["nu2_provoking"]).statistic)}
        for e, byk in est.items():
            for k, v in byk.items():
                v = np.array(v)
                wres[f"{e}|{k}"] = {"rmse": float(np.sqrt(np.mean((v - truth[k]) ** 2))),
                                    "bias": float(np.mean(v - truth[k])),
                                    "corr": float(np.corrcoef(v, truth[k])[0, 1])}
        res[f"xi={xi}"] = wres
    out = {"protocol_sha256": PROTOCOL_SHA, "code_sha256": digest(Path(__file__)), "persons": N, "turns": T,
           "worlds": res, "neural_calls": 0, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (OUT / "combined_world.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
