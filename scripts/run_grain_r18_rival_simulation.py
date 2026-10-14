#!/usr/bin/env python3
"""Packet 3 Part A (R18): rival mechanisms with person parameters + convention rival."""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_R18_R21_PACKET3_PROTOCOL_V1.md"
PROTOCOL_SHA = "5b72f2dd3f678b9e7690d8107147ba8479181479fe068e67cd29f4bd7ada6128"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F09/rival_person_simulation_r1"
SEED, N, REPS, BOOT = 20260926, 400, 3, 1000
MEANS = np.array([1.0, 0.5, 0.5, 0.5])
SDS = np.array([1.0, 0.8, 0.8, 0.8])
MECH = ("M1_uncertainty", "M2_evaluation", "M3_relationship", "M4_strategic")


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def persons(rng, world):
    theta = MEANS + SDS * rng.standard_normal((N, 4)) if world == "W1" else np.tile(MEANS, (N, 1))
    alpha = rng.normal(-1.0, 0.5, N)
    prof = rng.standard_normal(N)
    gamma = 0.3 + 0.8 * prof
    z = (theta - theta.mean(0)) / np.where(theta.std(0) > 0, theta.std(0), 1)
    noise = rng.standard_normal(N)
    trait = (0.5 * z[:, 1] + 0.3 * z[:, 0] + np.sqrt(1 - 0.25 - 0.09) * noise
             if world == "W1" else noise)
    return theta, alpha, gamma, trait


def trials(rng, design):
    cells = np.array(list(itertools.product([0, 1], repeat=5)), float)  # B e r k g
    x = np.repeat(cells, REPS, axis=0)
    if design == "D4":
        tie = rng.random(len(x)) < 0.9
        x[tie, 4] = x[tie, 1]
    return x


def simulate_session(rng, theta, alpha, gamma, design):
    rows = []
    for u in range(N):
        x = trials(rng, design)
        B, e, r, k, g = x.T
        eta = (alpha[u] + theta[u, 0] * (1 - B) + theta[u, 1] * e + theta[u, 2] * r
               + theta[u, 3] * B * k + gamma[u] * g)
        y = (rng.random(len(x)) < 1 / (1 + np.exp(-eta))).astype(float)
        C = np.where(rng.random(len(x)) < 0.9, B, 1 - B)
        rows.append((x, y, C))
    return rows


def design_matrix(x, C, design):
    B, e, r, k, g = x.T
    one = np.ones(len(x))
    if design == "D1":
        cols = {"M1_uncertainty": 1 - B, "M2_evaluation": e, "M3_relationship": r, "M4_strategic": B * k, "M5_convention": g}
    elif design == "D2":
        cols = {"M2_evaluation": e, "M3_relationship": r, "M4_strategic": k, "M5_convention": g}
    elif design == "D3":
        cols = {"M1_uncertainty": 1 - C, "M2_evaluation": e, "M3_relationship": r, "M4_strategic": C * k, "M5_convention": g}
    else:
        cols = {"M1_uncertainty": 1 - B, "M2_evaluation": e, "M3_relationship": r, "M4_strategic": B * k}
    names = list(cols)
    return np.column_stack([one, *cols.values()]), names


def estimate(session, design):
    est, var = [], []
    for x, y, C in session:
        X, names = design_matrix(x, C, design)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        s2 = resid @ resid / (len(y) - X.shape[1])
        cov = s2 * np.linalg.pinv(X.T @ X)
        est.append(beta[1:])
        var.append(np.diag(cov)[1:])
    return np.array(est), np.array(var), names


def corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def partial(a, b, c):
    def res(v):
        X = np.column_stack([np.ones(len(c)), c])
        return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    return corr(res(a), res(b))


def boot_ci(fn, rng):
    vals = [fn(rng.integers(0, N, N)) for _ in range(BOOT)]
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for wi, world in enumerate(("W1", "W2")):
        for di, design in enumerate(("D1", "D2", "D3", "D4")):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, wi, di]))
            theta, alpha, gamma, trait = persons(rng, world)
            s1 = simulate_session(rng, theta, alpha, gamma, design)
            s2 = simulate_session(rng, theta, alpha, gamma, design)
            e1, v1, names = estimate(s1, design)
            e2, _, _ = estimate(s2, design)
            brng = np.random.default_rng(np.random.SeedSequence([SEED, wi, di, 99]))
            for j, name in enumerate(names):
                truth = gamma if name == "M5_convention" else theta[:, MECH.index(name)]
                true_sd = float(np.std(truth))
                a, b = e1[:, j], e2[:, j]
                row = {
                    "world": world, "design": design, "coordinate": name,
                    "true_person_sd": true_sd,
                    "mean_estimate": float(a.mean()),
                    "true_mean": float(truth.mean()),
                    "recovery_r": corr(a, truth) if true_sd > 0 else float("nan"),
                    "retest_r": corr(a, b),
                    "retest_ci": boot_ci(lambda i, a=a, b=b: corr(a[i], b[i]), brng),
                    "est_person_sd": float(np.sqrt(max(0.0, a.var(ddof=1) - v1[:, j].mean()))),
                    "corr_with_proficiency_gamma": corr(a, gamma),
                    "r2_on_trait": corr(a, trait) ** 2,
                    "partial_retest_given_trait": partial(a, b, trait),
                }
                row["identified"] = bool(world == "W1" and row["recovery_r"] >= 0.5 and row["retest_r"] >= 0.5)
                row["non_redundant"] = bool(row["r2_on_trait"] <= 0.5 and row["partial_retest_given_trait"] >= 0.3)
                row["false_mechanism"] = bool(true_sd == 0 and row["retest_r"] >= 0.30)
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rival_person_metrics.csv", index=False)
    manifest = {"protocol_sha256": PROTOCOL_SHA, "code_sha256": digest(Path(__file__)),
                "metrics_sha256": digest(OUT / "rival_person_metrics.csv"),
                "neural_calls": 0, "persons": N, "trials_per_session": 32 * REPS,
                "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    cols = ["world", "design", "coordinate", "true_person_sd", "recovery_r", "retest_r",
            "est_person_sd", "corr_with_proficiency_gamma", "r2_on_trait",
            "partial_retest_given_trait", "identified", "non_redundant", "false_mechanism"]
    print(df[cols].round(3).to_string())


if __name__ == "__main__":
    main()
