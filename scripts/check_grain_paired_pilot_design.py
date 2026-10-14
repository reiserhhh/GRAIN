#!/usr/bin/env python3
"""Analytic check for the C2 paired-pilot specification (small synthetic arrays, no text)."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/grain_construction_v1/C2/evidence_unit_r1"
rng = np.random.default_rng(20261024)
D, W = 16, 8
v = rng.normal(size=D)
v /= np.linalg.norm(v)
q = rng.normal(size=D)
q -= v * (v @ q)
s = 0.6 * v + 0.8 * q / np.linalg.norm(q)          # overlapping style loading, cos .6
res = {}
for rho in (0.0, 0.6):
    a = rng.normal(size=W)                          # writer target scale (unused by contrasts)
    r = rho * a + np.sqrt(1 - rho ** 2) * rng.normal(size=W)   # writer register tendency, correlated
    x = {}
    for w in range(W):
        for t in (0, 1):                            # declared target level, fixed register
            for g in (0, 1):                        # register, fixed target
                x[(w, t, g)] = v * t + s * g + (a[w] * v + r[w] * s)   # writer effects cancel in contrasts
    v_hat = np.mean([x[(w, 1, g)] - x[(w, 0, g)] for w in range(W) for g in (0, 1)], axis=0)
    s_hat = np.mean([x[(w, t, 1)] - x[(w, t, 0)] for w in range(W) for t in (0, 1)], axis=0)
    gmat = np.column_stack([v_hat, s_hat])
    w_dual = gmat @ np.linalg.solve(gmat.T @ gmat, np.array([1.0, 0.0]))
    cov = np.cov(np.column_stack([a, r]).T) if W > 2 else None
    c_src = np.column_stack([v, s]) @ cov @ np.column_stack([v, s]).T
    lam, U = np.linalg.eigh(c_src)
    keep = lam > 1e-10
    cp = U[:, keep] @ np.diag(1 / lam[keep]) @ U[:, keep].T
    w_lcmv = cp @ v_hat / (v_hat @ cp @ v_hat)
    design = np.array([[1, t, g] for t in (0, 1) for g in (0, 1)], float)
    res[f"rho_{rho}"] = {"design_rank_intercept_target_register": int(np.linalg.matrix_rank(design)),
                         "v_hat_error": float(np.linalg.norm(v_hat - v)), "s_hat_error": float(np.linalg.norm(s_hat - s)),
                         "paired_dual_wTv": float(w_dual @ v), "paired_dual_wTs": float(w_dual @ s),
                         "source_lcmv_wTs (uses writer covariance)": float(w_lcmv @ s),
                         "sample_corr_a_r": float(np.corrcoef(a, r)[0, 1])}
res["note"] = ("Within-writer target contrast at fixed register estimates v; within-writer register contrast at fixed "
               "target estimates s; writer effects cancel, so correlated writer target/register tendencies do not bias "
               "either contrast. The paired dual satisfies w'v=1, w's=0 for any rho; source minimum-variance weights "
               "inherit the writer correlation. Noiseless arrays; not a power or sample-size result.")
(OUT / "paired_pilot_check.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
