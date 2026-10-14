#!/usr/bin/env python3
"""H01: person sample size from parameter recovery with state dynamics and person x context slopes."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm, pearsonr

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_H01_PERSON_N_RECOVERY_PROTOCOL_V1.md"
PROTOCOL_SHA = "b70ed83ee8bb738218c646a46c4c4a178a0c4126723dc0c20e5ae574ae1cb203"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F17/h01_person_n_recovery_r1"
Z = np.array([-1.5, -0.5, 0.5, 1.5])
D, RHO, SI, SE, SIMS, SEED = 30, 0.6, 0.5, 0.7, 200, 20260926


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def one(rng, n):
    a = rng.standard_normal(n)
    b = rng.normal(0.5, 0.5, n)
    s = np.zeros((n, D))
    s[:, 0] = rng.normal(0, SI / np.sqrt(1 - RHO ** 2), n)
    for d in range(1, D):
        s[:, d] = RHO * s[:, d - 1] + rng.normal(0, SI, n)
    y = a[:, None, None] + b[:, None, None] * Z[None, None, :] + s[:, :, None] + rng.normal(0, SE, (n, D, len(Z)))
    zc = Z - Z.mean()

    def slope(yy):
        return (yy.mean(1) * zc).sum(1) / (zc ** 2).sum()

    bh = slope(y)
    b1, b2 = slope(y[:, 0::2]), slope(y[:, 1::2])
    r12 = np.corrcoef(b1, b2)[0, 1]
    rel = 2 * r12 / (1 + r12)
    resid = y - y.mean(2, keepdims=True) - (bh[:, None, None] * zc[None, None, :])
    se2 = (resid ** 2).sum((1, 2)) / (D * len(Z) - D - 1) / (D * (zc ** 2).sum())
    var_b = bh.var(ddof=1) - se2.mean()
    boot = []
    for _ in range(200):
        i = rng.integers(0, n, n)
        boot.append(bh[i].var(ddof=1) - se2[i].mean())
    zb = (b - b.mean()) / b.std()
    crit = 0.2 * zb + np.sqrt(1 - 0.04) * rng.standard_normal(n)
    r, p = pearsonr(bh, crit)
    return {"rel": rel, "var_b": var_b, "var_lo": float(np.percentile(boot, 2.5)), "r": r, "p": p}


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []
    for n in (100, 150, 200, 300, 450, 600):
        res = [one(rng, n) for _ in range(SIMS)]
        rows.append({"persons": n, "slope_split_half_reliability": float(np.mean([x["rel"] for x in res])),
                     "person_slope_variance_mean": float(np.mean([x["var_b"] for x in res])),
                     "variance_detected_rate": float(np.mean([x["var_lo"] > 0 for x in res])),
                     "criterion_r_mean": float(np.mean([x["r"] for x in res])),
                     "criterion_power": float(np.mean([x["p"] < 0.05 and x["r"] > 0 for x in res]))})
    rec = next((r["persons"] for r in rows if r["criterion_power"] >= 0.80 and r["variance_detected_rate"] >= 0.95), None)
    rel = rows[-1]["slope_split_half_reliability"]
    r_obs = 0.2 * np.sqrt(rel * 0.8)
    n_analytic = float(((norm.ppf(0.975) + norm.ppf(0.8)) / np.arctanh(r_obs)) ** 2 + 3)
    out = {"protocol_sha256": PROTOCOL_SHA, "code_sha256": digest(Path(__file__)), "grid": rows,
           "recommended_persons": rec, "analytic_n_if_criterion_reliability_0.8": round(n_analytic, 1),
           "assumptions": "criterion effect r .2 on the true slope; criterion reliability 1 in simulation; 30 days x 4 contexts",
           "neural_calls": 0, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (OUT / "person_n_recovery.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
