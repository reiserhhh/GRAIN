#!/usr/bin/env python3
"""R11 amendment A1: joint (theta, beta_u) estimator and planner for the four modes."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.special import expit

from scripts.run_grain_r11_measurement_modes import (
    BOOT,
    GAM,
    LAM,
    OUT,
    PROBES,
    PROTOCOL,
    SEED,
    TRIG_SD,
    N,
    T,
    digest,
)

PROTOCOL_SHA_A1 = "a15e1034969a5ae36b082b09c0b5cbbf93ca61af88cf7fdd1d3a95ca7d9d0307"
TH = np.linspace(-4, 4, 81)[:, None]
BE = np.linspace(-1, 3, 21)[None, :]
PRIOR = np.exp(-0.5 * TH ** 2) * np.exp(-0.5 * ((BE - 1) / 0.5) ** 2)
PRIOR /= PRIOR.sum()


def var_theta(w):
    m = (w * TH).sum()
    return (w * (TH - m) ** 2).sum()


def lik(kind, eff=0.0):
    if kind == "nat_high":
        return expit(TH + BE)
    if kind == "nat_low":
        return np.broadcast_to(expit(TH + 0 * BE), PRIOR.shape)
    return np.broadcast_to(expit(TH + eff + 0 * BE), PRIOR.shape)


def gain(w, p1):
    m1 = (w * p1).sum()
    return var_theta(w) - (m1 * var_theta(w * p1 / m1) + (1 - m1) * var_theta(w * (1 - p1) / (1 - m1)))


def run(mode, theta, h, beta, rng):
    w = PRIOR.copy()
    state, probes, distortion = 0.0, 0, 0.0
    for t in range(T):
        choice = None
        if mode == "CAL" and t in (5, 15):
            choice = "P1"
        elif mode in ("TRIG", "ADAPT") and (mode == "ADAPT" or np.sqrt(var_theta(w)) > TRIG_SD):
            options = {"none": 0.5 * gain(w, lik("nat_high")) + 0.5 * gain(w, lik("nat_low"))}
            for name, (eff, react) in PROBES.items():
                options[name] = gain(w, lik("probe", eff)) - LAM - GAM * react
            best = max(options, key=options.get)
            choice = None if best == "none" else best
        if choice is None:
            high = rng.random() < expit(2 * h)
            true_eff, p = (beta, lik("nat_high")) if high else (0.0, lik("nat_low"))
        else:
            eff, react = PROBES[choice]
            true_eff, p = eff, lik("probe", eff)
            probes += 1
            distortion += react
        y = rng.random() < expit(theta + true_eff + state)
        if choice is not None:
            state += PROBES[choice][1]
        w = w * (p if y else 1 - p)
        w = w / w.sum()
    return float((w * TH).sum()), probes, distortion


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA_A1:
        raise ValueError("Protocol changed after A1")
    rng = np.random.default_rng(SEED)
    theta = rng.standard_normal(N)
    h = 0.7 * theta + 0.71 * rng.standard_normal(N)
    beta = rng.normal(1, 0.5, N)
    res, util = {}, {}
    for i, mode in enumerate(("NAT", "CAL", "TRIG", "ADAPT")):
        mrng = np.random.default_rng(np.random.SeedSequence([SEED, i]))
        est, pr, dist = zip(*(run(mode, theta[u], h[u], beta[u], mrng) for u in range(N)), strict=True)
        est, pr, dist = np.array(est), np.array(pr), np.array(dist)
        err = est - theta
        util[mode] = -(err ** 2) - LAM * pr - GAM * dist
        res[f"J-{mode}"] = {"rmse": float(np.sqrt(np.mean(err ** 2))), "corr": float(np.corrcoef(est, theta)[0, 1]),
                            "error_corr_with_hidden_cue": float(np.corrcoef(err, h)[0, 1]),
                            "error_corr_with_env_beta": float(np.corrcoef(err, beta)[0, 1]),
                            "probes_per_person": float(pr.mean()), "distortion": float(dist.mean()),
                            "realized_utility": float(util[mode].mean())}
    idx = np.random.default_rng(SEED + 5).integers(0, N, (BOOT, N))
    paired = {f"J-{m}": {"point": float((util[m] - util["NAT"]).mean()),
                         "ci": [float(x) for x in np.percentile((util[m] - util["NAT"])[idx].mean(1), [2.5, 97.5])]}
              for m in ("CAL", "TRIG", "ADAPT")}
    out = {"protocol_sha256": PROTOCOL_SHA_A1, "code_sha256": digest(Path(__file__)), "modes": res,
           "utility_minus_J_NAT": paired, "neural_calls": 0, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (OUT / "measurement_modes_joint.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"modes": res, "utility_minus_J_NAT": paired}, indent=1))


if __name__ == "__main__":
    main()
