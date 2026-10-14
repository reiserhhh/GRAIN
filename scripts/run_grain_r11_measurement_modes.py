#!/usr/bin/env python3
"""R11 measurement modes: naturalistic, calibration probes, triggered adaptive, always adaptive."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_R11_MEASUREMENT_MODES_PROTOCOL_V1.md"
PROTOCOL_SHA = "a192b0261fe5131440d5f67187186895f9c7e81e07cdecd64f1f75744ce2e63b"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F06/measurement_modes_r1"
N, T, SEED, BOOT = 500, 20, 20260926, 2000
GRID = np.linspace(-4, 4, 81)
PRIOR = np.exp(-0.5 * GRID ** 2)
PRIOR /= PRIOR.sum()
PROBES = {"P1": (0.5, 0.05), "P2": (1.5, 0.20)}
LAM, GAM, TRIG_SD = 0.02, 0.10, 0.45


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def post_var(w):
    m = (w * GRID).sum()
    return (w * (GRID - m) ** 2).sum()


def exp_gain(w, effect):
    p1 = expit(GRID + effect)
    m1 = (w * p1).sum()
    w1, w0 = w * p1 / m1, w * (1 - p1) / (1 - m1)
    return post_var(w) - (m1 * post_var(w1) + (1 - m1) * post_var(w0))


def run(mode, theta, h, beta, rng):
    w = PRIOR.copy()
    state, probes, distortion = 0.0, 0, 0.0
    for t in range(T):
        choice = None
        if mode == "CAL" and t in (5, 15):
            choice = "P1"
        elif mode in ("TRIG", "ADAPT") and (mode == "ADAPT" or np.sqrt(post_var(w)) > TRIG_SD):
            nat_c_prob = 0.5
            nat_gain = nat_c_prob * exp_gain(w, 1.0) + (1 - nat_c_prob) * exp_gain(w, 0.0)
            options = {"none": nat_gain}
            for name, (eff, react) in PROBES.items():
                options[name] = exp_gain(w, eff) - LAM - GAM * react
            best = max(options, key=options.get)
            choice = None if best == "none" else best
        if choice is None:
            high = rng.random() < expit(2 * h)
            true_eff, assumed = (beta if high else 0.0), (1.0 if high else 0.0)
        else:
            eff, react = PROBES[choice]
            true_eff, assumed = eff, eff
            probes += 1
            distortion += react
        y = rng.random() < expit(theta + true_eff + state)
        if choice is not None:
            state += PROBES[choice][1]
        lik = expit(GRID + assumed)
        w = w * (lik if y else 1 - lik)
        w /= w.sum()
    return float((w * GRID).sum()), probes, distortion


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
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
        u = -(err ** 2) - LAM * pr - GAM * dist
        util[mode] = u
        res[mode] = {"rmse": float(np.sqrt(np.mean(err ** 2))), "corr": float(np.corrcoef(est, theta)[0, 1]),
                     "error_corr_with_hidden_cue": float(np.corrcoef(err, h)[0, 1]),
                     "error_corr_with_env_beta": float(np.corrcoef(err, beta)[0, 1]),
                     "probes_per_person": float(pr.mean()), "distortion": float(dist.mean()),
                     "realized_utility": float(u.mean())}
    brng = np.random.default_rng(SEED + 5)
    idx = brng.integers(0, N, (BOOT, N))
    paired = {m: {"point": float((util[m] - util["NAT"]).mean()),
                  "ci": [float(x) for x in np.percentile((util[m] - util["NAT"])[idx].mean(1), [2.5, 97.5])]}
              for m in ("CAL", "TRIG", "ADAPT")}
    out = {"protocol_sha256": PROTOCOL_SHA, "code_sha256": digest(Path(__file__)), "persons": N, "turns": T,
           "modes": res, "utility_minus_NAT": paired, "neural_calls": 0,
           "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (OUT / "measurement_modes.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"modes": res, "utility_minus_NAT": paired}, indent=1))


if __name__ == "__main__":
    main()
