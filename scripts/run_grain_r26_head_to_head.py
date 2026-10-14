#!/usr/bin/env python3
"""R26/A07 adaptive head-to-head: fixed, ALBA-style scalar CAT, operator-aware, operator+cost."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_R26_A07_HEAD_TO_HEAD_PROTOCOL_V1.md"
PROTOCOL_SHA = "b34de1fdf789cb8704baffdb42868f66b154f31d65f5f341137da02d9eb41979"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F06/head_to_head_r1"
N, K, SEED, BOOT, CSTAR = 400, 8, 20260926, 2000, 2.0
CS = np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
ITEMS = np.array([(c, a, art, n) for c in CS for (a, art) in ((1.0, 0), (2.0, 1)) for n in (0.2, 1.0)])
TAU = np.linspace(-3, 3, 41)
KAP = np.linspace(-1, 3, 21)
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(7)
GH_W = GH_W / GH_W.sum()


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prior_op():
    pt = np.exp(-0.5 * TAU ** 2)
    pk = np.exp(-0.5 * ((KAP - 1) / 0.5) ** 2)
    w = pt[:, None, None] * pk[None, :, None] * GH_W[None, None, :]
    return w / w.sum()


def lik_op(item, y):
    c, a, _, n = item
    p = expit(a * (TAU[:, None, None] + KAP[None, :, None] * c) + n * GH_X[None, None, :])
    return p if y == 1 else 1 - p


def target_vals(target):
    t = TAU[:, None, None] + (KAP[None, :, None] * CSTAR if target == "T2" else 0 * KAP[None, :, None])
    return np.broadcast_to(t, (len(TAU), len(KAP), len(GH_X)))


def post_var(w, tv):
    m = (w * tv).sum()
    return (w * (tv - m) ** 2).sum()


def op_choose(w, target, cost):
    tv = target_vals(target)
    v0 = post_var(w, tv)
    best, arg = -np.inf, 0
    for j, item in enumerate(ITEMS):
        l1 = lik_op(item, 1)
        p1 = (w * l1).sum()
        w1 = w * l1 / p1
        w0 = w * (1 - l1) / (1 - p1)
        gain = v0 - (p1 * post_var(w1, tv) + (1 - p1) * post_var(w0, tv))
        u = gain / (1.0 if target == "T1" else 1.0 + 0.25 * CSTAR ** 2)
        if cost:
            u -= 0.05 * item[2] + 0.05 * item[3]
        if u > best:
            best, arg = u, j
    return arg


def scalar_prior():
    w = np.exp(-0.5 * TAU ** 2)
    return w / w.sum()


def scalar_lik(item, y):
    c, a, _, _ = item
    p = expit(a * (TAU + c))
    return p if y == 1 else 1 - p


def scalar_choose(w):
    m = (w * TAU).sum()
    info = [a * a * expit(a * (m + c)) * (1 - expit(a * (m + c))) for c, a, _, _ in ITEMS]
    return int(np.argmax(info))


FIXED = [int(np.flatnonzero((ITEMS[:, 0] == c) & (ITEMS[:, 1] == 1) & (ITEMS[:, 3] == 0.2))[0])
         for c in (-1.5, 1.5, -0.5, 0.5, -1, 1, 0, 0)]


def run_person(rng, tau, kap, nu, policy, target):
    w = prior_op() if policy != "S" else scalar_prior()
    used = []
    for t in range(K):
        if policy == "F":
            j = FIXED[t]
        elif policy == "S":
            j = scalar_choose(w)
        else:
            j = op_choose(w, target, cost=(policy == "OC"))
        c, a, _art, n = ITEMS[j]
        y = int(rng.random() < expit(a * (tau + kap * c) + n * nu))
        w = w * (scalar_lik(ITEMS[j], y) if policy == "S" else lik_op(ITEMS[j], y))
        w = w / w.sum()
        used.append(j)
    if policy == "S":
        m = (w * TAU).sum()
        est = m if target == "T1" else m + 1.0 * CSTAR
    else:
        est = (w * target_vals(target)).sum()
    return est, used


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    tau, kap, nu = rng.standard_normal(N), 1 + 0.5 * rng.standard_normal(N), rng.standard_normal(N)
    res, errs = {}, {}
    for target in ("T1", "T2"):
        truth = tau if target == "T1" else tau + kap * CSTAR
        for policy in ("F", "S", "O", "OC"):
            prng = np.random.default_rng(np.random.SeedSequence([SEED, hash(policy) % 997, 1 if target == "T1" else 2]))
            est, arts, nus = [], [], []
            for u in range(N):
                e, used = run_person(prng, tau[u], kap[u], nu[u], policy, target)
                est.append(e)
                arts.append(ITEMS[used, 2].mean())
                nus.append(ITEMS[used, 3].mean())
            err = np.array(est) - truth
            errs[(policy, target)] = err
            res[f"{policy}|{target}"] = {"rmse": float(np.sqrt(np.mean(err ** 2))), "bias": float(err.mean()),
                                         "artificial_share": float(np.mean(arts)), "mean_nuisance_loading": float(np.mean(nus))}
    brng = np.random.default_rng(SEED + 7)
    idx = brng.integers(0, N, (BOOT, N))
    comps = {}
    for target in ("T1", "T2"):
        for pol in ("O", "OC", "F"):
            a, b = errs[(pol, target)] ** 2, errs[("S", target)] ** 2
            diff = np.sqrt(a[idx].mean(1)) - np.sqrt(b[idx].mean(1))
            comps[f"RMSE({pol})-RMSE(S)|{target}"] = {
                "point": float(np.sqrt(a.mean()) - np.sqrt(b.mean())),
                "ci": [float(x) for x in np.percentile(diff, [2.5, 97.5])]}
    out = {"protocol_sha256": PROTOCOL_SHA, "code_sha256": digest(Path(__file__)), "persons": N, "budget": K,
           "policies": res, "paired": comps,
           "note": "policy seed uses Python hash of a one-letter string; PYTHONHASHSEED fixed to 0 in the run command",
           "neural_calls": 0, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (OUT / "head_to_head.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"policies": res, "paired": comps}, indent=1))


if __name__ == "__main__":
    main()
