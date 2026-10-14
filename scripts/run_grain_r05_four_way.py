#!/usr/bin/env python3
"""R05 four-way opportunity/selection/response/condition decomposition (exact)."""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_R05_FOUR_WAY_PROTOCOL_V1.md"
PROTOCOL_SHA = "cebb5c4a2f207a513aaf0dfb5ee69e64045bd7ee8c84e8879b88d81e5a63986c"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F03/four_way_r1"
DELTA = np.array([-0.8, 0.0, 0.8])
A = np.array([0.5, 0.0, -0.5])
SC = np.array([-1.0, 0.0, 1.0])
N, SEED = 2000, 20260926
PLAYERS = ("O", "Q", "R", "C")


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def share(a, b, b2, r, k, cond_on=True):
    pi = expit(a + A)
    q = expit(b + b2 * SC)
    w = pi * q / np.sum(pi * q)
    p = expit(r + k * (DELTA if cond_on else 0 * DELTA))
    return float(w @ p), pi, q, w, p


def value(person, ref, on):
    a = person["a"] if "O" in on else ref["a"]
    b, b2 = (person["b"], person["b2"]) if "Q" in on else (ref["b"], ref["b2"])
    r, k = (person["r"], person["k"]) if "R" in on else (ref["r"], ref["k"])
    return share(a, b, b2, r, k, cond_on="C" in on)[0]


def shapley(person, ref):
    phi = dict.fromkeys(PLAYERS, 0.0)
    orders = list(itertools.permutations(PLAYERS))
    for order in orders:
        on = set()
        for pl in order:
            before = value(person, ref, on)
            on.add(pl)
            phi[pl] += (value(person, ref, on) - before) / len(orders)
    return phi


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pop = {"a": rng.normal(0, 0.8, N), "b": rng.normal(0, 0.6, N), "b2": rng.normal(0, 0.8, N),
           "r": rng.normal(-0.5, 0.7, N), "k": rng.normal(1, 0.3, N)}
    ref = {k: float(v.mean()) for k, v in pop.items()}
    s_ref = value(ref, ref, set())
    s_full = []
    phis = {p: [] for p in PLAYERS}
    max_eff = 0.0
    for u in range(N):
        person = {k: float(v[u]) for k, v in pop.items()}
        phi = shapley(person, ref)
        s = value(person, ref, set(PLAYERS))
        max_eff = max(max_eff, abs(sum(phi.values()) - (s - s_ref)))
        s_full.append(s)
        for p in PLAYERS:
            phis[p].append(phi[p])
    s_full = np.array(s_full)
    dev = s_full - s_ref
    shares = {p: float(np.cov(np.array(phis[p]), dev)[0, 1] / np.var(s_full, ddof=1)) for p in PLAYERS}
    mean_abs = {p: float(np.mean(np.abs(phis[p]))) for p in PLAYERS}
    # D_full constructive solve for one person: logged pi, q identify a, b, b2; logged p identifies r, k
    u = 0
    person = {k: float(v[u]) for k, v in pop.items()}
    _, pi, q, _, p = share(person["a"], person["b"], person["b2"], person["r"], person["k"])
    a_hat = float(np.mean(logit(pi) - A))
    X = np.column_stack([np.ones(3), SC])
    b_hat, b2_hat = np.linalg.lstsq(X, logit(q), rcond=None)[0]
    r_hat, k_hat = np.linalg.lstsq(np.column_stack([np.ones(3), DELTA]), logit(p), rcond=None)[0]
    solve_err = max(abs(a_hat - person["a"]), abs(b_hat - person["b"]), abs(b2_hat - person["b2"]),
                    abs(r_hat - person["r"]), abs(k_hat - person["k"]))
    # D_noO counterexample: move opportunity into selection keeping pi*q per condition fixed up to scale
    alt = dict(person)
    alt["a"] = person["a"] + 1.0
    s1, pi1, q1, w1, p1 = share(person["a"], person["b"], person["b2"], person["r"], person["k"])
    # choose b for alt so that w (normalized pi*q) is identical: need q_alt = q * pi/pi_alt * const
    pi_alt = expit(alt["a"] + A)
    target_q = q1 * pi1 / pi_alt
    scale = 0.9 / target_q.max()
    target_q = target_q * scale
    # q_alt enters only through normalized w, so any positive rescale keeps observables
    w_alt = pi_alt * target_q / np.sum(pi_alt * target_q)
    obs_equal_noO = float(np.max(np.abs(w_alt - w1)))
    # Shapley for original vs alternative (alt uses free per-condition q; compare O share)
    def s_given(pi_v, q_v, pv):
        w = pi_v * q_v / np.sum(pi_v * q_v)
        return float(w @ pv)
    s_alt = s_given(pi_alt, target_q, p1)
    o_share_orig = s_given(pi1, expit(ref["b"] + ref["b2"] * SC), p1) - s_given(expit(ref["a"] + A), expit(ref["b"] + ref["b2"] * SC), p1)
    o_share_alt = s_given(pi_alt, expit(ref["b"] + ref["b2"] * SC), p1) - s_given(expit(ref["a"] + A), expit(ref["b"] + ref["b2"] * SC), p1)
    # D_noC counterexample: pooled share equal, different response/condition split
    # world 1: person r,k with delta; world 2: no condition effect, r2 chosen to match pooled share
    target = s1
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        val = float(w1 @ expit(mid + 0 * DELTA))
        lo, hi = (mid, hi) if val < target else (lo, mid)
    r2 = (lo + hi) / 2
    obs_equal_noC = abs(float(w1 @ expit(r2 + 0 * DELTA)) - target)
    out = {
        "protocol_sha256": PROTOCOL_SHA, "code_sha256": digest(Path(__file__)), "persons": N,
        "s_ref": s_ref, "var_S": float(np.var(s_full, ddof=1)),
        "shapley_efficiency_max_error": max_eff,
        "variance_attribution_shares": shares, "shares_sum": float(sum(shares.values())),
        "mean_abs_shapley": mean_abs,
        "D_full_constructive_solve_max_error": float(solve_err),
        "D_noO_counterexample": {
            "observable_weights_max_diff": obs_equal_noO, "published_share_diff": abs(s_alt - s1),
            "opportunity_contribution_original": o_share_orig, "opportunity_contribution_alternative": o_share_alt},
        "D_noC_counterexample": {
            "pooled_share_diff": obs_equal_noC, "world1_condition_effect_sd_logit": float(np.std(person["k"] * DELTA)),
            "world2_condition_effect": 0.0, "world1_r": person["r"], "world2_r": r2},
        "neural_calls": 0, "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    (OUT / "four_way.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
