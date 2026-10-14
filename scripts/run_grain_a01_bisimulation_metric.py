#!/usr/bin/env python3
"""A01/R08 approximate bisimulation metric on declared finite worlds (exact computation)."""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from grain_research.gptpro_foundation_v3.f15_theory import markov_world

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_A01_BISIMULATION_METRIC_PROTOCOL_V1.md"
PROTOCOL_SHA = "eb9f52102f69eb1f35c31ace93ba441df361db2449c9a696eb76bfb89f13f3d3"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F15/bisimulation_metric_r1"
C_R, C_T, TOL, MERGE = 0.1, 0.9, 1e-12, 0.01


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def world(name):
    """Return (order, list of per-context emission functions)."""
    if name.startswith("c2_iid_eps"):
        e = float(name.removeprefix("c2_iid_eps"))
        return 2, [lambda w, e=e: 0.5 + e * (2 * w[-1] - 1)]
    if name == "c2_xor_ctx":
        order, p = markov_world("c2_xor")
        return order, [p, lambda w: float(np.clip(p(w) + (0.10 if w[-1] == 1 else -0.10), 0.01, 0.99))]
    order, p = markov_world(name)
    return order, [p]


def mdp(name):
    order, ps = world(name)
    states = list(itertools.product((0, 1), repeat=order))
    idx = {s: i for i, s in enumerate(states)}
    n, m = len(states), len(ps)
    R = np.zeros((n, m))
    P = np.zeros((m, n, n))
    for s in states:
        for a, p in enumerate(ps):
            q = p(s)
            R[idx[s], a] = q
            P[a, idx[s], idx[(*s[1:], 1)]] += q
            P[a, idx[s], idx[(*s[1:], 0)]] += 1 - q
    return states, R, P


def w1(p, q, d):
    n = len(p)
    if np.allclose(p, q, atol=0, rtol=0):
        return 0.0
    c = d.ravel()
    a_eq, b_eq = [], []
    for i in range(n):
        row = np.zeros((n, n))
        row[i, :] = 1
        a_eq.append(row.ravel())
        b_eq.append(p[i])
    for j in range(n):
        col = np.zeros((n, n))
        col[:, j] = 1
        a_eq.append(col.ravel())
        b_eq.append(q[j])
    res = linprog(c, A_eq=np.array(a_eq), b_eq=np.array(b_eq), bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError("W1 LP failed")
    return float(res.fun)


def metric(R, P):
    n, m = R.shape
    d = np.zeros((n, n))
    for it in range(10_000):
        new = np.zeros_like(d)
        for s in range(n):
            for t in range(s + 1, n):
                new[s, t] = new[t, s] = max(
                    C_R * abs(R[s, a] - R[t, a]) + C_T * w1(P[a, s], P[a, t], d) for a in range(m))
        change = np.abs(new - d).max()
        d = new
        if change < TOL:
            return d, it + 1
    raise RuntimeError("no convergence")


def exact_classes(R, P):
    """Partition refinement: same rewards, same class-level transition mass per context."""
    n, m = R.shape
    labels = np.zeros(n, int)
    while True:
        sig = []
        for s in range(n):
            mass = tuple(tuple(round(float(P[a, s, labels == k].sum()), 12) for k in range(labels.max() + 1)) for a in range(m))
            sig.append((labels[s], tuple(round(float(x), 12) for x in R[s]), mass))
        keys = {k: i for i, k in enumerate(dict.fromkeys(sig))}
        new = np.array([keys[k] for k in sig])
        if len(set(new)) == len(set(labels)):
            return new
        labels = new


def value(R, P):
    n, m = R.shape
    v = np.zeros(n)
    for _ in range(100_000):
        q = np.stack([R[:, a] + C_T * P[a] @ v for a in range(m)], 1)
        new = q.max(1)
        if np.abs(new - v).max() < 1e-14:
            return new
        v = new
    raise RuntimeError("value iteration failed")


def aggregate(R, P, d):
    n, m = R.shape
    blocks, seen = [], set()
    for s in range(n):
        if s in seen:
            continue
        b = [t for t in range(n) if t not in seen and d[s, t] < MERGE]
        seen.update(b)
        blocks.append(b)
    k = len(blocks)
    Rb, Pb = np.zeros((k, m)), np.zeros((m, k, k))
    for i, b in enumerate(blocks):
        Rb[i] = R[b].mean(0)
        for a in range(m):
            for j, c in enumerate(blocks):
                Pb[a, i, j] = P[a][np.ix_(b, c)].sum(1).mean()
    vb = value(Rb, Pb)
    v = value(R, P)
    owner = {s: i for i, b in enumerate(blocks) for s in b}
    err = max(abs(v[s] - vb[owner[s]]) for s in range(n))
    within = max((d[s, t] for b in blocks for s in b for t in b), default=0.0)
    return len(blocks), float(err), float(within)


def main():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Protocol changed")
    OUT.mkdir(parents=True, exist_ok=True)
    names = ["c2_xor", "c2_iid", "hidden_order3", "c2_iid_eps0.01", "c2_iid_eps0.05", "c2_iid_eps0.10", "c2_xor_ctx"]
    rows = {}
    for name in names:
        states, R, P = mdp(name)
        d, iters = metric(R, P)
        cls = exact_classes(R, P)
        zero = d < 1e-10
        same = cls[:, None] == cls[None, :]
        v = value(R, P)
        dv = np.abs(v[:, None] - v[None, :])
        bound_ok = bool(np.all(dv <= d / C_R + 1e-9))
        slack = float(np.min(np.where(d > 0, d / C_R - dv, np.inf))) if np.any(d > 0) else float("inf")
        short = [d[i, j] for i, s in enumerate(states) for j, t in enumerate(states) if i < j and s[-2:] == t[-2:]]
        nb, err, within = aggregate(R, P, d)
        rows[name] = {
            "states": len(states), "contexts": R.shape[1], "iterations": iters,
            "exact_classes": len(set(cls)),
            "zero_iff_bisimilar": bool(np.array_equal(zero, same)),
            "max_d": float(d.max()),
            "value_bound_holds": bound_ok, "min_bound_slack": slack,
            "max_d_sharing_last_two": float(max(short)) if short else 0.0,
            "aggregated_blocks": nb, "aggregation_value_error": err,
            "max_within_block_d": within,
            "declared_aggregation_bound": float(2 * within / (C_R * (1 - C_T))),
            "aggregation_bound_holds": bool(err <= 2 * within / (C_R * (1 - C_T)) + 1e-12),
            "metric": [[round(float(x), 10) for x in r] for r in d],
            "state_labels": ["".join(map(str, s)) for s in states],
        }
    out = {"protocol_sha256": PROTOCOL_SHA, "code_sha256": digest(Path(__file__)),
           "c_R": C_R, "c_T": C_T, "worlds": rows, "neural_calls": 0,
           "created": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (OUT / "bisimulation_metric.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in rows.items():
        print(k, {a: b for a, b in v.items() if a not in ("metric", "state_labels")})


if __name__ == "__main__":
    main()
