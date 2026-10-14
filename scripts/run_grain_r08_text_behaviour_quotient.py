#!/usr/bin/env python
"""R08 text/behaviour quotient (docs/GRAIN_R08_TEXT_BEHAVIOUR_QUOTIENT_PROTOCOL_V1.md). Exact finite computation."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import csv
import hashlib
import itertools
import json
import math
import sys
import time
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_R08_TEXT_BEHAVIOUR_QUOTIENT_PROTOCOL_V1.md"
PROTOCOL_SHA = "64a9ed3d1bafee8d1c0573e822f748d3689810c0755350081118275e93c1af9a"
AMEND = ROOT / "docs/GRAIN_R08_TEXT_BEHAVIOUR_QUOTIENT_PROTOCOL_V1_A1.md"
AMEND_SHA = "9336b844716c55d1437875f5751cdb020bc9a6462095863c726de37af4715075"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F15/text_behaviour_quotient_r1"
C_R, C_T, MTOL = 0.1, 0.9, 1e-9
F = Fr


def q(x):
    return F(str(x))


# ---------------------------------------------------------------- worlds (exact Fractions)
def bern(p, v):
    return p if v == 1 else 1 - p


def register_step(r, r2, nreg):
    if nreg == 1:
        return F(1)
    return q(.9) if r2 == r else q(.1) / (nreg - 1)


def form_prob(f, r, nreg, stay):
    return stay if f == r else (1 - stay) / (nreg - 1)


def world_W(name):
    """Return dict(states, obs, K[a][s][o][s'] as Fractions, labels)."""
    nreg = 3 if name == "W5" else 2
    stay_f = q(.8) if name == "W5" else q(.9)

    def ph(x, h, r, a):
        base = q(.3) + q(.4) * h + q(.1) * a
        if name != "W6":
            base += q(.15) * x
        if name == "W3":
            base -= q(.2) * r          # amendment A1 (V1 had +.2r, which exceeds 1 at x=1,h=1,r=1)
        return base

    def pc(x, h, r, a):
        return q(.2) + q(.5) * h + q(.2) * a * x

    def pb(x, h, r, a):
        v = q(.2) + q(.3) * x + q(.3) * h
        if name == "W2":
            v -= q(.2) * r
        if name == "W5":
            v -= q(.02) * r
        return v
    states = [(x, h, r) for x in (0, 1) for h in (0, 1) for r in range(nreg)]
    obs = [(c, f, b) for c in (0, 1) for f in range(nreg) for b in (0, 1)]
    K = []
    for a in (0, 1):
        Ka = []
        for (x, h, r) in states:
            row = []
            for (c, f, b) in obs:
                e = bern(pc(x, h, r, a), c) * form_prob(f, r, nreg, stay_f) * bern(pb(x, h, r, a), b)
                row.append([e * (F(1) if x2 == x else F(0)) * bern(ph(x, h, r, a), h2) * register_step(r, r2, nreg) for (x2, h2, r2) in states])
            Ka.append(row)
        K.append(Ka)
    labels = [f"x{x}h{h}r{r}" for (x, h, r) in states]
    views = {"raw": lambda o: o, "Y": lambda o: (o[0], o[1]), "B": lambda o: (o[2],), "construct": lambda o: (o[0], o[2]), "content": lambda o: (o[0],)}
    return {"name": name, "states": states, "labels": labels, "obs": obs, "K": K, "views": views, "nreg": nreg}


def world_W4():
    """Augmented space {K1, K2} x (x, h); observation (c', B) with c' = K(c)."""
    def ph(x, h, a):
        return q(.5) + q(.3) * (2 * h - 1) + q(.1) * (2 * x - 1)

    def pc(x, h, a):
        return (q(.8) if h == 1 else q(.2)) + q(.1) * a * (2 * x - 1)

    def pb(x, h, a):
        return q(.3) + q(.4) * x
    states = [(k, x, h) for k in (1, 2) for x in (0, 1) for h in (0, 1)]
    obs = [(c, b) for c in (0, 1) for b in (0, 1)]
    K = []
    for a in (0, 1):
        Ka = []
        for (k, x, h) in states:
            row = []
            for (c2, b) in obs:
                c = c2 if k == 1 else 1 - c2
                e = bern(pc(x, h, a), c) * bern(pb(x, h, a), b)
                row.append([e * (F(1) if (k2, x2) == (k, x) else F(0)) * bern(ph(x, h, a), h2) for (k2, x2, h2) in states])
            Ka.append(row)
        K.append(Ka)
    labels = [f"K{k}x{x}h{h}" for (k, x, h) in states]
    views = {"cprime": lambda o: (o[0],), "joint": lambda o: o, "B": lambda o: (o[1],)}
    return {"name": "W4", "states": states, "labels": labels, "obs": obs, "K": K, "views": views}


def c2_xor_world():
    """The archived A01 c2_xor world: window (a, b), emission y ~ Bern(p), transition to (b, y)."""
    states = [(a, b) for a in (0, 1) for b in (0, 1)]
    obs = [(0,), (1,)]
    K0 = []
    for (a, b) in states:
        p = q(.75) if (a ^ b) == 1 else q(.25)
        K0.append([[bern(p, y) if s2 == (b, y) else F(0) for s2 in states] for (y,) in obs])
    return {"name": "c2_xor", "states": states, "labels": ["00", "01", "10", "11"], "obs": obs, "K": [K0], "views": {"raw": lambda o: o}}


# ---------------------------------------------------------------- operators
def view_ops(w, view, actions, exact):
    """M[(a,u)] = S x S matrix: sum over o with phi(o) = u of K_a(o, s' | s)."""
    phi = w["views"][view]
    S = len(w["states"])
    ops = {}
    for a in actions:
        for oi, o in enumerate(w["obs"]):
            u = phi(o)
            key = (a, u)
            if key not in ops:
                ops[key] = [[F(0)] * S for _ in range(S)] if exact else np.zeros((S, S))
            for s in range(S):
                for s2 in range(S):
                    v = w["K"][a][s][oi][s2]
                    if exact:
                        ops[key][s][s2] += v
                    else:
                        ops[key][s, s2] += float(v)
    return ops


def krylov_classes_float(w, view, actions):
    ops = view_ops(w, view, actions, exact=False)
    S = len(w["states"])
    basis, gen = [], [np.ones(S)]
    Q = np.zeros((S, 0))
    while gen:
        v = gen.pop()
        r = v - Q @ (Q.T @ v)
        r = r - Q @ (Q.T @ r)
        if np.linalg.norm(r) > 1e-10 * max(1.0, np.linalg.norm(v)):
            Q = np.column_stack([Q, r / np.linalg.norm(r)])
            basis.append(v)
            gen.extend(M @ v for M in ops.values())
    Fm = np.column_stack(basis)
    return classes_from_rows([tuple(np.round(row, 10)) for row in Fm]), Fm.shape[1]


def krylov_classes_exact(w, view, actions):
    ops = view_ops(w, view, actions, exact=True)
    S = len(w["states"])
    ech, basis, gen = [], [], [[F(1)] * S]
    while gen:
        v = gen.pop()
        r = list(v)
        for piv, row in ech:
            if r[piv] != 0:
                f = r[piv] / row[piv]
                r = [x - f * y for x, y in zip(r, row)]
        nz = next((i for i, x in enumerate(r) if x != 0), None)
        if nz is not None:
            ech.append((nz, r))
            basis.append(v)
            for M in ops.values():
                gen.append([sum(M[s][s2] * v[s2] for s2 in range(S)) for s in range(S)])
    rows = [tuple(b[s] for b in basis) for s in range(S)]
    return classes_from_rows(rows), len(basis)


def classes_from_rows(rows):
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(r, []).append(i)
    return sorted(groups.values())


def bisimulation(w, view, actions):
    ops = view_ops(w, view, actions, exact=True)
    S = len(w["states"])
    block = [0] * S
    while True:
        nb = max(block) + 1
        sig = []
        for s in range(S):
            parts = []
            for key in sorted(ops, key=str):
                M = ops[key]
                parts.append(tuple(sum(M[s][s2] for s2 in range(S) if block[s2] == b) for b in range(nb)))
            sig.append((block[s], tuple(parts)))
        ids = {}
        new = [ids.setdefault(x, len(ids)) for x in sig]
        if len(ids) == nb:
            return classes_from_rows([(b,) for b in new])
        block = new


def lumpability(w, view, actions, partition):
    ops = view_ops(w, view, actions, exact=True)
    S = len(w["states"])
    bid = {s: i for i, blk in enumerate(partition) for s in blk}
    worst = (F(0), None)
    for key, M in ops.items():
        for blk in partition:
            for b2 in range(len(partition)):
                vals = [sum(M[s][s2] for s2 in range(S) if bid[s2] == b2) for s in blk]
                spread = max(vals) - min(vals)
                if spread > worst[0]:
                    worst = (spread, {"action": key[0], "view_obs": list(key[1]), "block": [w["labels"][s] for s in blk], "to_block": [w["labels"][s] for s in partition[b2]]})
    return {"lumpable": worst[0] == 0, "max_violation": float(worst[0]), "where": worst[1]}


def future_law(w, view, s, actions_seq):
    phi = w["views"][view]
    S = len(w["states"])
    dist = {((), s): F(1)}
    for a in actions_seq:
        new = {}
        for (seq, st), p in dist.items():
            for oi, o in enumerate(w["obs"]):
                for s2 in range(S):
                    v = w["K"][a][st][oi][s2]
                    if v:
                        key = (seq + (phi(o),), s2)
                        new[key] = new.get(key, F(0)) + p * v
        dist = new
    marg = {}
    for (seq, _st), p in dist.items():
        marg[seq] = marg.get(seq, F(0)) + p
    return marg


def tv(p, qd):
    keys = set(p) | set(qd)
    return sum(abs(p.get(k, F(0)) - qd.get(k, F(0))) for k in keys) / 2


def transition_matrix(w, a):
    S = len(w["states"])
    T = np.zeros((S, S))
    for s in range(S):
        for oi in range(len(w["obs"])):
            for s2 in range(S):
                T[s, s2] += float(w["K"][a][s][oi][s2])
    return T


def emission(w, view, a):
    phi = w["views"][view]
    us = sorted({phi(o) for o in w["obs"]})
    S = len(w["states"])
    E = np.zeros((S, len(us)))
    for s in range(S):
        for oi, o in enumerate(w["obs"]):
            E[s, us.index(phi(o))] += float(sum(w["K"][a][s][oi]))
    return E, us


def w1(p, qd, d):
    n = len(p)
    if np.array_equal(p, qd):
        return 0.0
    A = []
    b = []
    for i in range(n):
        row = np.zeros((n, n)); row[i, :] = 1; A.append(row.ravel()); b.append(p[i])
    for j in range(n):
        col = np.zeros((n, n)); col[:, j] = 1; A.append(col.ravel()); b.append(qd[j])
    res = linprog(d.ravel(), A_eq=np.array(A), b_eq=np.array(b), bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError("W1 LP failed")
    return float(res.fun)


def bisim_metric(w, view, actions, tol=MTOL):
    S = len(w["states"])
    Es = {a: emission(w, view, a)[0] for a in actions}
    Ts = {a: transition_matrix(w, a) for a in actions}
    d = np.zeros((S, S))
    for it in range(10_000):
        new = np.zeros_like(d)
        for s in range(S):
            for t in range(s + 1, S):
                new[s, t] = new[t, s] = max(C_R * 0.5 * np.abs(Es[a][s] - Es[a][t]).sum() + C_T * w1(Ts[a][s], Ts[a][t], d) for a in actions)
        change = np.abs(new - d).max()
        d = new
        if change < tol:
            return d, it + 1
    raise RuntimeError("no convergence")


# ---------------------------------------------------------------- analyses
VIEWS_LONG = [("Y_passive", "Y", (0,)), ("Y_probed", "Y", (0, 1)), ("B", "B", (0, 1)), ("raw", "raw", (0, 1)), ("construct", "construct", (0, 1))]


def labels(w, cls):
    return [[w["labels"][i] for i in c] for c in cls]


def partitions(w):
    out = {}
    S = len(w["states"])
    Ec, _ = emission(w, "content", 0)
    mean_c = [round(float(Ec[s][1]), 12) for s in range(S)]
    out["mean_content_a0"] = {"classes": labels(w, classes_from_rows([(m,) for m in mean_c]))}
    Ey, _ = emission(w, "Y", 0)
    out["Y_one_step_a0"] = {"classes": labels(w, classes_from_rows([tuple(np.round(Ey[s], 12)) for s in range(S)]))}
    for name, view, acts in VIEWS_LONG:
        cf, dim = krylov_classes_float(w, view, acts)
        ce, dime = krylov_classes_exact(w, view, acts)
        out[name] = {"classes": labels(w, cf), "future_span_dim": dim, "exact_agrees": cf == ce and dim == dime}
    out["bisimulation_construct"] = {"classes": labels(w, bisimulation(w, "construct", (0, 1)))}
    out["bisimulation_raw"] = {"classes": labels(w, bisimulation(w, "raw", (0, 1)))}
    for v in out.values():
        v["size"] = len(v["classes"])
    return out


def g_checks(w):
    """g: r -> 1-r on states, f -> 1-f on observations (W1-W3)."""
    st, ob = w["states"], w["obs"]
    gs = {i: st.index((x, h, 1 - r)) for i, (x, h, r) in enumerate(st)}
    go = {i: ob.index((c, 1 - f, b)) for i, (c, f, b) in enumerate(ob)}
    raw_inv, equiv = F(0), F(0)
    for a in (0, 1):
        for s in range(len(st)):
            for oi in range(len(ob)):
                for s2 in range(len(st)):
                    v = w["K"][a][s][oi][s2]
                    raw_inv = max(raw_inv, abs(w["K"][a][gs[s]][oi][gs[s2]] - v))
                    equiv = max(equiv, abs(w["K"][a][gs[s]][go[oi]][gs[s2]] - v))
    orbits = sorted({tuple(sorted((s, gs[s]))) for s in range(len(st))})
    orbit_part = [list(o) for o in orbits]
    lump = lumpability(w, "construct", (0, 1), orbit_part)
    one_step = max(tv(future_law(w, "construct", s, (a,)), future_law(w, "construct", gs[s], (a,))) for s in range(len(st)) for a in (0, 1))
    raw_one = max(tv(future_law(w, "raw", s, (0,)), future_law(w, "raw", gs[s], (0,))) for s in range(len(st)))
    res = {"raw_law_invariance_max_diff": float(raw_inv), "equivariance_max_diff": float(equiv), "is_automorphism": equiv == 0,
           "raw_one_step_TV_s_vs_gs_max": float(raw_one), "construct_one_step_TV_s_vs_gs_max": float(one_step),
           "orbit_partition": labels(w, orbit_part), "orbit_lumpability_construct": lump}
    if lump["lumpable"]:
        # quotient kernel for the construct view on orbits
        ops = view_ops(w, "construct", (0, 1), exact=True)
        bid = {s: i for i, blk in enumerate(orbit_part) for s in blk}
        qk = {}
        for (a, u), M in ops.items():
            for i, blk in enumerate(orbit_part):
                s = blk[0]
                for j in range(len(orbit_part)):
                    qk[f"a{a}|{u}|{i}->{j}"] = str(sum(M[s][s2] for s2 in range(len(st)) if bid[s2] == j))
        res["quotient_kernel"] = qk
    return res


def pair_tvs(w, pairs, views):
    rows = []
    for (a_lab, b_lab) in pairs:
        s, t = w["labels"].index(a_lab), w["labels"].index(b_lab)
        for view in views:
            for pol_name, seqf in (("passive", lambda k: (0,) * k), ("probe_first", lambda k: (1,) + (0,) * (k - 1))):
                for k in range(1, 5):
                    rows.append({"world": w["name"], "s": a_lab, "t": b_lab, "view": view, "policy": pol_name, "k": k,
                                 "TV": float(tv(future_law(w, view, s, seqf(k)), future_law(w, view, t, seqf(k))))})
    return rows


def merge_losses(w, parts):
    S = len(w["states"])
    T0 = transition_matrix(w, 0)
    pi = np.full(S, 1 / S) @ np.linalg.matrix_power(T0, 500)
    EB, usB = emission(w, "B", 0)
    Ec, usc = emission(w, "content", 0)
    pB, pc = EB[:, usB.index((1,))], Ec[:, usc.index((1,))]
    targets = {"B_t+1": T0 @ pB, "B_t+3": np.linalg.matrix_power(T0, 3) @ pB, "content_t+1": T0 @ pc}
    out = []
    for view, info in parts.items():
        idx = {l: i for i, l in enumerate(w["labels"])}
        row = {"world": w["name"], "view": view, "classes": info["size"]}
        for tname, p in targets.items():
            loss = 0.0
            for blk in info["classes"]:
                ii = [idx[l] for l in blk]
                wts = pi[ii]
                pm = float(wts @ p[ii] / wts.sum()) if wts.sum() > 0 else float(p[ii].mean())
                loss += float(sum(pi[i] * (pm - p[i]) ** 2 for i in ii))
            row[f"brier_increase_{tname}"] = loss
        out.append(row)
    return out


def run_world(name):
    t0 = time.time()
    w = world_W(name)
    res = {"world": name, "states": len(w["states"]), "labels": w["labels"]}
    parts = partitions(w)
    res["partitions"] = parts
    if name in ("W1", "W2", "W3", "W5") and w["nreg"] == 2:
        res["g_checks"] = g_checks(w)
        parts["G_orbit_quotient"] = {"classes": res["g_checks"]["orbit_partition"], "size": len(res["g_checks"]["orbit_partition"])}
    pairs = {"W1": [("x0h0r0", "x1h0r0"), ("x0h0r0", "x0h0r1")], "W2": [("x0h0r0", "x0h0r1")], "W3": [("x0h0r0", "x0h0r1")],
             "W6": [("x0h0r0", "x1h0r0")], "W5": [("x0h0r0", "x0h0r1"), ("x0h0r1", "x0h0r2"), ("x0h0r0", "x0h0r2")]}[name]
    res["pair_tv"] = pair_tvs(w, pairs, ["Y", "B", "construct"])
    res["losses"] = merge_losses(w, parts)
    d, iters = bisim_metric(w, "construct", (0, 1))
    res["metric"] = {"matrix": np.round(d, 12).tolist(), "iterations": iters,
                     "zero_classes": labels(w, classes_from_rows([tuple(np.round(row, 9) == 0) for row in d]))}
    zero = [[w["labels"].index(l) for l in c] for c in parts["bisimulation_construct"]["classes"]]
    res["metric"]["zero_iff_bisimilar"] = bool(all((d[s, t] < 1e-9) == any(s in c and t in c for c in zero) for s in range(len(d)) for t in range(len(d))))
    if name == "W5":
        i = {l: w["labels"].index(l) for l in w["labels"]}
        out = {}
        for x in (0, 1):
            for h in (0, 1):
                a, b, c = (i[f"x{x}h{h}r{r}"] for r in (0, 1, 2))
                d01, d12, d02 = d[a, b], d[b, c], d[a, c]
                out[f"x{x}h{h}"] = {"d01": d01, "d12": d12, "d02": d02, "nontransitive_threshold_exists": bool(max(d01, d12) < d02),
                                    "epsilon_interval": [max(d01, d12), d02] if max(d01, d12) < d02 else None}
        res["w5_threshold"] = out
    res["seconds"] = time.time() - t0
    return res


def run_w4():
    t0 = time.time()
    w = world_W4()
    lab = w["labels"]
    res = {"world": "W4", "states": len(lab), "labels": lab}
    parts = {}
    for view, acts in (("cprime", (0, 1)), ("joint", (0, 1)), ("B", (0, 1))):
        cf, dim = krylov_classes_float(w, view, acts)
        ce, dime = krylov_classes_exact(w, view, acts)
        parts[view] = {"classes": labels(w, cf), "size": len(cf), "future_span_dim": dim, "exact_agrees": cf == ce and dim == dime}
    res["partitions"] = parts
    pi = lambda s: (s[0], 1 - s[1], 1 - s[2])  # noqa: E731
    checks = []
    for (k, x, h) in [st for st in w["states"] if st[0] == 2]:
        s = lab.index(f"K2x{x}h{h}")
        t = lab.index(f"K1x{1 - x}h{1 - h}")
        row = {"pair": [lab[s], lab[t]]}
        for view in ("cprime", "joint"):
            row[f"TV_{view}_k1to4_max"] = float(max(tv(future_law(w, view, s, seq), future_law(w, view, t, seq))
                                                   for kk in range(1, 5) for seq in ((0,) * kk, (1,) + (0,) * (kk - 1))))
        checks.append(row)
    res["K2_vs_K1_pi_pairs"] = checks
    # population law under uniform initial over base states
    def pop(kval, view, seq):
        tot = {}
        for (k, x, h) in w["states"]:
            if k != kval:
                continue
            for key, p in future_law(w, view, lab.index(f"K{k}x{x}h{h}"), seq).items():
                tot[key] = tot.get(key, F(0)) + p / 4
        return tot
    res["population_TV_K1_vs_K2"] = {view: float(max(tv(pop(1, view, (0,) * kk), pop(2, view, (0,) * kk)) for kk in range(1, 5))) for view in ("cprime", "joint")}
    # invariant vs directional scoring from each base state
    inv = []
    for (x, h) in itertools.product((0, 1), repeat=2):
        r = {"state": f"x{x}h{h}"}
        for k in (1, 2):
            law = future_law(w, "cprime", lab.index(f"K{k}x{x}h{h}"), (0, 0))
            r[f"K{k}_P_repeat"] = float(sum(p for seq, p in law.items() if seq[0] == seq[1]))
            r[f"K{k}_E_next_cprime"] = float(sum(p for seq, p in law.items() if seq[1] == (1,)))
        inv.append(r)
    res["invariant_vs_directional_scores"] = inv
    anchor = lab.index("K1x1h1"), lab.index("K2x1h1")
    p1 = future_law(w, "cprime", anchor[0], (0,))
    p2 = future_law(w, "cprime", anchor[1], (0,))
    res["calibrated_transport_expected_LLR_per_anchor_nats"] = float(sum(float(p1[k]) * math.log(float(p1[k]) / float(p2[k])) for k in p1))
    res["seconds"] = time.time() - t0
    return res


def c2_xor_crosscheck():
    w = c2_xor_world()
    d, iters = bisim_metric(w, "raw", (0,), tol=1e-12)
    blob = json.loads((ROOT / "results/grain_gptpro_foundation_v3/F15/bisimulation_metric_r1/bisimulation_metric.json").read_text())
    arch = (blob["worlds"] if "worlds" in blob else blob)["c2_xor"]
    am = np.array(json.loads(arch["metric"]) if isinstance(arch["metric"], str) else arch["metric"])
    return {"max_d": float(d.max()), "archived_max_d": arch["max_d"], "max_abs_matrix_diff": float(np.abs(d - am).max()),
            "status": "PASS" if abs(d.max() - arch["max_d"]) < 1e-6 and np.abs(d - am).max() < 1e-6 else "FAIL"}


def kernel_rows(name):
    w = world_W4() if name == "W4" else world_W(name)
    rows = []
    for a in (0, 1):
        for s, sl in enumerate(w["labels"]):
            for oi, o in enumerate(w["obs"]):
                for s2, sl2 in enumerate(w["labels"]):
                    v = w["K"][a][s][oi][s2]
                    if v:
                        rows.append({"world": name, "action": a, "s": sl, "obs": "".join(map(str, o)), "s_next": sl2, "prob": str(v)})
    return rows


def validity_gate():
    """Every kernel entry in [0, 1] and every (s, a) row summing to exactly 1 (Fractions); fail closed otherwise."""
    for name in ("W1", "W2", "W3", "W4", "W5", "W6"):
        w = world_W4() if name == "W4" else world_W(name)
        for a in (0, 1):
            for s, row in enumerate(w["K"][a]):
                flat = [v for o in row for v in o]
                if any(v < 0 or v > 1 for v in flat) or sum(flat) != 1:
                    raise RuntimeError(f"invalid kernel in {name} at action {a}, state {w['labels'][s]}")


def main() -> int:
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA or hashlib.sha256(AMEND.read_bytes()).hexdigest() != AMEND_SHA:
        raise RuntimeError("protocol or amendment sha mismatch")
    validity_gate()
    if OUT.exists():
        raise FileExistsError("text_behaviour_quotient_r1 exists; preserve earlier runs")
    OUT.mkdir(parents=True)
    t0 = time.time()
    xc = c2_xor_crosscheck()
    (OUT / "c2_xor_metric_crosscheck.json").write_text(json.dumps(xc, indent=1) + "\n")
    if xc["status"] != "PASS":
        raise RuntimeError("metric cross-check failed; stop")
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=2)(delayed(run_world)(n) for n in ("W1", "W2", "W3", "W5", "W6"))
    w4 = run_w4()
    worlds = {r["world"]: r for r in res}
    worlds["W4"] = w4
    (OUT / "results.json").write_text(json.dumps(worlds, indent=1, default=str) + "\n")
    with (OUT / "pair_tv.csv").open("w", newline="") as h:
        rows = [r for x in res for r in x["pair_tv"]]
        wr = csv.DictWriter(h, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    with (OUT / "merge_losses.csv").open("w", newline="") as h:
        rows = [r for x in res for r in x["losses"]]
        wr = csv.DictWriter(h, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    with (OUT / "kernels.csv").open("w", newline="") as h:
        rows = [r for n in ("W1", "W2", "W3", "W4", "W5", "W6") for r in kernel_rows(n)]
        wr = csv.DictWriter(h, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    (OUT / "runtime.json").write_text(json.dumps({"seconds": time.time() - t0, "neural_calls": 0, "workers": 2,
                                                  "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=1) + "\n")
    print(json.dumps({"seconds": time.time() - t0, "crosscheck": xc["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
