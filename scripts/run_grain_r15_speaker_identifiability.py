#!/usr/bin/env python
"""R15-M1 speaker-choice identifiability (docs/GRAIN_R15_SPEAKER_IDENTIFIABILITY_PROTOCOL_V1.md). CPU, 0 neural calls."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_R15_SPEAKER_IDENTIFIABILITY_PROTOCOL_V1.md"
PROTOCOL_SHA = "962231c3728d89748b56b965090eb856f012ac8a0c9892d2b2a81b68e0a015a8"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F17/speaker_identifiability_r1"
NAMES = ["beta0", "lam", "w_face", "w_eval", "w_lc", "w_opt", "w_pres", "w_acc"]
UTT = ["assert", "hedge", "withhold"]
L0T = np.array([.95, .70, .50])
L0F = 1 - L0T
C = L0T - .5
COST = np.array([0.0, 0.10, 0.05])
STYLE = {"neutral": np.full(3, 1 / 3), "hedgy": np.array([.15, .65, .20]), "direct": np.array([.70, .15, .15])}
DELTA, ETA = 2.0, 1.0
REF = np.array([.5, 2.0, 1, 1, 1, 1, 1, 1])
BASE = {"e": 0, "a": .5, "p": .5, "s": .5, "f": .5, "m": "neutral"}
CONTEXTS = {"C0": {}, "E": {"e": 1}, "A0": {"a": 0}, "A1": {"a": 1}, "P0": {"p": 0}, "P1": {"p": 1}, "S0": {"s": 0}, "S1": {"s": 1},
            "F0": {"f": 0}, "F1": {"f": 1}, "Mh": {"m": "hedgy"}, "Md": {"m": "direct"}}
PROBES = {"E": ["E"], "A": ["A0", "A1"], "P": ["P0", "P1"], "S": ["S0", "S1"], "F": ["F0", "F1"], "M": ["Mh", "Md"]}
TOL = 1e-8


def sig(x):
    return 1 / (1 + np.exp(-x))


def u_inf(b):
    return b * np.log(L0T) + (1 - b) * np.log(L0F)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def presentation_profile():
    """m(u) = ln L1(cautious | u), with S1_phi(u|b) = softmax(3[U_inf(u;b) - kappa_phi c(u)])."""
    num, den = np.zeros(3), np.zeros(3)
    for kappa, is_c in ((1.5, 1), (0.0, 0)):
        for b in (.55, .75, .95):
            s1 = softmax(3 * (u_inf(b) - kappa * C))
            den += .5 * s1 / 3
            if is_c:
                num += .5 * s1 / 3
    return np.log(num / den)


M = presentation_profile()


def ctx(name):
    return {**BASE, **CONTEXTS[name]}


def utility(theta, c):
    b0, lam, wf, we, wl, wo, wp, wa = theta
    b = sig(b0 + DELTA * c["e"])
    U = u_inf(b) - (wf * c["a"] + we * c["s"] * (1 - b) + wl + wo * c["f"]) * C + wp * c["p"] * M + wa * np.log(STYLE[c["m"]]) - COST
    return U, b


def probs(theta, c):
    U, b = utility(theta, c)
    return softmax(theta[1] * U), sig(ETA * np.log(b / (1 - b)))


def jac(theta, c):
    """Analytic d(text probs)/d theta (3x8) and d pB/d theta (8)."""
    b0, lam, wf, we, wl, wo, wp, wa = theta
    U, b = utility(theta, c)
    p = softmax(lam * U)
    dU = np.zeros((3, 8))
    dUdb = (np.log(L0T) - np.log(L0F)) + we * c["s"] * C
    dU[:, 0] = dUdb * b * (1 - b)
    dU[:, 2] = -c["a"] * C
    dU[:, 3] = -c["s"] * (1 - b) * C
    dU[:, 4] = -C
    dU[:, 5] = -c["f"] * C
    dU[:, 6] = c["p"] * M
    dU[:, 7] = np.log(STYLE[c["m"]])
    dz = lam * dU
    dz[:, 1] = U
    dp = p[:, None] * (dz - p @ dz)
    pb = sig(ETA * np.log(b / (1 - b)))
    dpb = np.zeros(8)
    dpb[0] = ETA * pb * (1 - pb)                      # d logit(b)/d beta0 = 1
    return dp, dpb, p, pb


def fim(theta, contexts, joint, n=1.0):
    Fm = np.zeros((8, 8))
    for name in contexts:
        dp, dpb, p, pb = jac(theta, ctx(name))
        Fm += n * (dp.T / p) @ dp
        if joint:
            Fm += n * np.outer(dpb, dpb) / (pb * (1 - pb))
    return Fm


def rank_info(Fm):
    s = np.linalg.svd(Fm, compute_uv=False)
    return int((s / s.max() > TOL).sum()), s


def null_dirs(Fm):
    w, V = np.linalg.eigh(Fm)
    keep = w / w.max() <= TOL
    return [{NAMES[i]: round(float(v[i]), 4) for i in range(8) if abs(v[i]) > 1e-3} for v in V[:, keep].T]


def fd_check():
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(10):
        th = REF + rng.uniform(-.3, .3, 8)
        for name in CONTEXTS:
            dp, dpb, _, _ = jac(th, ctx(name))
            for k in range(8):
                e = np.zeros(8); e[k] = 1e-6
                p1, b1 = probs(th + e, ctx(name)); p2, b2 = probs(th - e, ctx(name))
                worst = max(worst, np.abs((p1 - p2) / 2e-6 - dp[:, k]).max(), abs((b1 - b2) / 2e-6 - dpb[k]))
    return worst


def designs():
    d = {"passive": ["C0"]}
    for k, v in PROBES.items():
        d[f"C0+{k}"] = ["C0"] + v
    d["C0+all"] = list(CONTEXTS)
    d["C0+A+S+F (no E,P,M)"] = ["C0", "A0", "A1", "S0", "S1", "F0", "F1"]
    return d


def rank_table():
    rng = np.random.default_rng(20260926)
    pts = [np.concatenate([[rng.uniform(-1, 2), rng.uniform(1, 3)], rng.uniform(.2, 2, 6)]) for _ in range(25)]
    rows = []
    for dname, cs in designs().items():
        for joint in (False, True):
            ranks = [rank_info(fim(p, cs, joint))[0] for p in pts]
            Fr = fim(REF, cs, joint)
            r_ref, s_ref = rank_info(Fr)
            rows.append({"design": dname, "observation": "joint" if joint else "text_only", "contexts": len(cs),
                         "max_rank": max(ranks), "points_at_max_rank": ranks.count(max(ranks)), "min_rank": min(ranks),
                         "rank_at_reference": r_ref, "min_rel_singular_ref": float(s_ref.min() / s_ref.max()),
                         "null_directions_at_reference": null_dirs(Fr)})
    return rows


def tv_laws(t1, t2, cs, joint):
    out = {}
    for name in cs:
        p1, b1 = probs(t1, ctx(name)); p2, b2 = probs(t2, ctx(name))
        if joint:
            l1 = np.concatenate([p1 * b1, p1 * (1 - b1)]); l2 = np.concatenate([p2 * b2, p2 * (1 - b2)])
        else:
            l1, l2 = p1, p2
        out[name] = float(.5 * np.abs(l1 - l2).sum())
    return out


def e1():
    t1 = REF.copy()
    t2 = REF.copy(); t2[4] += .5; t2[2] -= 1.0                     # kappa0 unchanged: +.5 - .5*1
    return {"theta": dict(zip(NAMES, t1.tolist())), "theta_prime": dict(zip(NAMES, t2.tolist())),
            "TV_passive_text": tv_laws(t1, t2, ["C0"], False)["C0"], "TV_passive_joint": tv_laws(t1, t2, ["C0"], True)["C0"],
            "TV_by_context_joint": tv_laws(t1, t2, list(CONTEXTS), True)}


def e2():
    t1 = REF.copy()
    b0p = t1[0] + 1.0
    c0 = ctx("C0")
    U1, b1 = utility(t1, c0)
    target = t1[1] * (U1[:2] - U1[2])
    bp = sig(b0p)
    ui = u_inf(bp)
    # lam [ (ui_u - ui_w) - kappa (c_u - c_w) + wp .5 (m_u - m_w) - (cost_u - cost_w) ] = target
    A = t1[1] * np.column_stack([-(C[:2] - C[2]), c0["p"] * (M[:2] - M[2])])
    rhs = target - t1[1] * ((ui[:2] - ui[2]) - (COST[:2] - COST[2]))
    kappa, wp = np.linalg.solve(A, rhs)
    t2 = t1.copy(); t2[0] = b0p; t2[6] = wp
    t2[4] = kappa - (.5 * t1[2] + .5 * (1 - bp) * t1[3] + .5 * t1[5])
    feasible = bool(t2[4] >= 0 and t2[6] >= 0)
    return {"theta_prime": dict(zip(NAMES, t2.tolist())), "feasible_nonnegative": feasible,
            "TV_passive_text": tv_laws(t1, t2, ["C0"], False)["C0"], "TV_passive_joint": tv_laws(t1, t2, ["C0"], True)["C0"],
            "TV_evidence_probe_text": tv_laws(t1, t2, ["E"], False)["E"]}


def e3():
    base = REF.copy()
    tA = base.copy(); tA[2], tA[4] = 2.0, 0.0
    tB = base.copy(); tB[2], tB[4] = 0.0, 2.0
    c0 = ctx("C0")
    mix = .5 * probs(tA, c0)[0] + .5 * probs(tB, c0)[0]
    target = np.log(mix[:2] / mix[2])
    U_ref, b = utility(base, c0)
    ui = u_inf(b)
    lam = base[1]
    A = lam * np.column_stack([-(C[:2] - C[2]), c0["p"] * (M[:2] - M[2])])
    rhs = target - lam * ((ui[:2] - ui[2]) - (COST[:2] - COST[2]))
    kappa, wp = np.linalg.solve(A, rhs)
    th = base.copy(); th[2] = 0.0; th[6] = wp
    th[4] = kappa - (.5 * (1 - b) * base[3] + .5 * base[5])
    single = float(.5 * np.abs(probs(th, c0)[0] - mix).sum())

    def pair(ctx1, ctx2, types):
        law = np.zeros((3, 3))
        for w, t in types:
            law += w * np.outer(probs(t, ctx(ctx1))[0], probs(t, ctx(ctx2))[0])
        return law
    out = {"homogeneous_theta": dict(zip(NAMES, th.tolist())), "feasible_nonnegative": bool(th[4] >= 0 and th[6] >= 0),
           "TV_single_observation_passive": single}
    for c1, c2 in (("C0", "C0"), ("A0", "A1")):
        out[f"TV_pair_{c1}_{c2}"] = float(.5 * np.abs(pair(c1, c2, [(.5, tA), (.5, tB)]) - pair(c1, c2, [(1.0, th)])).sum())
    return out


def functionals():
    def f_hedge(ctxname):
        return lambda t: probs(t, ctx(ctxname))[0][1]
    fs = {"P(hedge|C0)": f_hedge("C0"), "P(hedge|a=0)": f_hedge("A0"), "P(hedge|e=1)": f_hedge("E"),
          "P(B=1|C0)": lambda t: probs(t, ctx("C0"))[1],
          "kappa0_at_C0": lambda t: t[4] + .5 * t[2] + .5 * (1 - sig(t[0])) * t[3] + .5 * t[5],
          "w_face": lambda t: t[2], "w_opt": lambda t: t[5], "w_lc": lambda t: t[4]}
    rows = []
    for dname, cs in designs().items():
        for joint in (False, True):
            Fm = fim(REF, cs, joint)
            w, V = np.linalg.eigh(Fm)
            R = V[:, w / w.max() > TOL]
            for fname, f in fs.items():
                g = np.array([(f(REF + e) - f(REF - e)) / 2e-6 for e in np.eye(8) * 1e-6])
                resid = float(np.linalg.norm(g - R @ (R.T @ g)) / max(np.linalg.norm(g), 1e-300))
                rows.append({"design": dname, "observation": "joint" if joint else "text_only", "functional": fname,
                             "identified": resid < 1e-6, "row_space_residual": resid})
    return rows


def recovery():
    rng = np.random.default_rng(20260927)
    authors = [np.concatenate([[rng.uniform(-1, 2), rng.uniform(1, 3)], rng.uniform(.2, 2, 6)]) for _ in range(40)]
    lo = np.array([-4, .1] + [0] * 6); hi = np.array([4, 10] + [10] * 6)
    starts = [REF, np.array([0, 1.5] + [.5] * 6), np.array([1, 3] + [2] * 6)]
    dsets = {"P_text": (["C0"] * 1, 120, False), "P_joint": (["C0"], 120, True), "F_joint": (list(CONTEXTS), 10, True)}

    def neglogpost(th, data, joint):
        v = 0.0
        for name, n_u, k, n in data:
            p, pb = probs(th, ctx(name))
            v -= float(n_u @ np.log(p + 1e-300))
            if joint:
                v -= k * np.log(pb + 1e-300) + (n - k) * np.log(1 - pb + 1e-300)
        v += th[0] ** 2 / 18 + (np.log(th[1]) - np.log(2)) ** 2 / 2 + (th[2:] ** 2).sum() / 18
        return v

    def prior_prec(th):
        P = np.zeros((8, 8)); P[0, 0] = 1 / 9; P[1, 1] = 1 / th[1] ** 2; P[2:, 2:] = np.eye(6) / 9
        return P
    rows = []
    for ai, truth in enumerate(authors):
        for dname, (cs, n, joint) in dsets.items():
            data = []
            for name in cs:
                p, pb = probs(truth, ctx(name))
                data.append((name, rng.multinomial(n, p).astype(float), float(rng.binomial(n, pb)), n))
            best = None
            for s in starts:
                r = minimize(neglogpost, s, args=(data, joint), method="L-BFGS-B", bounds=list(zip(lo, hi)))
                if best is None or r.fun < best.fun:
                    best = r
            th = best.x
            Fm = fim(th, cs, joint, n=n) + prior_prec(th)
            sd = np.sqrt(np.clip(np.diag(np.linalg.inv(Fm)), 0, None))
            row = {"author": ai, "design": dname, "converged": bool(best.success)}
            for k, nm in enumerate(NAMES):
                row[f"est_{nm}"] = float(th[k]); row[f"sd_{nm}"] = float(sd[k]); row[f"true_{nm}"] = float(truth[k])
            k_true = truth[4] + .5 * truth[2] + .5 * (1 - sig(truth[0])) * truth[3] + .5 * truth[5]
            k_est = th[4] + .5 * th[2] + .5 * (1 - sig(th[0])) * th[3] + .5 * th[5]
            row.update({"true_kappa0": float(k_true), "est_kappa0": float(k_est),
                        "true_phedge_C0": float(probs(truth, ctx("C0"))[0][1]), "est_phedge_C0": float(probs(th, ctx("C0"))[0][1])})
            rows.append(row)
    return rows


def main() -> int:
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA:
        raise RuntimeError("protocol sha mismatch")
    if OUT.exists():
        raise FileExistsError("speaker_identifiability_r1 exists; preserve earlier runs")
    OUT.mkdir(parents=True)
    t0 = time.time()
    res = {"presentation_profile_m": M.tolist(), "commitment_c": C.tolist(), "fd_max_abs_error": fd_check()}
    if res["fd_max_abs_error"] > 1e-6:
        raise RuntimeError("analytic Jacobian failed the finite-difference check; stop")
    res["rank_table"] = rank_table()
    res["E1_commitment_aliasing"] = e1()
    res["E2_epistemic_vs_preference"] = e2()
    res["E3_mixture_vs_homogeneous"] = e3()
    res["functionals"] = functionals()
    res["expected_information_per_observation_ref"] = {d: np.linalg.eigvalsh(fim(REF, cs, True) / len(cs)).round(8).tolist() for d, cs in designs().items()}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    import pandas as pd
    pd.DataFrame(recovery()).to_csv(OUT / "recovery.csv", index=False)
    (OUT / "runtime.json").write_text(json.dumps({"seconds": time.time() - t0, "neural_calls": 0,
                                                  "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=1) + "\n")
    print(json.dumps({"seconds": time.time() - t0, "fd": res["fd_max_abs_error"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
