#!/usr/bin/env python
"""R08 controlled source-fitted predictive-state witness (docs/GRAIN_R08_CONTROLLED_PSR_WITNESS_PROTOCOL_V1.md)."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_grain_r08_text_behaviour_quotient as r1  # noqa: E402  (frozen kernels, imported only)

PROTOCOL = ROOT / "docs/GRAIN_R08_CONTROLLED_PSR_WITNESS_PROTOCOL_V1.md"
PROTOCOL_SHA = "05edd6302634fa60f5709b5c43f201047da73240e08571b5afb3aed0d7449f5d"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F15/controlled_psr_witness_r1"
L = 10
N_SRC, N_VAL, N_APP = 8000, 1500, 2500
SEEDS = (20261101, 20261102, 20261103)
EPS = 1e-6


# ---------------------------------------------------------------- generator (the only user of the kernel)
def construct_kernel(name):
    w = r1.world_W(name)
    S = len(w["states"])
    K = np.zeros((2, S, 4, S))
    for a in (0, 1):
        for s in range(S):
            for oi, (c, f, b) in enumerate(w["obs"]):
                for s2 in range(S):
                    K[a, s, 2 * c + b, s2] += float(w["K"][a][s][oi][s2])
    T = K.sum(axis=2)
    Tmix = .5 * (T[0] + T[1])
    xs = np.array([st[0] for st in w["states"]])
    init = np.zeros(S)
    for x in (0, 1):
        idx = np.where(xs == x)[0]
        p = np.full(len(idx), 1 / len(idx))
        sub = Tmix[np.ix_(idx, idx)]
        for _ in range(5000):
            p = p @ sub
        init[idx] = .5 * p / p.sum()
    return K, init


def policy_actions(kind, t, prev, rng, n):
    if kind == "random":
        return rng.integers(0, 2, n)
    if kind == "a0only":
        return np.zeros(n, int)
    if kind == "no_a1_after_a1":
        a = rng.integers(0, 2, n)
        a[prev == 1] = 0
        return a
    raise ValueError(kind)


def generate(K, init, n, policy, rng):
    S = len(init)
    flat = K.reshape(2, S, 4 * S)
    cum = np.cumsum(flat, axis=2)
    s = rng.choice(S, size=n, p=init)
    A = np.zeros((n, L), int)
    O = np.zeros((n, L), int)
    prev = np.zeros(n, int)
    for t in range(L):
        a = policy_actions(policy, t, prev, rng, n)
        u = rng.random(n)
        idx = np.minimum((cum[a, s] < u[:, None]).sum(1), 4 * S - 1)
        O[:, t] = idx // S
        s = idx % S
        A[:, t] = a
        prev = a
    return A, O


# ---------------------------------------------------------------- spectral learner (source windows only)
def fit_spectral_stats(A, O):
    Z = 4 * A + O
    n = Z.shape[0]
    CTH = np.zeros((72, 64)); DTH = np.zeros((72, 64))
    CTaoH = np.zeros((8, 72, 64)); DTaoH = np.zeros((8, 72, 64))
    CT = np.zeros(72); DT = np.zeros(72); CH = np.zeros(64); DH = np.zeros(64)
    tidx1 = lambda z: z  # noqa: E731
    tidx2 = lambda z1, z2: 8 + 8 * z1 + z2  # noqa: E731
    A_ = A
    for i in range(L - 4):
        h = 8 * Z[:, i] + Z[:, i + 1]
        ha = (A_[:, i], A_[:, i + 1])
        # history probabilities
        np.add.at(CH, h, 1)
        for hh in range(64):
            pass
        # denominators by action pattern: every observation pattern sharing the same actions
        # P_H: count windows whose history actions equal each history's actions
        hact = 2 * A_[:, i] + A_[:, i + 1]
        for hv in range(64):
            DH[hv] += (hact == (2 * (hv // 8 // 4) + (hv % 8) // 4)).sum()
        # tests after history (P_TH), tests at window start (P_T), and ao-extended (P_TaoH)
        for tlen in (1, 2):
            if tlen == 1:
                t = tidx1(Z[:, i + 2]); tact = A_[:, i + 2]; tstart = tidx1(Z[:, i]); tsact = A_[:, i]
                t_ao = tidx1(Z[:, i + 3]); t_ao_act = A_[:, i + 3]
            else:
                t = tidx2(Z[:, i + 2], Z[:, i + 3]); tact = 2 * A_[:, i + 2] + A_[:, i + 3]
                tstart = tidx2(Z[:, i], Z[:, i + 1]); tsact = 2 * A_[:, i] + A_[:, i + 1]
                t_ao = tidx2(Z[:, i + 3], Z[:, i + 4]); t_ao_act = 2 * A_[:, i + 3] + A_[:, i + 4]
            np.add.at(CTH, (t, h), 1)
            np.add.at(CT, tstart, 1)
            ao = Z[:, i + 2]
            np.add.at(CTaoH, (ao, t_ao, h), 1)
            rng_t = range(0, 8) if tlen == 1 else range(8, 72)
            for tv in rng_t:
                if tlen == 1:
                    ta = tv // 4
                    m_t = tact == ta; m_ts = tsact == ta; m_tao = t_ao_act == ta
                else:
                    z1, z2 = (tv - 8) // 8, (tv - 8) % 8
                    ta = 2 * (z1 // 4) + z2 // 4
                    m_t = tact == ta; m_ts = tsact == ta; m_tao = t_ao_act == ta
                DT[tv] += m_ts.sum()
                for hv in range(64):
                    hav = 2 * ((hv // 8) // 4) + (hv % 8) // 4
                    mh = hact == hav
                    DTH[tv, hv] += (m_t & mh).sum()
                    for aov in range(8):
                        DTaoH[aov, tv, hv] += (m_tao & mh & (A_[:, i + 2] == aov // 4)).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        P = {"TH": CTH / DTH, "T": CT / DT, "H": CH / DH, "TaoH": CTaoH / DTaoH}
    P["support"] = {"TH": DTH > 0, "T": DT > 0, "H": DH > 0, "TaoH": DTaoH > 0}
    P["windows"] = int(n * (L - 4))
    return P


def restrict_support(P, actions_supported):
    """Greedy removal of rows/columns until every retained cell (TH, T, H and TaoH for supported ao) is supported."""
    rows = [t for t in range(72) if all((a in actions_supported) for a in ([t // 4] if t < 8 else [((t - 8) // 8) // 4, ((t - 8) % 8) // 4]))]
    cols = [h for h in range(64) if (h // 8) // 4 in actions_supported and (h % 8) // 4 in actions_supported]
    aos = [z for z in range(8) if z // 4 in actions_supported]
    while True:
        bad = np.zeros((72, 64), int)
        for t in rows:
            for h in cols:
                ok = P["support"]["TH"][t, h] and all(P["support"]["TaoH"][z, t, h] for z in aos)
                bad[t, h] = 0 if ok else 1
        rb = {t: bad[t, cols].sum() for t in rows}
        cb = {h: bad[rows, h].sum() for h in cols}
        worst_r = max(rb.items(), key=lambda kv: kv[1]) if rb else (None, 0)
        worst_c = max(cb.items(), key=lambda kv: kv[1]) if cb else (None, 0)
        if worst_r[1] == 0 and worst_c[1] == 0:
            break
        if worst_r[1] / max(len(cols), 1) >= worst_c[1] / max(len(rows), 1):
            rows.remove(worst_r[0])
        else:
            cols.remove(worst_c[0])
    rows = [t for t in rows if P["support"]["T"][t]]
    cols = [h for h in cols if P["support"]["H"][h]]
    return rows, cols, aos


def spectral_model(P, k, rows, cols, aos):
    TH = P["TH"][np.ix_(rows, cols)]
    U, s, _ = np.linalg.svd(TH, full_matrices=False)
    Uk = U[:, :k]
    UP = Uk.T @ TH
    UPp = np.linalg.pinv(UP)
    b1 = Uk.T @ P["T"][rows]
    binf = P["H"][cols] @ UPp
    B = {z: (Uk.T @ P["TaoH"][z][np.ix_(rows, cols)] @ UPp if z in aos else None) for z in range(8)}
    b1 = b1 / (binf @ b1)
    return {"k": k, "b1": b1, "binf": binf, "B": B, "sing": s.tolist()}


def psr_cond(m, b, a):
    if any(m["B"][4 * a + o] is None for o in range(4)):
        return None, None
    raw = np.array([m["binf"] @ m["B"][4 * a + o] @ b for o in range(4)])
    p = np.clip(raw, EPS, None)
    return p / p.sum(), raw


def psr_update(m, b, a, o):
    Bz = m["B"][4 * a + o]
    if Bz is None:
        return None, False
    v = Bz @ b
    d = m["binf"] @ v
    if d <= 0:
        return m["b1"].copy(), True
    return v / d, False


# ---------------------------------------------------------------- baselines
def pop_given_action(A, O):
    P = np.full((2, 4), np.nan)
    for a in (0, 1):
        m = A == a
        if m.sum():
            P[a] = (np.bincount(O[m], minlength=4) + 1) / (m.sum() + 4)
    return P


def order1_table(A, O):
    C = np.ones((4, 2, 4))
    C[:, :, :] = 1
    seen = np.zeros((4, 2), bool)
    for t in range(1, L):
        np.add.at(C, (O[:, t - 1], A[:, t], O[:, t]), 1)
        np.logical_or.at(seen, (O[:, t - 1], A[:, t]), True)
    return C / C.sum(2, keepdims=True), seen


def window_features(A, O, t):
    X = np.zeros((A.shape[0], 32))
    for j in range(4):
        z = 4 * A[:, t - 4 + j] + O[:, t - 4 + j]
        X[np.arange(A.shape[0]), 8 * j + z] = 1
    return X


def fit_pca_decoder(A, O, k):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    Xs, As, Ys = [], [], []
    for t in range(4, L):
        Xs.append(window_features(A, O, t)); As.append(A[:, t]); Ys.append(O[:, t])
    X, Aa, Y = np.vstack(Xs), np.concatenate(As), np.concatenate(Ys)
    pca = PCA(n_components=k, random_state=0).fit(X)
    dec = {}
    for a in (0, 1):
        m = Aa == a
        if m.sum() > 50 and len(np.unique(Y[m])) == 4:
            dec[a] = LogisticRegression(C=1.0, max_iter=2000).fit(pca.transform(X[m]), Y[m])
    return pca, dec


def oracle_filter(K, init):
    def run(Aseq, Oseq, upto):
        b = init.copy()
        for t in range(upto):
            b = b @ K[Aseq[t], :, Oseq[t], :]
            b = b / b.sum()
        return b
    def pred(b, a):
        return b @ K[a].sum(axis=2)
    return run, pred


# ---------------------------------------------------------------- scoring
def score_application(name, K, init, m, popP, o1, o1_seen, pca, dec, alpha, Aap, Oap):
    run, opred = oracle_filter(K, init)
    n = Aap.shape[0]
    res = {k: [] for k in ("psr", "static", "order1", "pca", "oracle")}
    refused = {k: 0 for k in res}
    clip_count, reset_count, total = 0, 0, 0
    two = {k: [] for k in res}
    closure = 0.0
    collision_rows = []
    for i in range(n):
        A_, O_ = Aap[i], Oap[i]
        b = m["b1"].copy(); psr_ok = True
        for t in range(L):
            a, o = A_[t], O_[t]
            if t >= 4:
                preds = {}
                pc, raw = psr_cond(m, b, a) if psr_ok else (None, None)
                preds["psr"] = pc
                if raw is not None:
                    clip_count += int((raw < EPS).sum()); total += 4
                cnt = np.bincount(O_[:t][A_[:t] == a], minlength=4)
                preds["static"] = None if np.isnan(popP[a]).any() else (cnt + alpha * popP[a]) / (cnt.sum() + alpha)
                preds["order1"] = o1[O_[t - 1], a] if o1_seen[O_[t - 1], a] else None
                preds["pca"] = dec[a].predict_proba(pca.transform(window_features(A_[None], O_[None], t)))[0] if a in dec else None
                preds["oracle"] = opred(run(A_, O_, t), a)
                for k2, p in preds.items():
                    if p is None:
                        refused[k2] += 1
                        res[k2].append((i, t, np.nan, np.nan))
                    else:
                        res[k2].append((i, t, -np.log(max(p[o], 1e-12)), float(((p - np.eye(4)[o]) ** 2).sum())))
                if t == 4:
                    # two-step continuation (o4, o5) given (a4, a5)
                    a5, o5 = A_[5], O_[5]
                    j = {}
                    if preds["psr"] is not None:
                        b2, _ = psr_update(m, b, a, o)
                        p2, raw2 = psr_cond(m, b2, a5) if b2 is not None else (None, None)
                        j["psr"] = None if p2 is None else preds["psr"][o] * p2[o5]
                        if p2 is not None:
                            direct = m["binf"] @ m["B"][4 * a5 + o5] @ m["B"][4 * a + o] @ b
                            prod = (m["binf"] @ m["B"][4 * a + o] @ b) * (raw2[o5])
                            closure = max(closure, abs(direct - prod))
                    else:
                        j["psr"] = None
                    cnt2 = cnt.copy()
                    if a5 == a:
                        cnt2[o] += 1
                    if preds["static"] is not None:
                        c5 = np.bincount(np.append(O_[:5], [])[A_[:5] == a5], minlength=4) if False else np.bincount(O_[:5][A_[:5] == a5], minlength=4)
                        j["static"] = preds["static"][o] * ((c5 + alpha * popP[a5]) / (c5.sum() + alpha))[o5]
                    else:
                        j["static"] = None
                    j["order1"] = None if preds["order1"] is None or not o1_seen[o, a5] else preds["order1"][o] * o1[o, a5][o5]
                    if preds["pca"] is not None and a5 in dec:
                        j["pca"] = preds["pca"][o] * dec[a5].predict_proba(pca.transform(window_features(A_[None], O_[None], 5)))[0][o5]
                    else:
                        j["pca"] = None
                    ob = run(A_, O_, 5)
                    j["oracle"] = preds["oracle"][o] * opred(ob, a5)[o5]
                    for k2, v in j.items():
                        two[k2].append((i, np.nan if v is None else -np.log(max(v, 1e-12))))
                    collision_rows.append({"i": i, "counts": tuple(np.bincount(4 * A_[:4] + O_[:4], minlength=8)), "last": int(O_[3]), "a4": int(a),
                                           "oracle": preds["oracle"].tolist(), "psr": None if preds["psr"] is None else preds["psr"].tolist(),
                                           "static": None if preds["static"] is None else preds["static"].tolist(),
                                           "order1": None if preds["order1"] is None else preds["order1"].tolist()})
            if psr_ok:
                b, reset = psr_update(m, b, a, o)
                if b is None:
                    psr_ok = False
                    b = m["b1"].copy()
                reset_count += int(reset)
    return res, two, refused, {"clip_rate": clip_count / max(total, 1), "resets": reset_count, "closure_max_abs": closure}, collision_rows


def collisions(rows):
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        g[(r["counts"], r["last"], r["a4"])].append(r)
    out = []
    for key, rs in g.items():
        for x in range(len(rs)):
            for y in range(x + 1, len(rs)):
                tv = .5 * np.abs(np.array(rs[x]["oracle"]) - np.array(rs[y]["oracle"])).sum()
                if tv > .1:
                    out.append({"tv_oracle": float(tv), "a": rs[x], "b": rs[y]})
    out.sort(key=lambda d: -d["tv_oracle"])
    return out


def bootstrap(res, ref="psr", draws=2000, seed=20260926):
    import pandas as pd
    df = pd.DataFrame([(k, i, t, ll, br) for k, v in res.items() for (i, t, ll, br) in v], columns=["model", "i", "t", "ll", "brier"])
    piv = df.pivot_table(index=["i", "t"], columns="model", values="ll")
    per_seq = piv.groupby(level=0).mean()
    rng = np.random.default_rng(seed)
    ids = per_seq.index.to_numpy()
    out = {}
    for mname in per_seq.columns:
        out[mname] = {"mean_logloss": float(piv[mname].mean()), "coverage": float(piv[mname].notna().mean()),
                      "mean_brier": float(df[df.model == mname].brier.mean())}
        if mname != ref:
            d = (per_seq[mname] - per_seq[ref]).dropna()
            bs = [d.iloc[rng.integers(0, len(d), len(d))].mean() for _ in range(draws)] if len(d) else [np.nan]
            out[mname][f"delta_vs_{ref}"] = float(d.mean()) if len(d) else None
            out[mname][f"delta_ci_vs_{ref}"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(d) else None
    return out


def run_arm(name, seed, src_policy="random", app_policy="random", n_val=N_VAL, label=None):
    t0 = time.time()
    K, init = construct_kernel(name)
    rng = np.random.default_rng(seed)
    As, Os = generate(K, init, N_SRC, src_policy, rng)
    Av, Ov = generate(K, init, n_val, src_policy, rng)
    Aa, Oa = generate(K, init, N_APP, app_policy, rng)
    steps = (N_SRC + n_val + N_APP) * L
    sup = sorted(set(np.unique(As).tolist()))
    P = fit_spectral_stats(As, Os)
    rows, cols, aos = restrict_support(P, sup)
    # rank by source-validation log loss (oracle never consulted)
    val = {}
    for k in range(1, 9):
        if k > min(len(rows), len(cols)):
            break
        m = spectral_model(P, k, rows, cols, aos)
        ll, cnt = 0.0, 0
        for i in range(n_val):
            b = m["b1"].copy()
            for t in range(L):
                if t >= 2:
                    p, _ = psr_cond(m, b, Av[i, t])
                    if p is not None:
                        ll += -np.log(max(p[Ov[i, t]], 1e-12)); cnt += 1
                b, _ = psr_update(m, b, Av[i, t], Ov[i, t])
                if b is None:
                    break
        val[k] = ll / max(cnt, 1)
    k_best = min(val, key=val.get)
    m = spectral_model(P, k_best, rows, cols, aos)
    # source reconstruction gate on retained tests from b1
    recon = {}
    for t in rows:
        seq = [t] if t < 8 else [(t - 8) // 8, (t - 8) % 8]
        b = m["b1"].copy(); prob = 1.0
        for z in seq:
            Bz = m["B"][z]
            if Bz is None:
                prob = np.nan; break
            v = Bz @ b; p = m["binf"] @ v
            prob *= p
            b = v / p if p > 0 else m["b1"]
        recon[t] = (prob, P["T"][t])
    groups = {}
    for t, (pm, pe) in recon.items():
        act = (t // 4,) if t < 8 else (((t - 8) // 8) // 4, ((t - 8) % 8) // 4)
        groups.setdefault(act, []).append(abs(pm - pe))
    recon_tv = {str(k): float(.5 * np.nansum(v)) for k, v in groups.items()}
    gate = bool(all(v <= .02 for v in recon_tv.values()))
    norm_dev = float(max(abs(sum(m["binf"] @ m["B"][4 * a + o] @ m["b1"] for o in range(4)) - 1) for a in sup))
    popP = pop_given_action(As, Os)
    o1, o1_seen = order1_table(As, Os)
    alphas = {}
    for alpha in (1, 3, 10, 30):
        ll, cnt = 0.0, 0
        for t in range(4, L):
            for i in range(n_val):
                a, o = Av[i, t], Ov[i, t]
                if np.isnan(popP[a]).any():
                    continue
                c = np.bincount(Ov[i, :t][Av[i, :t] == a], minlength=4)
                p = (c + alpha * popP[a]) / (c.sum() + alpha)
                ll += -np.log(p[o]); cnt += 1
        alphas[alpha] = ll / max(cnt, 1)
    alpha = min(alphas, key=alphas.get)
    pca, dec = fit_pca_decoder(As, Os, k_best)
    res, two, refused, diag, crow = score_application(name, K, init, m, popP, o1, o1_seen, pca, dec, alpha, Aa, Oa)
    summary = bootstrap(res)
    two_step = {k: float(np.nanmean([v for _, v in lst])) if lst else None for k, lst in two.items()}
    two_cov = {k: float(np.mean([not np.isnan(v) for _, v in lst])) if lst else None for k, lst in two.items()}
    col = collisions(crow)
    out = {"arm": label or f"{name}_{seed}", "world": name, "seed": seed, "src_policy": src_policy, "app_policy": app_policy,
           "generated_steps": steps, "source_windows": P["windows"], "supported_actions": sup,
           "retained_tests": len(rows), "retained_histories": len(cols), "rank_validation_logloss": val, "rank": k_best,
           "singular_values_TH": [round(x, 6) for x in m["sing"][:10]], "alpha_static": alpha,
           "source_reconstruction_TV_by_action_seq": recon_tv, "support_gate_pass": gate, "normalization_deviation_b1": norm_dev,
           "diagnostics": diag, "refused_next_step": refused, "next_step": summary, "two_step_logloss": two_step, "two_step_coverage": two_cov,
           "history_collisions_found": len(col), "collision_examples": col[:2], "seconds": time.time() - t0}
    if src_policy == "no_a1_after_a1":
        import pandas as pd
        df = pd.DataFrame([(k, i, t, ll) for k, v in res.items() for (i, t, ll, _b) in v], columns=["model", "i", "t", "ll"])
        mask = [(i, t) for i in range(N_APP) for t in range(4, L) if Aa[i, t] == 1 and Aa[i, t - 1] == 1]
        ms = set(mask)
        df["novel_combo"] = [(i, t) in ms for i, t in zip(df.i, df.t)]
        out["novel_combination_steps"] = len(mask)
        out["novel_combination_logloss"] = df[df.novel_combo].groupby("model").ll.mean().to_dict()
        out["supported_combination_logloss"] = df[~df.novel_combo].groupby("model").ll.mean().to_dict()
    if src_policy == "a0only":
        out["a1_steps_in_application"] = int(sum(1 for i in range(N_APP) for t in range(4, L) if Aa[i, t] == 1))
    return out


def main() -> int:
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA:
        raise RuntimeError("protocol sha mismatch")
    if OUT.exists():
        raise FileExistsError("controlled_psr_witness_r1 exists; preserve earlier runs")
    OUT.mkdir(parents=True)
    t0 = time.time()
    from joblib import Parallel, delayed
    arms = [("W1", s, "random", "random", None) for s in SEEDS] + [("W6", s, "random", "random", None) for s in SEEDS]
    arms += [("W6", SEEDS[0], "a0only", "random", "W6_V_unsup"), ("W6", SEEDS[0], "no_a1_after_a1", "random", "W6_V_comb")]
    res = Parallel(n_jobs=2)(delayed(run_arm)(w, s, sp, ap, label=lab) for w, s, sp, ap, lab in arms)
    total_steps = sum(r["generated_steps"] for r in res)
    if total_steps > 1_000_000:
        raise RuntimeError("step budget exceeded")
    unsup = next(r for r in res if r["arm"] == "W6_V_unsup")
    q3_ok = unsup["refused_next_step"]["psr"] >= unsup["a1_steps_in_application"] and unsup["refused_next_step"]["static"] >= unsup["a1_steps_in_application"]
    (OUT / "results.json").write_text(json.dumps({"arms": res, "total_generated_steps": total_steps, "Q3_unsupported_refusal_ok": bool(q3_ok)}, indent=1, default=str) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"seconds": time.time() - t0, "neural_calls": 0, "workers": 2,
                                                  "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=1) + "\n")
    print(json.dumps({"seconds": time.time() - t0, "steps": total_steps, "Q3": q3_ok}))
    return 0 if q3_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
