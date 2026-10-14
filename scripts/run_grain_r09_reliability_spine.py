#!/usr/bin/env python
"""R09 reliability spine runner (docs/GRAIN_R09_RELIABILITY_SPINE_PROTOCOL_V1.md). CPU, one process, single-thread BLAS."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import reliability_spine as rsp  # noqa: E402

PROTOCOL = ROOT / "docs/GRAIN_R09_RELIABILITY_SPINE_PROTOCOL_V1.md"
PROTOCOL_SHA = "6ca1b74a2b3e2bc8ab5917884e1592a60b289f31fc8fb8cdf17765308ab02012"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F04/reliability_spine_r1"
TDIR = ROOT / "results/grain_gptpro_foundation_v3/F04/source_full_r1"
T_PINS = {"T_20260915": ("807249d63a6424df06f0b54ffde6babee457399f581ac76cf31c6c6170c87430", "81d43ef9eea9450ad906cc0be3705e6fd8baea3be95d0551cf8b8fbdc42f57f2"),
          "T_20260916": ("ed8685e249c681b37adf581b34464c52f90a8596e7072c48be62c0c813ff03f7", "fd9bb812d8eaf088513052662081245ee4a162b1ff0984603693c75c39381839"),
          "T_20260917": ("aec8137734820e2f5bee448f33fde4337bab35cf5df66f23a662781975e427ac", "b7225cd712d9aa2adcc327aa4ed5686f0ab86618c4b7853f83a11e0111c71c24")}
SEEDS, BOOT, BSEED = (1, 2, 3), 1000, 18091
NAMES = ["a1", "a2", "B11", "B12", "B21", "B22"]
T_STATE = (0, 11, 23)
SHARED = [((0, f), (1, f)) for f in range(3)]
DISJOINT = [((0, f), (1, g)) for f in range(3) for g in range(3) if f != g]
MAX_ROWS = 500_000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def num_hessian(f, x, h=1e-4):
    n = len(x); H = np.zeros((n, n)); f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            e_i = np.zeros(n); e_i[i] = h; e_j = np.zeros(n); e_j[j] = h
            if i == j:
                H[i, i] = (f(x + e_i) - 2 * f0 + f(x - e_i)) / h ** 2
            else:
                H[i, j] = H[j, i] = (f(x + e_i + e_j) - f(x + e_i - e_j) - f(x - e_i + e_j) + f(x - e_i - e_j)) / (4 * h ** 2)
    return H


def fit_source(data):
    ns = data["n_src"]; Y = data["Y"][:ns]
    groups = {}
    for u in range(ns):
        groups.setdefault(data["pairs"][u], []).append(u)
    groups = [(p, Y[idx]) for p, idx in groups.items()]
    ols = np.array([rsp.ols_theta(Y[u])[0] for u in range(ns)])
    S0 = np.cov(ols.T) + 1e-6 * np.eye(6)
    x0 = rsp.pack(S0, .1, .5, .2, .2)
    f = lambda x: rsp.reml_objective(x, groups)
    t0 = time.time()
    r1 = minimize(f, x0, method="L-BFGS-B", options={"maxiter": 1000})
    attempts = [{"method": "L-BFGS-B", "success": bool(r1.success), "fun": float(r1.fun), "nit": int(r1.nit), "message": str(r1.message)}]
    best = r1
    if not r1.success:
        r2 = minimize(f, r1.x, method="Powell", options={"maxiter": 4000})
        attempts.append({"method": "Powell", "success": bool(r2.success), "fun": float(r2.fun), "nit": int(r2.nit), "message": str(r2.message)})
        if r2.success or r2.fun < r1.fun:
            best = r2
        if not r2.success and not r1.success:
            # both flagged; keep the lower objective but mark
            attempts.append({"note": "neither optimizer reported success; lower objective kept and flagged"})
    par = rsp.unpack(best.x)
    beta = rsp.gls_beta(par, groups)
    H = num_hessian(f, best.x)
    try:
        cov = 2 * np.linalg.inv(H)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(len(best.x), np.nan)
    flags = {"variance_below_1e-6": bool(min(par["sd2"], par["sw2"], par["se2"], np.linalg.eigvalsh(par["Sig"]).min()) < 1e-6), "abs_rho_above_.99": abs(par["rho"]) > .99}
    return {"par": par, "beta": beta, "psi": best.x, "attempts": attempts, "flags": flags, "seconds": time.time() - t0,
            "se": {"sd2": float(par["sd2"] * se[21]), "rho": float((1 - par["rho"] ** 2) * se[22]), "sw2": float(par["sw2"] * se[23]), "se2": float(par["se2"] * se[24])},
            "ols_theta_cov_init": S0.tolist()}


def score(data, fit, true_par, true_beta):
    ns = data["n_src"]; n = len(data["Y"])
    cache = {}

    def post(arm, pair, key, rows):
        k = (arm, pair, key)
        if k not in cache:
            par = fit["par"] if arm == "fit" else true_par
            cache[k] = rsp.posterior(par, pair, rows)
        return cache[k]
    out = []
    for u in range(ns, n):
        pair = data["pairs"][u]; y = data["Y"][u]
        rec = {"u": u, "pair": pair, "theta": data["theta"][u], "m_held": data["m_held"][u]}
        full = np.arange(rsp.NROW)
        for arm, beta, par in (("fit", fit["beta"], fit["par"]), ("orc", true_beta, true_par)):
            resid = y - rsp.X @ beta
            Kg, pc = post(arm, pair, "full", full)
            b = Kg @ resid
            th = beta[:6] + b[:6]
            rec[f"theta_{arm}"] = th; rec[f"theta_var_{arm}"] = np.diag(pc)[:6].copy(); rec[f"theta_cov_{arm}"] = pc[:6, :6].copy()
            for t in T_STATE:
                for k in range(2):
                    r = rsp.row_at(pair, 1, t, k)
                    rec[f"s_smooth_{arm}_{t}_{k}"] = (float(b[30 + r]), float(pc[30 + r, 30 + r]))
                    rows = rsp.rows_prefix(pair, 1, t)
                    Kf, pf = post(arm, pair, f"prefix{t}", rows)
                    bf = Kf @ resid[rows]
                    rec[f"s_filter_{arm}_{t}_{k}"] = (float(bf[30 + r]), float(pf[30 + r, 30 + r]))
                    rec[f"s_true_{t}_{k}"] = float(data["S"][u, r])
            if arm == "fit":
                for o in range(2):
                    for f in range(3):
                        rows = rsp.rows_half(o, f)
                        Kh, _ = post(arm, pair, f"half{o}{f}", rows)
                        rec[f"half_{o}{f}"] = beta[:6] + (Kh @ resid[rows])[:6]
        th_ols, s2, XtXi = rsp.ols_theta(y)
        rec["theta_ols"] = th_ols; rec["ols_s2"] = s2; rec["ols_XtXi"] = XtXi
        out.append(rec)
    return out


def known_reliability(true_par, pairs_app, curved):
    res = {}
    pcs = {p: rsp.posterior(true_par, p, np.arange(rsp.NROW))[1] for p in set(pairs_app)}
    for j, nm in enumerate(NAMES):
        res[nm] = rsp.reliability(true_par["Sig"][j, j], [pcs[p][j, j] for p in pairs_app])
    if not curved:
        for k in range(2):
            for h in range(4):
                L = rsp.held_L(k, h)
                res[f"m{k + 1}_held{h}"] = rsp.reliability(L @ true_par["Sig"] @ L, [L @ pcs[p][:6, :6] @ L for p in pairs_app])
    for t in T_STATE:
        res[f"state_o2_t{t}_k1_smoothed"] = rsp.reliability(rsp.SS2, [pcs[p][30 + rsp.row_at(p, 1, t, 0), 30 + rsp.row_at(p, 1, t, 0)] for p in pairs_app])
    return res


def summarize(recs, boot_idx):
    n = len(recs)
    TH = np.array([r["theta"] for r in recs]); FIT = np.array([r["theta_fit"] for r in recs]); ORC = np.array([r["theta_orc"] for r in recs])
    HALF = {h: np.array([r[f"half_{h[0]}{h[1]}"] for r in recs]) for h in [(o, f) for o in range(2) for f in range(3)]}
    MH = np.array([r["m_held"] for r in recs])                     # n x 2 x 4

    def stats(idx):
        s = {}
        for j, nm in enumerate(NAMES):
            for lab, prs in (("shared", SHARED), ("disjoint", DISJOINT)):
                vals = np.array([(*rsp.icc_two_way(np.c_[HALF[a][idx, j], HALF[b][idx, j]]), np.corrcoef(HALF[a][idx, j], HALF[b][idx, j])[0, 1]) for a, b in prs])
                s[f"{nm}|{lab}|iccA"], s[f"{nm}|{lab}|iccC"], s[f"{nm}|{lab}|r"] = vals.mean(0)
            s[f"{nm}|recovery_r_full_fit"] = np.corrcoef(FIT[idx, j], TH[idx, j])[0, 1]
            s[f"{nm}|recovery_r_full_orc"] = np.corrcoef(ORC[idx, j], TH[idx, j])[0, 1]
            s[f"{nm}|half_recovery_r2"] = np.mean([np.corrcoef(HALF[h][idx, j], TH[idx, j])[0, 1] ** 2 for h in HALF])
        for mode in ("smooth", "filter"):
            for arm in ("fit", "orc"):
                for t in T_STATE:
                    for k in range(2):
                        est = np.array([recs[i][f"s_{mode}_{arm}_{t}_{k}"][0] for i in idx]); var = np.array([recs[i][f"s_{mode}_{arm}_{t}_{k}"][1] for i in idx])
                        tru = np.array([recs[i][f"s_true_{t}_{k}"] for i in idx])
                        s[f"state|{mode}|{arm}|t{t}|k{k + 1}|r"] = np.corrcoef(est, tru)[0, 1]
                        s[f"state|{mode}|{arm}|t{t}|k{k + 1}|rmse"] = np.sqrt(np.mean((est - tru) ** 2))
                        s[f"state|{mode}|{arm}|t{t}|k{k + 1}|post_sd"] = np.sqrt(var).mean()
                        s[f"state|{mode}|{arm}|t{t}|k{k + 1}|cov95"] = np.mean(np.abs(est - tru) <= 1.96 * np.sqrt(var))
        # held-context prediction
        for k in range(2):
            Ls = [rsp.held_L(k, h) for h in range(4)]
            tru = MH[idx, k, :]
            preds = {}
            for arm in ("fit", "orc"):
                th = FIT if arm == "fit" else ORC
                m = np.stack([th[idx] @ L for L in Ls], 1)
                v = np.stack([np.array([L @ recs[i][f"theta_cov_{arm}"] @ L for i in idx]) for L in Ls], 1)
                preds[f"EB_{arm}"] = (m, v)
            mu = recs[0]["_mu_fit"]; Sg = recs[0]["_Sig_fit"]
            pm = np.tile([mu @ L for L in Ls], (len(idx), 1)); pv = np.tile([L @ Sg @ L for L in Ls], (len(idx), 1))
            preds["pooled_static"] = (pm, pv)
            om = np.stack([np.array([recs[i]["theta_ols"] @ L for i in idx]) for L in Ls], 1)
            ov = np.stack([np.array([np.r_[1, rsp.HELD[h]] @ recs[i]["ols_XtXi"][k] @ np.r_[1, rsp.HELD[h]] * recs[i]["ols_s2"][k] for i in idx]) for h in range(4)], 1)
            preds["personal_OLS_naive"] = (om, ov)
            for nm_, (m, v) in preds.items():
                s[f"held|k{k + 1}|{nm_}|mse"] = np.mean((m - tru) ** 2)
                s[f"held|k{k + 1}|{nm_}|cov95"] = np.mean(np.abs(m - tru) <= 1.96 * np.sqrt(v))
                # contrasts: m(.5,0)-m(-.5,0) and m(0,.5)-m(0,-.5)
                for cname, (hi, lo) in (("c1", (1, 0)), ("c2", (3, 2))):
                    s[f"held|k{k + 1}|{nm_}|contrast_{cname}_mse"] = np.mean(((m[:, hi] - m[:, lo]) - (tru[:, hi] - tru[:, lo])) ** 2)
        return s
    point = stats(np.arange(n))
    B = [stats(b) for b in boot_idx]
    keys = list(point)
    ci = {k: [float(np.nanpercentile([b[k] for b in B], 2.5)), float(np.nanpercentile([b[k] for b in B], 97.5))] for k in keys}
    return {k: {"est": float(point[k]), "ci": ci[k]} for k in keys}


def threshold_analysis(rng_seed=BSEED):
    rows = []
    for name, (h_app, h_truth) in T_PINS.items():
        d = TDIR / name
        if sha(d / "application.jsonl") != h_app or sha(d / "truth.json") != h_truth:
            raise RuntimeError(f"T artifact changed: {name}")
        ev = pd.DataFrame([json.loads(x) for x in (d / "application.jsonl").read_text().splitlines()])
        truth = {t["author_id"]: t for t in json.loads((d / "truth.json").read_text())}
        gaps = np.array([-2 / 3, 0., 2 / 3])
        for a, g in ev.groupby("author_id"):
            g = g.sort_values("t")

            def bracket(sub):
                best, bi = np.inf, None
                for i, gp in enumerate(gaps):
                    A = np.c_[np.ones(len(sub)), (sub.z.to_numpy() > gp).astype(float)]
                    coef, *_ = np.linalg.lstsq(A, sub.y.to_numpy(), rcond=None)
                    rss = ((sub.y.to_numpy() - A @ coef) ** 2).sum()
                    if rss < best:
                        best, bi = rss, i
                return bi
            tau = truth[a]["tau"]; b = truth[a]["b"]
            true_br = 0 if tau < -1 / 3 else (1 if tau < 1 / 3 else 2)
            rows.append({"world_seed": name, "author": a, "true_bracket": true_br, "abs_b": abs(b), "full": bracket(g), "h1": bracket(g[g.t < 16]), "h2": bracket(g[g.t >= 16])})
    D = pd.DataFrame(rows)

    def kappa(x, y):
        po = np.mean(x == y); pe = sum(np.mean(x == c) * np.mean(y == c) for c in range(3))
        return (po - pe) / (1 - pe) if pe < 1 else float("nan")
    rng = np.random.default_rng(rng_seed)
    draws = rng.integers(0, len(D), (BOOT, len(D)))
    agree = D.h1.to_numpy() == D.h2.to_numpy()
    res = {"authors": int(len(D)), "true_bracket_distribution": D.true_bracket.value_counts().sort_index().to_dict(),
           "half_agreement": float(agree.mean()), "half_agreement_ci": [float(v) for v in np.percentile([agree[b].mean() for b in draws], [2.5, 97.5])],
           "half_kappa": float(kappa(D.h1.to_numpy(), D.h2.to_numpy())),
           "half_kappa_ci": [float(v) for v in np.percentile([kappa(D.h1.to_numpy()[b], D.h2.to_numpy()[b]) for b in draws], [2.5, 97.5])],
           "full_recovery": float(np.mean(D.full == D.true_bracket)), "half_recovery": float(np.mean(np.r_[D.h1 == D.true_bracket, D.h2 == D.true_bracket]))}
    D["b_tercile"] = pd.qcut(D.abs_b, 3, labels=["low", "mid", "high"])
    res["by_abs_b_tercile"] = {str(k): {"half_agreement": float(np.mean(g.h1 == g.h2)), "full_recovery": float(np.mean(g.full == g.true_bracket)), "abs_b_mean": float(g.abs_b.mean())}
                               for k, g in D.groupby("b_tercile", observed=True)}
    res["note"] = "four observed z levels identify only which gap holds tau (3 brackets); continuous tau within a bracket is not identified; no point reliability"
    return res, D


def main():
    t0 = time.time()
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol changed")
    if OUT.exists():
        raise FileExistsError("reliability_spine_r1 exists; preserve earlier runs")
    OUT.mkdir(parents=True)
    total_rows = len(rsp.WORLDS) * len(SEEDS) * 420 * 48
    assert total_rows <= MAX_ROWS
    true_beta = rsp.BETA_TRUE
    res = {"protocol_sha256": PROTOCOL_SHA, "module_sha256": sha(ROOT / "grain_research/reliability_spine.py"), "rows_multichannel": total_rows,
           "orders": rsp.P_ORDERS.tolist(), "fits": {}, "known_model_reliability": {}, "mc_check_corr2": {}, "summary": {}, "neural_calls": 0,
           "data_provenance": "all spine data are NEW draws (generator seeds [18092, world, seed]); only the existing F04 T-world artifacts are read (item 6)"}
    person_rows = []
    for w in rsp.WORLDS:
        true_par = {"Sig": rsp.SIG, "sd2": rsp.WORLDS[w]["sd2"], "rho": rsp.RHO, "sw2": rsp.SW2, "se2": rsp.SE2}
        recs_all = []
        for sd in SEEDS:
            data = rsp.generate(w, sd)
            fit = fit_source(data)
            if not any(a.get("success") for a in fit["attempts"] if "success" in a):
                res["fits"][f"{w}|{sd}"] = {"attempts": fit["attempts"], "status": "BOTH_OPTIMIZERS_FAILED"}
            par = fit["par"]
            res["fits"][f"{w}|{sd}"] = {"attempts": fit["attempts"], "flags": fit["flags"], "seconds": fit["seconds"],
                                         "sd2_hat": par["sd2"], "sd2_se": fit["se"]["sd2"], "sd2_true": rsp.WORLDS[w]["sd2"],
                                         "rho_hat": par["rho"], "rho_se": fit["se"]["rho"], "sw2_hat": par["sw2"], "sw2_se": fit["se"]["sw2"], "se2_hat": par["se2"], "se2_se": fit["se"]["se2"],
                                         "Sig_hat_diag": np.diag(par["Sig"]).tolist(), "Sig_hat": par["Sig"].tolist(), "beta_hat": fit["beta"].tolist(),
                                         "g_hat": {"k1": [fit["beta"][6], fit["beta"][7], -fit["beta"][6] - fit["beta"][7]], "k2": [fit["beta"][8], fit["beta"][9], -fit["beta"][8] - fit["beta"][9]]}}
            recs = score(data, fit, true_par, true_beta)
            for r in recs:
                r["_mu_fit"] = fit["beta"][:6]; r["_Sig_fit"] = par["Sig"]; r["seed"] = sd
            recs_all += recs
            print(w, sd, "fit", round(fit["seconds"], 1), "s; sd2", round(par["sd2"], 4), "rho", round(par["rho"], 3), [a.get("method") + ":" + str(a.get("success")) for a in fit["attempts"] if "method" in a], flush=True)
        pairs_app = [r["pair"] for r in recs_all]
        res["known_model_reliability"][w] = known_reliability(true_par, pairs_app, rsp.WORLDS[w]["curved"])
        TH = np.array([r["theta"] for r in recs_all]); ORC = np.array([r["theta_orc"] for r in recs_all])
        res["mc_check_corr2"][w] = {nm: float(np.corrcoef(ORC[:, j], TH[:, j])[0, 1] ** 2) for j, nm in enumerate(NAMES)}
        rng = np.random.default_rng([BSEED, list(rsp.WORLDS).index(w)])
        boot_idx = [rng.integers(0, len(recs_all), len(recs_all)) for _ in range(BOOT)]
        res["summary"][w] = summarize(recs_all, boot_idx)
        for r in recs_all:
            row = {"world": w, "seed": r["seed"], "u": r["u"], "pair": str(r["pair"])}
            for j, nm in enumerate(NAMES):
                row[f"true_{nm}"] = r["theta"][j]; row[f"fit_{nm}"] = r["theta_fit"][j]; row[f"orc_{nm}"] = r["theta_orc"][j]; row[f"ols_{nm}"] = r["theta_ols"][j]
                for o in range(2):
                    for f in range(3):
                        row[f"half{o}{f}_{nm}"] = r[f"half_{o}{f}"][j]
            for k in range(2):
                for h in range(4):
                    row[f"m_true_k{k + 1}_h{h}"] = r["m_held"][k, h]
            for t in T_STATE:
                for k in range(2):
                    row[f"s_true_t{t}_k{k + 1}"] = r[f"s_true_{t}_{k}"]
                    for mode in ("smooth", "filter"):
                        for arm in ("fit", "orc"):
                            row[f"s_{mode}_{arm}_t{t}_k{k + 1}"], row[f"s_{mode}_{arm}_var_t{t}_k{k + 1}"] = r[f"s_{mode}_{arm}_{t}_{k}"]
            person_rows.append(row)
        print(w, "summarized", round(time.time() - t0, 1), "s", flush=True)
    thr, D = threshold_analysis()
    res["threshold_bracket"] = thr
    pd.DataFrame(person_rows).to_csv(OUT / "application_person_metrics.csv", index=False)
    D.to_csv(OUT / "threshold_bracket_rows.csv", index=False)
    res["seconds"] = time.time() - t0; res["created"] = datetime.now(timezone.utc).isoformat()
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"seconds": res["seconds"], "neural_calls": 0, "threads": 1, "code_sha256": sha(Path(__file__)),
                                                  "module_sha256": res["module_sha256"], "protocol_sha256": PROTOCOL_SHA,
                                                  "T_inputs": {k: {"application.jsonl": v[0], "truth.json": v[1]} for k, v in T_PINS.items()}}, indent=1) + "\n")
    print("done", round(res["seconds"], 1))


if __name__ == "__main__":
    main()
