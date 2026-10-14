#!/usr/bin/env python
"""Packet 34 runner (docs/GRAIN_R26_ALIRT_REFERENCE_PROTOCOL_V1.md): paper-equation ALIRT vs source-chosen FIXED order vs
OPERATOR-aware policy, matched items/observations/budgets, common and own scorers. 0 neural calls."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import alirt_reference as A  # noqa: E402

PROTOCOL = ROOT / "docs/GRAIN_R26_ALIRT_REFERENCE_PROTOCOL_V1.md"
PROTOCOL_SHA = "be9ec20329477f442c1167ab59703677fd8f5355e6baf907b80bbd5336efc8cd"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F06/alirt_reference_r1"
J, NP, NT, NA, SEEDS, BUDGETS, ZSTAR, BOOT = 16, 400, 400, 400, (0, 1, 2), (3, 6), 2.5, 2000
Z = np.linspace(-1.5, 1.5, J)
D = np.array([0.1, -0.3, 0.4, -0.1, 0.0, 0.3, -0.4, 0.2, -0.2, 0.5, -0.5, 0.1, 0.3, -0.3, 0.0, 0.2])
WF = np.array([1.0, 0.6, -0.4, 0.2])
WORLDS = {"W_pos": 0.0, "W_cond": 0.5}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol changed")
    for rel, hh in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        if sha(ROOT / rel) != hh:
            raise RuntimeError(f"binding mismatch {rel}")


def draws(seed, role, N):
    rng = np.random.default_rng(np.random.SeedSequence([20260927, 34, seed, role]))
    return {"a": rng.standard_normal(N), "zb": rng.standard_normal(N), "eps": rng.standard_normal((N, J)), "eta": rng.standard_normal((N, J, 4)), "yn": rng.standard_normal(N)}


def generate(B, sd_b):
    """SIMULATOR ONLY."""
    b = sd_b * B["zb"]
    e = D[None, :] + B["a"][:, None] + b[:, None] * Z[None, :] + 0.5 * B["eps"]
    X = e[:, :, None] * WF[None, None, :] + 0.5 * B["eta"]
    return X, B["a"] + 0.2 * B["yn"], {"tau": B["a"], "R": B["a"] + b * ZSTAR}


def main():
    t0 = time.time(); started = datetime.now(timezone.utc).isoformat()
    verify()
    if OUT.exists():
        raise FileExistsError("alirt_reference_r1 exists")
    OUT.mkdir(parents=True)
    fitlog, rows, con, info_rows, item_params = [], [], [], [], {}
    checks = {"no_repeated_items": True, "scored_equals_executed": True, "all_grm_fits_converged": True, "prob_sum_max_err": 0.0}
    rec_check = {}
    indep = 0.0
    for wname, sd_b in WORLDS.items():
        for seed in SEEDS:
            Xp, Yp, _ = generate(draws(seed, 1, NP), sd_b); Xt, _, _ = generate(draws(seed, 2, NT), sd_b); Xa, _, truth = generate(draws(seed, 3, NA), sd_b)
            pm = A.polytomize_fit(Xp, Yp)                                  # D_poly only
            Ot = A.categorize(pm, A.predict_items(pm, Xt)); Yhat_a = A.predict_items(pm, Xa); Oa = A.categorize(pm, Yhat_a)
            g = A.fit_grm(Ot)                                               # D_tr only
            checks["all_grm_fits_converged"] &= g["converged"]
            checks["prob_sum_max_err"] = max(checks["prob_sum_max_err"], float(np.abs(A.cat_probs(np.linspace(-5, 5, 41), g["a"], g["b"]).sum(2) - 1).max()))
            fitlog.append({"world": wname, "seed": seed, "converged": g["converged"], "nit": g["nit"], "nll": g["nll"], "message": g["message"],
                           "cat_freq_tr": json.dumps(np.bincount(Ot.ravel(), minlength=4).tolist())})
            item_params[f"{wname}_s{seed}_a"] = g["a"]; item_params[f"{wname}_s{seed}_b"] = g["b"]
            if seed == 0:   # parametric refit recovery check of the estimator
                rr = np.random.default_rng(np.random.SeedSequence([20260927, 34, 99, int(sd_b * 10)])); thr = rr.standard_normal(NT)
                P = A.cat_probs(thr, g["a"], g["b"]); U = rr.random((NT, J, 1)); Osim = (U > np.cumsum(P, 2)).sum(2)
                g2 = A.fit_grm(Osim)
                rec_check[wname] = {"corr_a": float(np.corrcoef(g2["a"], g["a"])[0, 1]), "corr_b": float(np.corrcoef(g2["b"].ravel(), g["b"].ravel())[0, 1]), "converged": g2["converged"]}
            Op = A.categorize(pm, A.predict_items(pm, Xp))
            Lpoly = np.array([A.map_theta(list(range(J)), Op[u].tolist(), g["a"], g["b"]) for u in range(NP)])
            bridgeL = np.polyfit(Lpoly, Yp, 1)
            op = A.fit_operator(Xt, Xp, Yp, Z)
            mp, cp = A.posterior(op, Xp @ op["w"], list(range(J))); Rpoly = A.operator_targets(op, mp, cp, ZSTAR)["newcond"]
            bridgeR = np.polyfit(Lpoly, Rpoly, 1)                           # declared adaptation: scalar ALIRT -> new-condition target
            order = list(np.argsort(-A.item_info(0.0, g["a"], g["b"])))
            Sa = Xa @ op["w"]
            Lfull = np.array([A.map_theta(list(range(J)), Oa[u].tolist(), g["a"], g["b"]) for u in range(NA)])
            rows.append({"world": wname, "seed": seed, "budget": J, "arm": "FULL_BANK", "scorer": "ALIRT_L", "target": "scalar",
                         "rmse": float(np.sqrt(np.mean((np.polyval(bridgeL, Lfull) - truth["tau"]) ** 2))), "bias": float(np.mean(np.polyval(bridgeL, Lfull) - truth["tau"])),
                         "sq_corr": float(np.corrcoef(Lfull, truth["tau"])[0, 1] ** 2), "coverage95": None})
            for B in BUDGETS:
                hist = {"ALIRT": [], "FIXED": [list(map(int, order[:B]))] * NA, "OPERATOR": [A.operator_select(op, B, ZSTAR)] * NA}
                for u in range(NA):
                    it, th = A.alirt_cat(Oa[u], g["a"], g["b"], B); hist["ALIRT"].append(it)
                    if abs(A.map_theta(it, Oa[u, it].tolist(), g["a"], g["b"]) - th) > 1e-9:
                        checks["scored_equals_executed"] = False
                sqerr = {}
                for arm, H in hist.items():
                    if any(len(set(h)) != len(h) for h in H):
                        checks["no_repeated_items"] = False
                    L = np.array([A.map_theta(h, Oa[u, h].tolist(), g["a"], g["b"]) for u, h in enumerate(H)])
                    Yb = np.array([Yhat_a[u, h].mean() for u, h in enumerate(H)])
                    opm = np.zeros(NA); opR = np.zeros(NA); sdS = np.zeros(NA); sdR = np.zeros(NA)
                    for key in {tuple(h) for h in H}:
                        idx = [u for u, h in enumerate(H) if tuple(h) == key]
                        mm, cc = A.posterior(op, Sa[idx], list(key)); tt = A.operator_targets(op, mm, cc, ZSTAR)
                        opm[idx], opR[idx], sdS[idx], sdR[idx] = tt["scalar"], tt["newcond"], tt["sd_scalar"], tt["sd_newcond"]
                    ests = {("ALIRT_L", "scalar"): (np.polyval(bridgeL, L), None), ("ALIRT_L", "newcond"): (np.polyval(bridgeR, L), None), ("ALIRT_Yb", "scalar"): (Yb, None),
                            ("OPERATOR_post", "scalar"): (opm, sdS), ("OPERATOR_post", "newcond"): (opR, sdR)}
                    for (sc, tg), (est, sd) in ests.items():
                        tr = truth["tau"] if tg == "scalar" else truth["R"]; err = est - tr
                        sqerr[(arm, sc, tg)] = err ** 2
                        rows.append({"world": wname, "seed": seed, "budget": B, "arm": arm, "scorer": sc, "target": tg, "rmse": float(np.sqrt(np.mean(err ** 2))), "bias": float(err.mean()),
                                     "sq_corr": float(np.corrcoef(est, tr)[0, 1] ** 2), "coverage95": float((np.abs(err) <= 1.96 * sd).mean()) if sd is not None else None})
                    info_rows.append({"world": wname, "seed": seed, "budget": B, "arm": arm, "mean_grm_info_at_final_theta": float(np.mean([A.item_info(L[u], g["a"], g["b"])[h].sum() for u, h in enumerate(H)])),
                                      "operator_post_sd_newcond_Yunits": float(sdR.mean()), "items_distinct_histories": len({tuple(h) for h in H}), "burden_items": B})
                if wname == "W_cond" and seed == 0 and B == 3:
                    e = np.sqrt(sqerr[("ALIRT", "ALIRT_L", "scalar")]); indep = abs(float(np.sqrt(np.mean(e ** 2))) - [r for r in rows if r["world"] == wname and r["seed"] == 0 and r["budget"] == 3 and r["arm"] == "ALIRT" and r["scorer"] == "ALIRT_L" and r["target"] == "scalar"][0]["rmse"])
                idxb = np.random.default_rng(np.random.SeedSequence([20260927, 34, seed, 9, B, int(sd_b * 10)])).integers(0, NA, (BOOT, NA))
                for (a1, a0) in (("ALIRT", "FIXED"), ("OPERATOR", "ALIRT"), ("OPERATOR", "FIXED")):
                    for (sc, tg) in (("ALIRT_L", "scalar"), ("ALIRT_L", "newcond"), ("ALIRT_Yb", "scalar"), ("OPERATOR_post", "scalar"), ("OPERATOR_post", "newcond")):
                        d = sqerr[(a1, sc, tg)] - sqerr[(a0, sc, tg)]; bs = d[idxb].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
                        con.append({"world": wname, "seed": seed, "budget": B, "contrast": f"{a1}-{a0}", "scorer": sc, "target": tg, "point_mse_diff": float(d.mean()),
                                    "ci_lo": float(lo), "ci_hi": float(hi), "ci_contains_0": bool(lo <= 0 <= hi)})
    checks["independent_rmse_recompute_max_abs"] = indep; checks["positive_control_refit"] = rec_check
    pd.DataFrame(rows).to_csv(OUT / "target_metrics.csv", index=False); pd.DataFrame(con).to_csv(OUT / "paired_contrasts.csv", index=False)
    pd.DataFrame(fitlog).to_csv(OUT / "grm_fit_log.csv", index=False); pd.DataFrame(info_rows).to_csv(OUT / "information_burden.csv", index=False)
    np.savez(OUT / "item_parameters.npz", **item_params)
    pd.DataFrame([{"role": r, "stream": f"SeedSequence([20260927,34,seed,{k}])", "authors": n, "observed": obs} for r, k, n, obs in
                  (("D_poly (mapping + thresholds + bridges)", 1, NP, "features x, measure Y"), ("D_tr (GRM calibration, operator model)", 2, NT, "features x only"),
                   ("application", 3, NA, "features x only; truth audit-only"))]).to_csv(OUT / "role_manifest.csv", index=False)
    res = {"protocol_sha256": PROTOCOL_SHA, "neural_calls": 0, "implementation": "paper-equation SciPy implementation; NOT authors' code (repo 404; mirt/mirtCAT absent)",
           "checks": checks, "seconds": time.time() - t0, "created": datetime.now(timezone.utc).isoformat()}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"started": started, "seconds": res["seconds"], "neural_calls": 0, "threads": 1, "code_sha256": sha(Path(__file__)),
                                                  "module_sha256": sha(ROOT / "grain_research/alirt_reference.py"), "protocol_sha256": PROTOCOL_SHA}, indent=1) + "\n")
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
