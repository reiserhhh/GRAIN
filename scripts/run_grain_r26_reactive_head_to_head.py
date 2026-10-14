#!/usr/bin/env python
"""R26 continuous-target head-to-head WITH reactivity (docs/GRAIN_R26_REACTIVE_HEAD_TO_HEAD_PROTOCOL_V1.md).
Reuses the F06/head_to_head_r1 bank, grids, priors and policies; adds a declared reaction state. CPU, 0 neural calls."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_R26_REACTIVE_HEAD_TO_HEAD_PROTOCOL_V1.md"
PROTOCOL_SHA = "c9ae7870e30d853019029c13b9e5c5c796fa601c5daabe39d5ec04134d522e6c"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F06/reactive_head_to_head_r1"
N, K, CSTAR, BOOT = 400, 8, 2.0, 2000
RHO_R, INC_R = .6, .20
CS = np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
ITEMS = np.array([(c, a, art, n) for c in CS for (a, art) in ((1.0, 0), (2.0, 1)) for n in (0.2, 1.0)])
TAU = np.linspace(-3, 3, 41); KAP = np.linspace(-1, 3, 21)
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(7); GH_W = GH_W / GH_W.sum()
FIXED = [int(np.flatnonzero((ITEMS[:, 0] == c) & (ITEMS[:, 1] == 1) & (ITEMS[:, 3] == 0.2))[0]) for c in (-1.5, 1.5, -0.5, 0.5, -1, 1, 0, 0)]
POLICIES, MODES, WORLDS, TARGETS = ("F", "S", "O", "OC"), ("naive", "aware"), ("ZERO", "REACT"), ("T1", "T2")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prior_op():
    pt = np.exp(-0.5 * TAU ** 2); pk = np.exp(-0.5 * ((KAP - 1) / 0.5) ** 2)
    w = pt[:, None, None] * pk[None, :, None] * GH_W[None, None, :]
    return w / w.sum()


def lik_op(item, y, r):
    c, a, _, n = item
    p = expit(a * (TAU[:, None, None] + KAP[None, :, None] * c + r) + n * GH_X[None, None, :])
    return p if y == 1 else 1 - p


def target_vals(target):
    t = TAU[:, None, None] + (KAP[None, :, None] * CSTAR if target == "T2" else 0 * KAP[None, :, None])
    return np.broadcast_to(t, (len(TAU), len(KAP), len(GH_X)))


TV = {t: target_vals(t) for t in TARGETS}


def post_var(w, tv):
    m = (w * tv).sum()
    return (w * (tv - m) ** 2).sum()


def op_choose(w, target, cost, r_plan, react_cost):
    tv = TV[target]; v0 = post_var(w, tv); best, arg = -np.inf, 0
    for j, item in enumerate(ITEMS):
        l1 = lik_op(item, 1, r_plan); p1 = (w * l1).sum()
        gain = v0 - (p1 * post_var(w * l1 / p1, tv) + (1 - p1) * post_var(w * (1 - l1) / (1 - p1), tv))
        u = gain / (1.0 if target == "T1" else 1.0 + 0.25 * CSTAR ** 2)
        if cost:
            u -= 0.05 * item[2] + 0.05 * item[3] + (0.10 * INC_R * item[2] if react_cost else 0.0)
        if u > best:
            best, arg = u, j
    return arg


def scalar_prior():
    w = np.exp(-0.5 * TAU ** 2)
    return w / w.sum()


def scalar_lik(item, y, r):
    c, a, _, _ = item
    p = expit(a * (TAU + c + r))
    return p if y == 1 else 1 - p


def scalar_choose(w, r_plan):
    m = (w * TAU).sum()
    info = [a * a * expit(a * (m + c + r_plan)) * (1 - expit(a * (m + c + r_plan))) for c, a, _, _ in ITEMS]
    return int(np.argmax(info))


def wquantile(vals, w, qs):
    o = np.argsort(vals); v = vals[o]; cw = np.cumsum(w[o]); cw = cw / cw[-1]
    return [float(v[np.searchsorted(cw, q)]) for q in qs]


def run_person(tau, kap, nu, U, policy, mode, world, target):
    """State clock: choose (own history only) -> respond with r_t -> update -> react."""
    w = scalar_prior() if policy == "S" else prior_op()
    r_true, r_sum, arts, nus, drs, used = 0.0, 0.0, [], [], [], []
    for t in range(K):
        r_plan = r_true if mode == "aware" else 0.0       # aware: deterministic from own actions via the declared law
        react_cost = (world == "REACT" and mode == "aware")
        if policy == "F":
            j = FIXED[t]
        elif policy == "S":
            j = scalar_choose(w, r_plan)
        else:
            j = op_choose(w, target, policy == "OC", r_plan, react_cost)
        c, a, art, n = ITEMS[j]
        y = int(U[t] < expit(a * (tau + kap * c + r_true) + n * nu))
        w = w * (scalar_lik(ITEMS[j], y, r_plan) if policy == "S" else lik_op(ITEMS[j], y, r_plan)); w = w / w.sum()
        r_sum += r_true; used.append(j); arts.append(art); nus.append(n)
        dr = INC_R * art if world == "REACT" else 0.0
        drs.append(dr)
        if world == "REACT":
            r_true = RHO_R * r_true + INC_R * art
    if policy == "S":
        mt = float((w * TAU).sum()); est1, est2 = mt, mt + CSTAR
        lo1, hi1 = wquantile(TAU, w, (.025, .975)); lo2 = hi2 = None
    else:
        est1 = float((w * TV["T1"]).sum()); est2 = float((w * TV["T2"]).sum()); mt = est1
        lo1, hi1 = wquantile(TV["T1"].ravel(), w.ravel(), (.025, .975)); lo2, hi2 = wquantile(TV["T2"].ravel(), w.ravel(), (.025, .975))
    cs_pred = mt + (r_true if mode == "aware" else 0.0)
    return {"est_T1": est1, "est_T2": est2, "est_CS": cs_pred, "lo_T1": lo1, "hi_T1": hi1, "lo_T2": lo2, "hi_T2": hi2,
            "probe_count": int(sum(arts)), "r_sum": r_sum, "r_final": r_true, "cost_art": .05 * sum(arts), "cost_nuis": .05 * sum(nus),
            "cost_react": .10 * sum(drs), "items": "|".join(map(str, used))}


def main():
    t0 = time.time(); started = datetime.now(timezone.utc).isoformat()
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol changed")
    if OUT.exists():
        raise FileExistsError("reactive_head_to_head_r1 exists; preserve earlier runs")
    OUT.mkdir(parents=True)
    lat = np.random.default_rng([19101, 1])
    tau = lat.standard_normal(N); kap = 1 + 0.5 * lat.standard_normal(N); nu = lat.standard_normal(N)
    U = np.random.default_rng([19101, 2]).random((N, K))
    np.savez(OUT / "latents_and_stream.npz", tau=tau, kap=kap, nu=nu, U=U)
    rows = []
    for world in WORLDS:
        for mode in MODES:
            for target in TARGETS:
                for policy in POLICIES:
                    for u in range(N):
                        r = run_person(tau[u], kap[u], nu[u], U[u], policy, mode, world, target)
                        rows.append({"world": world, "mode": mode, "plan_target": target, "policy": policy, "u": u, **r})
                    print(world, mode, target, policy, round(time.time() - t0, 1), flush=True)
    D = pd.DataFrame(rows)
    D["true_T1"] = np.tile(tau, len(D) // N); D["true_T2"] = np.tile(tau + CSTAR * kap, len(D) // N)
    D["true_CS"] = np.tile(tau, len(D) // N) + D.r_final
    D.to_csv(OUT / "per_person_runs.csv", index=False)
    # ZERO-world identity naive vs aware
    z = D[D.world == "ZERO"]
    zn = z[z["mode"] == "naive"].reset_index(drop=True); za = z[z["mode"] == "aware"].reset_index(drop=True)
    identity = bool(all(np.array_equal(zn[c].to_numpy(), za[c].to_numpy()) for c in ("est_T1", "est_T2", "est_CS", "items")))

    def sel(world, mode, target, policy):
        return D[(D.world == world) & (D["mode"] == mode) & (D.plan_target == target) & (D.policy == policy)].sort_values("u")
    summ = {}
    for world in WORLDS:
        for mode in MODES:
            for target in TARGETS:
                for policy in POLICIES:
                    g = sel(world, mode, target, policy)
                    e = g[f"est_{target}"] - g[f"true_{target}"]; ecs = g.est_CS - g.true_CS
                    covs = None
                    if not (policy == "S" and target == "T2"):
                        covs = float(np.mean((g[f"lo_{target}"] <= g[f"true_{target}"]) & (g[f"true_{target}"] <= g[f"hi_{target}"])))
                    summ[f"{world}|{mode}|{target}|{policy}"] = {"bias": float(e.mean()), "mse": float((e ** 2).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                                                                 "coverage95": covs if covs is not None else "NOT_SUPPORTED",
                                                                 "cs_bias": float(ecs.mean()), "cs_rmse": float(np.sqrt((ecs ** 2).mean())),
                                                                 "probe_share": float(g.probe_count.mean() / K), "r_final_mean": float(g.r_final.mean()), "r_sum_mean": float(g.r_sum.mean()),
                                                                 "cost_art": float(g.cost_art.mean()), "cost_nuis": float(g.cost_nuis.mean()), "cost_react": float(g.cost_react.mean())}
    idx = np.random.default_rng([19101, 3]).integers(0, N, (BOOT, N))

    def paired(ga, gb, col, truth):
        a = (ga[col].to_numpy() - ga[truth].to_numpy()) ** 2; b = (gb[col].to_numpy() - gb[truth].to_numpy()) ** 2
        d = np.sqrt(a[idx].mean(1)) - np.sqrt(b[idx].mean(1))
        return {"point": float(np.sqrt(a.mean()) - np.sqrt(b.mean())), "ci": [float(x) for x in np.percentile(d, [2.5, 97.5])]}
    comps = {}
    for target in TARGETS:
        for policy in POLICIES:
            comps[f"REACT aware-naive|{target}|{policy}"] = paired(sel("REACT", "aware", target, policy), sel("REACT", "naive", target, policy), f"est_{target}", f"true_{target}")
            comps[f"REACT aware-naive CS|{target}|{policy}"] = paired(sel("REACT", "aware", target, policy), sel("REACT", "naive", target, policy), "est_CS", "true_CS")
            for mode in MODES:
                comps[f"REACT-ZERO|{mode}|{target}|{policy}"] = paired(sel("REACT", mode, target, policy), sel("ZERO", mode, target, policy), f"est_{target}", f"true_{target}")
        for mode in MODES:
            for world in WORLDS:
                for pa, pb in (("O", "F"), ("OC", "F"), ("OC", "O"), ("O", "S")):
                    comps[f"{world}|{mode}|{target}|{pa}-{pb}"] = paired(sel(world, mode, target, pa), sel(world, mode, target, pb), f"est_{target}", f"true_{target}")
    res = {"protocol_sha256": PROTOCOL_SHA, "neural_calls": 0, "persons": N, "budget": K, "zero_world_naive_aware_identical": identity,
           "summary": summ, "paired_rmse_deltas": comps, "seconds": time.time() - t0, "created": datetime.now(timezone.utc).isoformat(),
           "notes": {"T2": "c*=2 extrapolates beyond the bank range [-1.5,1.5]", "S": "misspecified in kappa/nu; S-vs-O is compound model/design",
                     "aware": "best-case known-law reference; no access to tau/kappa/nu", "CS": "current-state target tau + r_8, separate from trait"}}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"started": started, "seconds": res["seconds"], "neural_calls": 0, "threads": 1, "code_sha256": sha(Path(__file__)),
                                                  "protocol_sha256": PROTOCOL_SHA, "runs": len(D)}, indent=1) + "\n")
    print("done", round(res["seconds"], 1), "identity", identity)


if __name__ == "__main__":
    main()
