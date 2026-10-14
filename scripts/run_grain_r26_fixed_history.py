#!/usr/bin/env python
"""R26 fixed-history inference + isolated reaction-cost control (docs/GRAIN_R26_FIXED_HISTORY_PROTOCOL_V1.md).
Imports the frozen packet19 functions unchanged. C1: 2,400 naive REACT histories rescored with the aware likelihood
(no planner calls). C2: 800 OC/aware/REACT runs with reaction-cost coefficient 0, paired to packet19 streams. CPU, 0 neural calls."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import hashlib
import importlib.util
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_R26_FIXED_HISTORY_PROTOCOL_V1.md"
PROTOCOL_SHA = "f5dfc5eeecefec3601318c9a9c6b3c81dab0a884845636a1f2d2870f8785bf70"
P19 = ROOT / "results/grain_gptpro_foundation_v3/F06/reactive_head_to_head_r1"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F06/fixed_history_inference_r1"
TOL = 1e-12


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol changed")
    for rel, h in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        if sha(ROOT / rel) != h:
            raise RuntimeError(f"binding mismatch {rel}")


def load_p19():
    spec = importlib.util.spec_from_file_location("r26_packet19", ROOT / "scripts/run_grain_r26_reactive_head_to_head.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


M = load_p19()


# ---------------------------------------------------------------- data-generating side (uses latents; never passed to inference)
def regenerate(tau, kap, nu, Urow, items):
    r, ys, rs, arts = 0.0, [], [], []
    for t, j in enumerate(items):
        c, a, art, n = M.ITEMS[j]
        ys.append(int(Urow[t] < expit(a * (tau + kap * c + r) + n * nu))); rs.append(r); arts.append(art)
        r = M.RHO_R * r + M.INC_R * art
    return ys, {"r_true_path": rs, "r_final": r, "r_sum": float(sum(rs)), "probe_count": int(sum(arts))}


# ---------------------------------------------------------------- inference side: items + responses only
def own_action_reaction(items):
    """Known-law reaction state before each turn, from the planner's own actions only."""
    r, out = 0.0, []
    for j in items:
        out.append(r); r = M.RHO_R * r + M.INC_R * M.ITEMS[j][2]
    return out, r


def rescore(items, ys, policy, aware):
    """Posterior summaries from own items and realized responses only (no latents, truth or planner)."""
    rs, r_final = own_action_reaction(items) if aware else ([0.0] * len(items), 0.0)
    w = M.scalar_prior() if policy == "S" else M.prior_op()
    for j, y, r in zip(items, ys, rs, strict=True):
        w = w * (M.scalar_lik(M.ITEMS[j], y, r) if policy == "S" else M.lik_op(M.ITEMS[j], y, r)); w = w / w.sum()
    if policy == "S":
        mt = float((w * M.TAU).sum()); lo1, hi1 = M.wquantile(M.TAU, w, (.025, .975))
        return {"est_T1": mt, "est_T2": mt + M.CSTAR, "est_CS": mt + (r_final if aware else 0.0), "lo_T1": lo1, "hi_T1": hi1, "lo_T2": np.nan, "hi_T2": np.nan}
    e1 = float((w * M.TV["T1"]).sum()); e2 = float((w * M.TV["T2"]).sum())
    lo1, hi1 = M.wquantile(M.TV["T1"].ravel(), w.ravel(), (.025, .975)); lo2, hi2 = M.wquantile(M.TV["T2"].ravel(), w.ravel(), (.025, .975))
    return {"est_T1": e1, "est_T2": e2, "est_CS": e1 + (r_final if aware else 0.0), "lo_T1": lo1, "hi_T1": hi1, "lo_T2": lo2, "hi_T2": hi2}


# ---------------------------------------------------------------- cost-0 wrapper (reaction-cost coefficient 0)
_ORIG_OP = M.op_choose


def op_choose_cost0(w, target, cost, r_plan, react_cost):
    return _ORIG_OP(w, target, cost, r_plan, False)


def utilities(w, target, r_plan, coef):
    tv = M.TV[target]; v0 = M.post_var(w, tv); out = []
    for item in M.ITEMS:
        l1 = M.lik_op(item, 1, r_plan); p1 = (w * l1).sum()
        gain = v0 - (p1 * M.post_var(w * l1 / p1, tv) + (1 - p1) * M.post_var(w * (1 - l1) / (1 - p1), tv))
        out.append(gain / (1.0 if target == "T1" else 1.0 + 0.25 * M.CSTAR ** 2) - 0.05 * item[2] - 0.05 * item[3] - coef * M.INC_R * item[2])
    return np.array(out)


def run_cost0(tau, kap, nu, Urow, target, world="REACT"):
    M.op_choose = op_choose_cost0
    try:
        return M.run_person(tau, kap, nu, Urow, "OC", "aware", world, target)
    finally:
        M.op_choose = _ORIG_OP


def near(a, b):
    if (a is None or (isinstance(a, float) and np.isnan(a))) and (b is None or (isinstance(b, float) and np.isnan(b))):
        return True
    return abs(float(a) - float(b)) <= TOL


FIELDS = ("est_T1", "est_T2", "est_CS", "lo_T1", "hi_T1", "lo_T2", "hi_T2")
RUNF = FIELDS + ("probe_count", "r_sum", "r_final", "cost_art", "cost_nuis", "cost_react")


def metrics(est, truth, lo=None, hi=None):
    e = np.asarray(est) - np.asarray(truth)
    out = {"bias": float(e.mean()), "mse": float((e ** 2).mean()), "rmse": float(np.sqrt((e ** 2).mean()))}
    if lo is not None and not np.isnan(np.asarray(lo, float)).any():
        out["coverage95"] = float(np.mean((np.asarray(lo) <= truth) & (truth <= np.asarray(hi))))
    else:
        out["coverage95"] = "NOT_SUPPORTED"
    return out


IDX = np.random.default_rng([19101, 3]).integers(0, M.N, (M.BOOT, M.N))


def paired(ea, eb):
    a, b = np.asarray(ea) ** 2, np.asarray(eb) ** 2
    d = np.sqrt(a[IDX].mean(1)) - np.sqrt(b[IDX].mean(1))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"point": float(np.sqrt(a.mean()) - np.sqrt(b.mean())), "ci": [float(lo), float(hi)], "verdict": "INCONCLUSIVE (CI contains 0)" if lo <= 0 <= hi else ("NEGATIVE (first lower RMSE)" if hi < 0 else "POSITIVE (first higher RMSE)")}


def main():
    t0 = time.time(); started = datetime.now(timezone.utc).isoformat()
    verify()
    if OUT.exists():
        raise FileExistsError("fixed_history_inference_r1 exists; preserve earlier runs")
    D = pd.read_csv(P19 / "per_person_runs.csv")
    L = np.load(P19 / "latents_and_stream.npz"); tau, kap, nu, U = L["tau"], L["kap"], L["nu"], L["U"]
    OUT.mkdir(parents=True)
    checks = {"c1_regeneration_failures": [], "c2_pairing_failures": [], "c2_zero_world_failures": [], "cost_zero_rule": {}}
    # ================= C1
    t1 = time.time(); c1 = []
    for policy in ("S", "O", "OC"):
        for target in ("T1", "T2"):
            G = D[(D.world == "REACT") & (D["mode"] == "naive") & (D.plan_target == target) & (D.policy == policy)].set_index("u").sort_index()
            A = D[(D.world == "REACT") & (D["mode"] == "aware") & (D.plan_target == target) & (D.policy == policy)].set_index("u").sort_index()
            for u in range(M.N):
                row = G.loc[u]; items = [int(x) for x in row["items"].split("|")]
                ys, gen = regenerate(tau[u], kap[u], nu[u], U[u], items)
                nv = rescore(items, ys, policy, aware=False)
                ok = all(near(nv[f], row[f]) for f in FIELDS) and gen["probe_count"] == row.probe_count and near(gen["r_final"], row.r_final) and near(gen["r_sum"], row.r_sum)
                if not ok:
                    checks["c1_regeneration_failures"].append({"policy": policy, "target": target, "u": u})
                    continue
                aw = rescore(items, ys, policy, aware=True)
                rec = {"policy": policy, "plan_target": target, "u": u, "items": row["items"], "responses": "".join(map(str, ys)), "probe_count": gen["probe_count"],
                       "probe_share": gen["probe_count"] / M.K, "r_final": gen["r_final"], "true_T1": row.true_T1, "true_T2": row.true_T2, "true_CS": row.true_CS,
                       "aware_online_true_CS": A.loc[u].true_CS, "aware_online_probe_count": int(A.loc[u].probe_count)}
                for f in FIELDS:
                    rec[f"naive_{f}"] = row[f]; rec[f"rescore_{f}"] = aw[f]; rec[f"online_aware_{f}"] = A.loc[u][f]
                c1.append(rec)
    C1 = pd.DataFrame(c1); C1.to_csv(OUT / "c1_per_history.csv", index=False)
    t_c1 = time.time() - t1
    s_dup = all(C1[(C1.policy == "S") & (C1.plan_target == "T1")]["items"].to_numpy() == C1[(C1.policy == "S") & (C1.plan_target == "T2")]["items"].to_numpy())
    c1_summ, c1_pair = {}, {}
    for (policy, target), g in C1.groupby(["policy", "plan_target"]):
        g = g.sort_values("u"); tr = g[f"true_{target}"].to_numpy(); other = "T2" if target == "T1" else "T1"
        blk = {}
        for arm in ("naive", "rescore", "online_aware"):
            blk[arm] = {"primary": metrics(g[f"{arm}_est_{target}"], tr, g[f"{arm}_lo_{target}"], g[f"{arm}_hi_{target}"]),
                        "secondary_" + other: metrics(g[f"{arm}_est_{other}"], g[f"true_{other}"].to_numpy()),
                        "CS": metrics(g[f"{arm}_est_CS"], (g["aware_online_true_CS"] if arm == "online_aware" else g["true_CS"]).to_numpy())}
        blk["probe_count_mean_naive_history"] = float(g.probe_count.mean()); blk["probe_share_mean_naive_history"] = float(g.probe_share.mean())
        blk["probe_count_mean_aware_online"] = float(g.aware_online_probe_count.mean()); blk["probe_share_mean_aware_online"] = float(g.aware_online_probe_count.mean() / M.K)
        c1_summ[f"{policy}|{target}"] = blk
        en = g[f"naive_est_{target}"] - tr; er = g[f"rescore_est_{target}"] - tr; ea = g[f"online_aware_est_{target}"] - tr
        c1_pair[f"{policy}|{target}"] = {"inference_effect rescore-naive": paired(er, en), "history_choice_effect online_aware-rescore": paired(ea, er),
                                         "regime_total online_aware-naive": paired(ea, en),
                                         "decomposition_point_identity_error": float(abs((paired(ea, er)["point"] + paired(er, en)["point"]) - paired(ea, en)["point"])),
                                         "CS_inference_effect rescore-naive": paired(g.rescore_est_CS - g.true_CS, g.naive_est_CS - g.true_CS)}
    # ================= C2
    t2 = time.time(); c2 = []
    for target in ("T1", "T2"):
        Old = D[(D.world == "REACT") & (D["mode"] == "aware") & (D.plan_target == target) & (D.policy == "OC")].set_index("u").sort_index()
        Z = D[(D.world == "ZERO") & (D["mode"] == "aware") & (D.plan_target == target) & (D.policy == "OC")].set_index("u").sort_index()
        for u in range(M.N):
            old_replay = M.run_person(tau[u], kap[u], nu[u], U[u], "OC", "aware", "REACT", target)
            o = Old.loc[u]
            if not (all(near(old_replay[f], o[f]) for f in RUNF) and old_replay["items"] == o["items"]):
                checks["c2_pairing_failures"].append({"target": target, "u": u})
            new = run_cost0(tau[u], kap[u], nu[u], U[u], target)
            c2.append({"plan_target": target, "u": u, **{f"cost10_{k}": o[k] for k in RUNF + ("items",)}, **{f"cost0_{k}": new[k] for k in RUNF + ("items",)},
                       "true_T1": o.true_T1, "true_T2": o.true_T2, "true_CS_cost10": o.true_CS, "true_CS_cost0": tau[u] + new["r_final"]})
            if u < 20:
                z = run_cost0(tau[u], kap[u], nu[u], U[u], target, world="ZERO"); zo = Z.loc[u]
                if not (all(near(z[f], zo[f]) for f in RUNF) and z["items"] == zo["items"]):
                    checks["c2_zero_world_failures"].append({"target": target, "u": u})
    C2 = pd.DataFrame(c2); C2.to_csv(OUT / "c2_per_person.csv", index=False)
    t_c2 = time.time() - t2
    # cost-zero rule at actual posterior states (first 20 persons, both targets)
    n_states, n_arg_ok, max_diff_rule = 0, 0, 0.0
    for rec in C2[C2.u < 20].itertuples():
        items = [int(x) for x in rec.cost0_items.split("|")]
        ys, _ = regenerate(tau[rec.u], kap[rec.u], nu[rec.u], U[rec.u], items)
        rs, _ = own_action_reaction(items)
        w = M.prior_op()
        for t, j in enumerate(items):
            u0 = utilities(w, rec.plan_target, rs[t], 0.0); u10 = utilities(w, rec.plan_target, rs[t], 0.10)
            arts = np.array([it[2] for it in M.ITEMS])
            max_diff_rule = max(max_diff_rule, float(np.max(np.abs((u0 - u10) - 0.10 * M.INC_R * arts))))
            orig_false = _ORIG_OP(w, rec.plan_target, True, rs[t], False)
            n_states += 1; n_arg_ok += int(int(np.argmax(u0)) == j == orig_false)
            w = w * M.lik_op(M.ITEMS[j], ys[t], rs[t]); w = w / w.sum()
    checks["cost_zero_rule"] = {"states": n_states, "argmax_equals_choice_equals_original_react_cost_false": n_arg_ok, "max_abs_utility_rule_error": max_diff_rule}
    c2_summ, c2_pair = {}, {}
    for target, g in C2.groupby("plan_target"):
        g = g.sort_values("u"); tr = g[f"true_{target}"].to_numpy(); other = "T2" if target == "T1" else "T1"
        blk = {}
        for arm in ("cost10", "cost0"):
            blk[arm] = {"primary": metrics(g[f"{arm}_est_{target}"], tr, g[f"{arm}_lo_{target}"], g[f"{arm}_hi_{target}"]),
                        "secondary_" + other: metrics(g[f"{arm}_est_{other}"], g[f"true_{other}"].to_numpy()),
                        "CS_policy_specific": metrics(g[f"{arm}_est_CS"], g[f"true_CS_{arm}"].to_numpy()),
                        "probe_count_mean": float(g[f"{arm}_probe_count"].mean()), "probe_share_mean": float(g[f"{arm}_probe_count"].mean() / M.K),
                        "cost_art_mean": float(g[f"{arm}_cost_art"].mean()), "cost_nuis_mean": float(g[f"{arm}_cost_nuis"].mean()),
                        "cost_react_accounting_mean": float(g[f"{arm}_cost_react"].mean()), "r_final_mean": float(g[f"{arm}_r_final"].mean())}
        blk["items_identical_share"] = float((g.cost0_items == g.cost10_items).mean())
        c2_summ[target] = blk
        c2_pair[target] = {"cost0-cost10 primary": paired(g[f"cost0_est_{target}"] - tr, g[f"cost10_est_{target}"] - tr),
                           "cost0-cost10 CS (policy-specific targets)": paired(g.cost0_est_CS - g.true_CS_cost0, g.cost10_est_CS - g.true_CS_cost10)}
    res = {"protocol_sha256": PROTOCOL_SHA, "neural_calls": 0, "c1_records": int(len(C1)), "c1_distinct_S_histories_duplicated_across_targets": bool(s_dup),
           "c2_new_policy_runs": int(len(C2)), "c2_pairing_replays": int(len(C2)), "c2_zero_world_replays": 40,
           "checks": {k: (v if not isinstance(v, list) else {"n_failures": len(v), "items": v[:20]}) for k, v in checks.items()},
           "c1_summary": c1_summ, "c1_paired": c1_pair, "c2_summary": c2_summ, "c2_paired": c2_pair,
           "labels": {"S": "scalar model misspecified in kappa/nu; est_T2 = mean(tau)+2; T2 interval NOT_SUPPORTED",
                      "rescore": "aware likelihood with own-action known-law reaction; same items and responses; no planner, latent or future access",
                      "cost0": "OC aware with .05 art/.05 nuisance and reaction-cost coefficient 0; isolates the policy-cost regime, not an individual causal state effect",
                      "CS": "tau + r_8 is policy/history-specific; compared only within the same history", "probe": "COUNT = number of art=1 items; SHARE = COUNT/8"},
           "timing_seconds": {"c1": t_c1, "c2": t_c2}, "seconds": time.time() - t0, "created": datetime.now(timezone.utc).isoformat()}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"started": started, "seconds": res["seconds"], "neural_calls": 0, "threads": 1, "code_sha256": sha(Path(__file__)),
                                                  "protocol_sha256": PROTOCOL_SHA, "new_policy_runs": int(len(C2)), "replays": int(len(C2)) + 40, "rescored_histories": int(len(C1))}, indent=1) + "\n")
    print("done", round(res["seconds"], 1), "c1", len(C1), "regen_fail", len(checks["c1_regeneration_failures"]), "pair_fail", len(checks["c2_pairing_failures"]),
          "zero_fail", len(checks["c2_zero_world_failures"]), "rule", checks["cost_zero_rule"])


if __name__ == "__main__":
    main()
