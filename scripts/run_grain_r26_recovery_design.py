#!/usr/bin/env python
"""R26/A07 recovery-parameter design comparison (docs/GRAIN_R26_RECOVERY_DESIGN_PROTOCOL_V1.md). CPU only, 0 neural calls.
Resumable: per-author gz checkpoints; rerun the same command to continue at complete author boundaries."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import gzip
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
from grain_research import f06_recovery_design as rd  # noqa: E402

PROTOCOL = ROOT / "docs/GRAIN_R26_RECOVERY_DESIGN_PROTOCOL_V1.md"
PROTOCOL_SHA = "ee1b811b97a41f63feb4a16ad89dc555a63e6ac191631b8e368bc6fd8384f870"
V3 = ROOT / "results/grain_gptpro_foundation_v3"
ARCH = V3 / "F06/gaussian_full_r1/authors"
OUT = V3 / "F06/recovery_target_design_r1"
SEEDS, N_APP, WORLDS, BUDGETS = (20260915, 20260916, 20260917), 200, ("zero", "distributed"), (8, 16, 24)
CAP_S = 25 * 60


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol changed")
    for rel, hh in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        if sha(ROOT / rel) != hh:
            raise RuntimeError(f"binding mismatch {rel}")


def source(seed):
    return json.loads((V3 / f"F05/source_full_r2/core_{seed}/source_model.json").read_text())


def run_author(seed, k, src):
    s = rd.author_streams(seed, k)
    rec = {"seed": seed, "ordinal": k, "author_id": s["author_id"], "arms": {}}
    for world in WORLDS:
        for arm in rd.ARMS:
            out = rd.run_arm(arm, world, s, src, BUDGETS)
            rec["arms"][f"{world}|{arm}"] = {"metrics": out["metrics"], "h": out["h"], "z": out["z"], "quota_ok": rd.quota_ok(out["h"]),
                                             "steps": [{k2: v for k2, v in st.items() if k2 != "info"} | {"scores": (st["info"] or {}).get("scores")} for st in out["steps"]]}
    return rec


def boot(df, col, a, b, auths, draws):
    pa = df[df.arm == a].set_index("key")[col].reindex(auths).to_numpy(); pb = df[df.arm == b].set_index("key")[col].reindex(auths).to_numpy()
    d = pa - pb; bs = d[draws].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
    ra, rb = pa[draws].mean(1), pb[draws].mean(1)
    rmse_d = np.sqrt(np.maximum(ra, 0)) - np.sqrt(np.maximum(rb, 0)); rlo, rhi = np.percentile(rmse_d, [2.5, 97.5])
    return {"delta_MSE": float(d.mean()), "ci_MSE": [float(lo), float(hi)], "delta_RMSE": float(np.sqrt(pa.mean()) - np.sqrt(pb.mean())), "ci_RMSE": [float(rlo), float(rhi)],
            "ci_contains_0": bool(lo <= 0 <= hi)}


def main():
    t0 = time.time(); started = datetime.now(timezone.utc).isoformat()
    verify()
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "private_authors").mkdir(exist_ok=True)
    srcs = {s: source(s) for s in SEEDS}
    res = {"protocol_sha256": PROTOCOL_SHA, "neural_calls": 0, "authors_planned": len(SEEDS) * N_APP, "gates": {}, "integrity": {}}
    # ---- gates
    mism = 0
    for seed in SEEDS:
        for k in range(N_APP):
            s = rd.author_streams(seed, k)
            a = json.load(gzip.open(ARCH / f"{seed}_application_{k:04d}.json.gz"))
            tr = a["truth"]
            ok = a["author_id"] == s["author_id"] and all(tr[x] == s[x] for x in ("a", "b", "rho", "beta", "initial")) and \
                np.array_equal(np.array(tr["innovations"]), s["innovations"]) and np.array_equal(np.array(tr["measurement"]), s["measurement"])
            mism += int(not ok)
    res["gates"]["stream_mismatches"] = mism
    rep = 0.0
    for k in range(4):
        s = rd.author_streams(SEEDS[0], k); a = json.load(gzip.open(ARCH / f"{SEEDS[0]}_application_{k:04d}.json.gz"))
        world_map = {"zero": "zero", "distributed": "distributed"}
        for world in WORLDS:
            out = rd.run_arm("archived_fixed_replay", world, s, srcs[SEEDS[0]], BUDGETS)
            for m in out["metrics"]:
                arch = [x for x in a["metrics"] if x["arm"] == "fixed" and x["reactivity"] == world_map[world] and x["target"] == "a" and x["budget"] == m["budget"]][0]
                rep = max(rep, abs(m["rho_hat"] - arch["rho_estimate"]), abs((m["a_err"] + s["a"]) - arch["estimate"]))
    res["gates"]["archived_fixed_replay_max_abs"] = rep
    if mism or rep > 1e-10:
        (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
        raise RuntimeError(f"gate failed {res['gates']}")
    # ---- run: pilot (ordinals 0-3 of each seed, part of the plan) then the fixed seed-major order
    order = [(s, k) for s in SEEDS for k in range(4)] + [(s, k) for s in SEEDS for k in range(4, N_APP)]
    tstart = time.time(); done = 0; pilot_s = None; stopped = False
    for idx, (seed, k) in enumerate(order):
        path = OUT / "private_authors" / f"{seed}_{k:04d}.json.gz"
        if not path.exists():
            if pilot_s is not None and (time.time() - tstart) + (time.time() - tstart) / max(done, 1) > CAP_S - (tstart - t0):
                stopped = True; break
            rec = run_author(seed, k, srcs[seed]); rec["hashes"] = {"protocol": PROTOCOL_SHA, "module": sha(ROOT / "grain_research/f06_recovery_design.py"), "runner": sha(Path(__file__))}
            with gzip.open(path, "wt") as fh:
                json.dump(rec, fh)
        done += 1
        if idx == 11:
            pilot_s = time.time() - tstart
            res["pilot"] = {"authors": 12, "seconds": pilot_s, "projected_total_seconds": pilot_s / 12 * len(order)}
    files = sorted((OUT / "private_authors").glob("*.json.gz"))
    files = [f for f in files if not f.name.startswith("._")]
    rows, ident, quota_bad, steps_rows = [], [], 0, []
    for f in files:
        rec = json.load(gzip.open(f))
        for world in WORLDS:
            sa, rc = rec["arms"][f"{world}|scalar_a"], rec["arms"][f"{world}|recovery"]
            ident.append(sa["h"] == rc["h"] and sa["z"] == rc["z"])
            for arm in rd.ARMS:
                A = rec["arms"][f"{world}|{arm}"]; quota_bad += int(not A["quota_ok"])
                pos = [t for t, hh in enumerate(A["h"]) if hh]
                for m in A["metrics"]:
                    rows.append({"seed": rec["seed"], "ordinal": rec["ordinal"], "key": f"{rec['seed']}_{rec['ordinal']:04d}", **m, "action_positions": "|".join(map(str, pos))})
    M = pd.DataFrame(rows)
    M["rho_se"] = M.rho_err ** 2; M["a_se"] = M.a_err ** 2; M["b_se"] = M.b_err ** 2; M["state_se"] = M.state_err ** 2; M["beta_se"] = M.beta_err ** 2
    M.to_csv(OUT / "private_budget_metrics.csv", index=False)
    res["authors_completed"] = int(len(files)); res["status"] = "COMPLETE" if len(files) == res["authors_planned"] else "PARTIAL"
    res["stopped_at_cap"] = stopped
    res["integrity"] = {"quota_violations": int(quota_bad), "identical_policy_rate_scalar_vs_recovery": float(np.mean(ident)) if ident else None,
                        "fit_failures_total": int(M.groupby(["key", "world", "arm"]).fit_failures.max().sum()), "nonfinite_total": int(M.groupby(["key", "world", "arm"]).nonfinite.max().sum())}
    summ, pairs = {}, {}
    for (world, budget), g in M.groupby(["world", "budget"]):
        auths = np.array(sorted(g.key.unique())); draws = np.random.default_rng(20260928).integers(0, len(auths), (2000, len(auths)))
        blk = {}
        for arm in rd.ARMS:
            ga = g[g.arm == arm]
            blk[arm] = {"rho_bias": float(ga.rho_err.mean()), "rho_RMSE": float(np.sqrt(ga.rho_se.mean())), "rho_coverage": float(ga.rho_cover.mean()), "rho_width": float(ga.rho_width.mean()),
                        "rho_pred_var_mean": float(ga.rho_var_delta.mean()), "a_RMSE": float(np.sqrt(ga.a_se.mean())), "b_RMSE": float(np.sqrt(ga.b_se.mean())),
                        "state_RMSE": float(np.sqrt(ga.state_se.mean())), "beta_RMSE": float(np.sqrt(ga.beta_se.mean())), "effective_actions_mean": float(ga.effective_actions.mean()),
                        "positive_condition_share": float(ga.positive_condition_share.mean()), "sum_sq_perturbation_mean": float(ga.sum_sq_perturbation.mean()),
                        "top_action_positions": ga.action_positions.value_counts().head(3).to_dict()}
            per_seed = {int(s): float(np.sqrt(gs.rho_se.mean())) for s, gs in ga.groupby("seed")}
            blk[arm]["rho_RMSE_by_seed"] = per_seed
        summ[f"{world}|{budget}"] = blk
        pairs[f"{world}|{budget}"] = {"recovery-fixed": boot(g, "rho_se", "recovery", "fixed", auths, draws), "recovery-scalar_a": boot(g, "rho_se", "recovery", "scalar_a", auths, draws),
                                      "scalar_a-fixed": boot(g, "rho_se", "scalar_a", "fixed", auths, draws),
                                      "level_a: recovery-scalar_a": boot(g, "a_se", "recovery", "scalar_a", auths, draws), "state: recovery-fixed": boot(g, "state_se", "recovery", "fixed", auths, draws)}
    res["summary"] = summ; res["paired_bootstrap"] = pairs
    res["seconds"] = time.time() - t0; res["created"] = datetime.now(timezone.utc).isoformat()
    res["resume_command"] = f"{sys.executable} scripts/run_grain_r26_recovery_design.py"
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"started": started, "seconds": res["seconds"], "neural_calls": 0, "threads": 1, "code_sha256": sha(Path(__file__)),
                                                  "module_sha256": sha(ROOT / "grain_research/f06_recovery_design.py"), "protocol_sha256": PROTOCOL_SHA, "authors_completed": res["authors_completed"]}, indent=1) + "\n")
    print("done", round(res["seconds"], 1), res["status"], res["authors_completed"], res.get("pilot"), res["integrity"])


if __name__ == "__main__":
    main()
