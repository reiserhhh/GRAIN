#!/usr/bin/env python
"""R26/A07 committed two-observation design comparison (docs/GRAIN_R26_COMMITTED_DESIGN_PROTOCOL_V1.md). CPU only,
0 neural calls. Resumable: rerun the same command; per-author gz checkpoints."""
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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import f06_committed_design as cd  # noqa: E402
from grain_research import f06_recovery_design as p25  # noqa: E402

PROTOCOL = ROOT / "docs/GRAIN_R26_COMMITTED_DESIGN_PROTOCOL_V1.md"
PROTOCOL_SHA = "557ce351f63db0928f0ee9443ed6710719d227f34f22ea4157bd6981209df436"
V3 = ROOT / "results/grain_gptpro_foundation_v3"
ARCH = V3 / "F06/gaussian_full_r1/authors"
R1 = V3 / "F06/recovery_target_design_r1"
OUT = V3 / "F06/recovery_committed_design_r1"
SEEDS, N_APP, WORLDS, BUDGETS = (20260915, 20260916, 20260917), 200, ("zero", "distributed"), (8, 16, 24)
CAP_S = 25 * 60
ZFIX = [-1.0 if t % 2 == 0 else 1.0 for t in range(24)]


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


def plan_mismatches(out):
    bad = 0
    for p in out["plans"]:
        t = p["t"]
        for k, (h, zn) in enumerate(p["committed"]):
            bad += int(out["h"][t + k] != h or out["z"][t + k + 1] != zn)
    return bad


def run_author(seed, k, src):
    s = p25.author_streams(seed, k)
    rec = {"seed": seed, "ordinal": k, "author_id": s["author_id"], "arms": {}, "addon": {}}
    for world in WORLDS:
        for arm in cd.ARMS:
            out = cd.run_arm(arm, world, s, src, BUDGETS)
            rec["arms"][f"{world}|{arm}"] = {"metrics": out["metrics"], "h": out["h"], "z": out["z"], "quota_ok": p25.quota_ok(out["h"]),
                                             "plan_mismatches": plan_mismatches(out), "plans": out["plans"], "y": out["y"]}
    yfix = rec["arms"]["zero|fixed"]["y"]
    for arm in ("scalar_a", "recovery"):
        sched = rec["arms"][f"zero|{arm}"]["h"]
        out = cd.run_arm("fixed", "zero", s, src, BUDGETS, fixed_schedule=(ZFIX, sched), record_plans=False)
        rec["addon"][arm] = {"metrics": out["metrics"], "y_identical_to_fixed": bool(np.array_equal(np.array(out["y"]), np.array(yfix))), "h": out["h"]}
    return rec


def boot(df, col, a, b, auths, draws, armcol="arm"):
    pa = df[df[armcol] == a].set_index("key")[col].reindex(auths).to_numpy(); pb = df[df[armcol] == b].set_index("key")[col].reindex(auths).to_numpy()
    d = pa - pb; bs = d[draws].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
    ra, rb = pa[draws].mean(1), pb[draws].mean(1); rd_ = np.sqrt(ra) - np.sqrt(rb); rlo, rhi = np.percentile(rd_, [2.5, 97.5])
    return {"delta_MSE": float(d.mean()), "ci_MSE": [float(lo), float(hi)], "delta_RMSE": float(np.sqrt(pa.mean()) - np.sqrt(pb.mean())), "ci_RMSE": [float(rlo), float(rhi)],
            "ci_contains_0": bool(lo <= 0 <= hi)}


def main():
    t0 = time.time(); started = datetime.now(timezone.utc).isoformat()
    verify()
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "private_authors").mkdir(exist_ok=True)
    srcs = {s: source(s) for s in SEEDS}
    r1 = pd.read_csv(R1 / "private_budget_metrics.csv")
    r1f = r1[r1.arm == "fixed"].set_index(["key", "world", "budget"])
    res = {"protocol_sha256": PROTOCOL_SHA, "neural_calls": 0, "authors_planned": len(SEEDS) * N_APP, "gates": {}}
    mism = 0
    for seed in SEEDS:
        for k in range(N_APP):
            s = p25.author_streams(seed, k); a = json.load(gzip.open(ARCH / f"{seed}_application_{k:04d}.json.gz")); tr = a["truth"]
            ok = a["author_id"] == s["author_id"] and all(tr[x] == s[x] for x in ("a", "b", "rho", "beta", "initial")) and \
                np.array_equal(np.array(tr["innovations"]), s["innovations"]) and np.array_equal(np.array(tr["measurement"]), s["measurement"])
            mism += int(not ok)
    res["gates"]["stream_mismatches"] = mism
    if mism:
        (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n"); raise RuntimeError("stream gate failed")
    order = [(s, k) for s in SEEDS for k in range(4)] + [(s, k) for s in SEEDS for k in range(4, N_APP)]
    tstart = time.time(); done = 0; stopped = False
    for idx, (seed, k) in enumerate(order):
        path = OUT / "private_authors" / f"{seed}_{k:04d}.json.gz"
        if not path.exists():
            if done >= 12 and (time.time() - tstart) * (1 + 1 / max(done, 1)) > CAP_S - (tstart - t0):
                stopped = True; break
            rec = run_author(seed, k, srcs[seed]); rec["hashes"] = {"protocol": PROTOCOL_SHA, "module": sha(ROOT / "grain_research/f06_committed_design.py"), "runner": sha(Path(__file__))}
            key = f"{seed}_{k:04d}"
            for world in WORLDS:                            # early fixed replay gate on every author as produced
                for m in rec["arms"][f"{world}|fixed"]["metrics"]:
                    ref = r1f.loc[(key, world, m["budget"])]
                    if max(abs(m["rho_hat"] - ref.rho_hat), abs(m["a_err"] - ref.a_err), abs(m["state_err"] - ref.state_err)) > 1e-12:
                        raise RuntimeError(f"fixed replay gate failed {key}")
            with gzip.open(path, "wt") as fh:
                json.dump(rec, fh)
        done += 1
        if idx == 11:
            res["pilot"] = {"authors": 12, "seconds": time.time() - tstart, "projected_total_seconds": (time.time() - tstart) / 12 * len(order)}
    files = [f for f in sorted((OUT / "private_authors").glob("*.json.gz")) if not f.name.startswith("._")]
    rows, addrows, ident, quota_bad, plan_bad, fixmax, yid, seqs = [], [], [], 0, 0, 0.0, [], {}
    for f in files:
        rec = json.load(gzip.open(f)); key = f"{rec['seed']}_{rec['ordinal']:04d}"
        for world in WORLDS:
            sa, rc = rec["arms"][f"{world}|scalar_a"], rec["arms"][f"{world}|recovery"]
            ident.append(sa["h"] == rc["h"] and sa["z"] == rc["z"])
            for arm in cd.ARMS:
                A = rec["arms"][f"{world}|{arm}"]; quota_bad += int(not A["quota_ok"]); plan_bad += A["plan_mismatches"]
                seqs.setdefault(f"{world}|{arm}", Counter())["".join("+" if z > 0 else "-" for z in A["z"]) + "|" + "".join(str(int(h)) for h in A["h"])] += 1
                pos = [t for t, hh in enumerate(A["h"]) if hh]
                for m in A["metrics"]:
                    rows.append({"seed": rec["seed"], "key": key, **m, "action_positions": "|".join(map(str, pos))})
                    if arm == "fixed":
                        ref = r1f.loc[(key, world, m["budget"])]; fixmax = max(fixmax, abs(m["rho_hat"] - ref.rho_hat), abs(m["a_err"] - ref.a_err))
        for arm, ad in rec["addon"].items():
            yid.append(ad["y_identical_to_fixed"])
            for m in ad["metrics"]:
                addrows.append({"seed": rec["seed"], "key": key, "schedule": f"{arm}_schedule_on_fixed_z", **{k2: m[k2] for k2 in ("budget", "rho_err", "rho_cover", "rho_width", "a_err", "beta_err")}})
    M = pd.DataFrame(rows)
    for c in ("rho", "a", "b", "state", "beta"):
        M[f"{c}_se"] = M[f"{c}_err"] ** 2
    M.to_csv(OUT / "private_budget_metrics.csv", index=False)
    A_ = pd.DataFrame(addrows); A_["rho_se"] = A_.rho_err ** 2; A_.to_csv(OUT / "private_addon_beta0_metrics.csv", index=False)
    res["gates"]["fixed_vs_packet25_max_abs"] = fixmax
    res["authors_completed"] = len(files); res["status"] = "COMPLETE" if len(files) == res["authors_planned"] else "PARTIAL"; res["stopped_at_cap"] = stopped
    res["integrity"] = {"quota_violations": quota_bad, "scored_vs_executed_mismatches": plan_bad, "identical_policy_rate_scalar_vs_recovery": float(np.mean(ident)),
                        "fit_failures_total": int(M.groupby(["key", "world", "arm"]).fit_failures.max().sum()), "nonfinite_total": int(M.groupby(["key", "world", "arm"]).nonfinite.max().sum())}
    r1pos = r1[r1.budget == 24].groupby(["world", "arm"]).positive_condition_share.mean().to_dict()
    summ, pairs = {}, {}
    for (world, budget), g in M.groupby(["world", "budget"]):
        auths = np.array(sorted(g.key.unique())); draws = np.random.default_rng(20260928).integers(0, len(auths), (2000, len(auths)))
        blk = {}
        for arm in cd.ARMS:
            ga = g[g.arm == arm]
            blk[arm] = {"rho_bias": float(ga.rho_err.mean()), "rho_RMSE": float(np.sqrt(ga.rho_se.mean())), "rho_coverage": float(ga.rho_cover.mean()), "rho_width": float(ga.rho_width.mean()),
                        "rho_pred_var_mean": float(ga.rho_var_delta.mean()), "rho_MSE": float(ga.rho_se.mean()), "a_RMSE": float(np.sqrt(ga.a_se.mean())), "b_RMSE": float(np.sqrt(ga.b_se.mean())),
                        "state_RMSE": float(np.sqrt(ga.state_se.mean())), "beta_RMSE": float(np.sqrt(ga.beta_se.mean())), "effective_actions_mean": float(ga.effective_actions.mean()),
                        "positive_condition_share": float(ga.positive_condition_share.mean()), "z_rank_deficient_share": float((ga.z_design_rank < 2).mean()),
                        "info_condition_median": float(ga.info_condition.median()), "sum_sq_perturbation_mean": float(ga.sum_sq_perturbation.mean()),
                        "top_action_positions": ga.action_positions.value_counts().head(3).to_dict(), "rho_RMSE_by_seed": {int(s): float(np.sqrt(gs.rho_se.mean())) for s, gs in ga.groupby("seed")}}
        summ[f"{world}|{budget}"] = blk
        pairs[f"{world}|{budget}"] = {"recovery-fixed": boot(g, "rho_se", "recovery", "fixed", auths, draws), "recovery-scalar_a": boot(g, "rho_se", "recovery", "scalar_a", auths, draws),
                                      "scalar_a-fixed": boot(g, "rho_se", "scalar_a", "fixed", auths, draws), "level_a: scalar_a-fixed": boot(g, "a_se", "scalar_a", "fixed", auths, draws),
                                      "level_a: recovery-scalar_a": boot(g, "a_se", "recovery", "scalar_a", auths, draws)}
    res["summary"] = summ; res["paired_bootstrap"] = pairs
    res["design_diagnostics"] = {"modal_sequence_share": {k: float(c.most_common(1)[0][1] / sum(c.values())) for k, c in seqs.items()},
                                 "distinct_sequences": {k: len(c) for k, c in seqs.items()},
                                 "packet25_positive_share_budget24": {f"{w}|{a}": float(v) for (w, a), v in r1pos.items()}}
    add = {"y_identical_all": bool(all(yid)), "y_identical_count": int(sum(yid)), "n": len(yid), "by_budget": {}}
    Mz = M[M.world == "zero"]
    for budget in BUDGETS:
        gz = Mz[Mz.budget == budget]; ga = A_[A_.budget == budget]
        auths = np.array(sorted(gz.key.unique())); draws = np.random.default_rng(20260929).integers(0, len(auths), (2000, len(auths)))
        comb = pd.concat([gz[["key", "arm", "rho_se"]], ga.rename(columns={"schedule": "arm"})[["key", "arm", "rho_se"]]])
        add["by_budget"][str(budget)] = {"rho_RMSE": {a: float(np.sqrt(comb[comb.arm == a].rho_se.mean())) for a in comb.arm.unique()},
                                         "scalar_schedule_on_fixed_z - fixed (estimator attribution, identical y)": boot(comb, "rho_se", "scalar_a_schedule_on_fixed_z", "fixed", auths, draws),
                                         "recovery_schedule_on_fixed_z - fixed (estimator attribution, identical y)": boot(comb, "rho_se", "recovery_schedule_on_fixed_z", "fixed", auths, draws),
                                         "recovery - recovery_schedule_on_fixed_z (conditional z-design contrast)": boot(comb, "rho_se", "recovery", "recovery_schedule_on_fixed_z", auths, draws),
                                         "scalar_a - scalar_a_schedule_on_fixed_z (conditional z-design contrast)": boot(comb, "rho_se", "scalar_a", "scalar_a_schedule_on_fixed_z", auths, draws)}
    res["addon_beta0"] = add
    res["seconds"] = time.time() - t0; res["created"] = datetime.now(timezone.utc).isoformat(); res["resume_command"] = f"{sys.executable} scripts/run_grain_r26_committed_design.py"
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "runtime.json").write_text(json.dumps({"started": started, "seconds": res["seconds"], "neural_calls": 0, "threads": 1, "code_sha256": sha(Path(__file__)),
                                                  "module_sha256": sha(ROOT / "grain_research/f06_committed_design.py"), "protocol_sha256": PROTOCOL_SHA, "authors_completed": len(files)}, indent=1) + "\n")
    print("done", round(res["seconds"], 1), res["status"], len(files), res.get("pilot"), res["integrity"], res["gates"])


if __name__ == "__main__":
    main()
