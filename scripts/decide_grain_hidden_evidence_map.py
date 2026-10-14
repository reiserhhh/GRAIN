#!/usr/bin/env python3
"""Apply the frozen GRAIN-C1-01 acceptance rule to the evaluator outputs."""

import json
import sys
from pathlib import Path

import pandas as pd

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/grain_construction_v1/C1/hidden_map_r1")
tol = pd.DataFrame(json.loads((out / "frozen_tolerances.json").read_text()))
rec = pd.read_csv(out / "recovery_by_mechanism.csv")
iv = pd.read_csv(out / "intervention_checks.csv")
PRIMARY = {"W1_shared_level": "level", "W2_conditional": "delta", "W3_missing_anchor": "delta",
           "W4_confounded_anchor": "delta"}
rows = []
for (world, seed, arm), t in tol.groupby(["world", "seed", "arm"]):
    t = t.iloc[0]
    for target in (["level"] if world == "W1_shared_level" else ["delta", "level"]):
        r = rec[(rec.world == world) & (rec.seed == seed) & (rec.arm == arm) & (rec.target == target)
                & (rec.material == "same_family")]
        g = iv[(iv.world == world) & (iv.seed == seed) & (iv.arm == arm) & (iv.target == target)]
        gain = g[g.statistic == "gain_T"]
        fp = g[g.intervention == "fingerprint"]
        st = g[g.intervention == "style"]
        flags = t["flags"] or []
        nonconst_min = (t["nonconstant_min"] or {}).get(target)
        ok = {
            "status_ok": t["status"] == "OK",
            "design_ok": "CONFOUNDED_PILOT_DESIGN" not in flags,
            "nonconstant": bool(len(r) and nonconst_min is not None and r.sd_score.iloc[0] >= nonconst_min),
            "unit_within_tau_T": bool(len(gain) and t["tau_T"] is not None
                                      and abs(gain.value.iloc[0] - 1) <= t["tau_T"]),
            "gain_ci_above_0": bool(len(gain) and gain.ci_lo.iloc[0] > 0),
            "fingerprint_R_F_ok": bool(len(fp) and fp.value.iloc[0] <= 0.20),
            "style_R_F_ok": bool(len(st) and st.value.iloc[0] <= 0.20),
        }
        rows.append({"world": world, "seed": seed, "arm": arm, "target": target,
                     "primary": PRIMARY[world] == target, "status": t["status"], "flags": ";".join(flags),
                     "tau_T": t["tau_T"],
                     "gain_T": gain.value.iloc[0] if len(gain) else None,
                     "R_F_fingerprint": fp.value.iloc[0] if len(fp) else None,
                     "R_F_style": st.value.iloc[0] if len(st) else None,
                     "r": r.r.iloc[0] if len(r) else None,
                     **ok, "pass": all(ok.values())})
tab = pd.DataFrame(rows)
tab.to_csv(out / "acceptance_by_seed.csv", index=False)
world_pass = (tab.groupby(["world", "arm", "target", "primary"])["pass"].agg(["sum", "count"]).reset_index())
world_pass["passes_world"] = world_pass["sum"] == 3
world_pass.to_csv(out / "acceptance_by_world.csv", index=False)
prim = world_pass[world_pass.primary]
w1 = set(prim[(prim.world == "W1_shared_level") & prim.passes_world].arm)
w2 = set(prim[(prim.world == "W2_conditional") & prim.passes_world].arm)
decision = {"W1_passing_arms": sorted(w1), "W2_passing_arms": sorted(w2),
            "same_arm_both": sorted(w1 & w2),
            "positive_acceptance": bool(w1) and bool(w2),
            "rule": "frozen protocol V1: pass per seed = status OK, no confounded design, nonconstant, "
                    "|gain_T-1|<=tau_T, gain CI>0, R_F fingerprint and style <= .20; world = 3/3 seeds"}
(out / "acceptance.json").write_text(json.dumps(decision, indent=1))
print(json.dumps(decision, indent=1))
print(world_pass.to_string())
