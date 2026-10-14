#!/usr/bin/env python
"""Post-hoc descriptive (not in protocol V1; changes no result): largest history collision among application prefixes
of the gate-passing arms, regenerating the identical application sequences from the frozen seeds and refitting the
frozen learner exactly as the runner does. Reports oracle TV and each model's prediction gap for the top pairs."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_grain_r08_controlled_psr_witness as w  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/F15/controlled_psr_witness_r1"
R = json.loads((OUT / "results.json").read_text())
res = {}
for arm in R["arms"]:
    if not arm["support_gate_pass"] or arm["src_policy"] != "random":
        continue
    K, init = w.construct_kernel(arm["world"])
    rng = np.random.default_rng(arm["seed"])
    As, Os = w.generate(K, init, w.N_SRC, "random", rng)
    w.generate(K, init, w.N_VAL, "random", rng)
    Aa, Oa = w.generate(K, init, w.N_APP, "random", rng)
    P = w.fit_spectral_stats(As, Os)
    rows, cols, aos = w.restrict_support(P, [0, 1])
    m = w.spectral_model(P, arm["rank"], rows, cols, aos)
    popP = w.pop_given_action(As, Os)
    run, opred = w.oracle_filter(K, init)
    groups = defaultdict(list)
    for i in range(w.N_APP):
        b = m["b1"].copy()
        for t in range(4):
            b, _ = w.psr_update(m, b, Aa[i, t], Oa[i, t])
        a4 = Aa[i, 4]
        key = (tuple(np.bincount(4 * Aa[i, :4] + Oa[i, :4], minlength=8)), int(Oa[i, 3]), int(a4))
        cnt = np.bincount(Oa[i, :4][Aa[i, :4] == a4], minlength=4)
        groups[key].append({"i": i, "hist": [(int(Aa[i, t]), int(Oa[i, t])) for t in range(4)], "oracle": opred(run(Aa[i], Oa[i], 4), a4),
                            "psr": w.psr_cond(m, b, a4)[0], "static": (cnt + arm["alpha_static"] * popP[a4]) / (cnt.sum() + arm["alpha_static"])})
    best = []
    for key, g in groups.items():
        seen = {}
        for r in g:
            seen.setdefault(tuple(r["hist"]), r)
        uniq = list(seen.values())
        for x in range(len(uniq)):
            for y in range(x + 1, len(uniq)):
                tv = .5 * np.abs(uniq[x]["oracle"] - uniq[y]["oracle"]).sum()
                best.append((tv, uniq[x], uniq[y]))
    best.sort(key=lambda z: -z[0])
    top = best[:3]
    res[arm["arm"]] = {"distinct_order_pairs_with_same_composition_last_action": len(best),
                       "max_oracle_TV": float(top[0][0]) if top else None,
                       "examples": [{"oracle_TV": float(tv), "hist_a": a["hist"], "hist_b": b["hist"],
                                     "oracle_a": a["oracle"].round(4).tolist(), "oracle_b": b["oracle"].round(4).tolist(),
                                     "psr_a": a["psr"].round(4).tolist(), "psr_b": b["psr"].round(4).tolist(), "psr_TV": float(.5 * np.abs(a["psr"] - b["psr"]).sum()),
                                     "static_TV": float(.5 * np.abs(a["static"] - b["static"]).sum())} for tv, a, b in top]}
(OUT / "collision_examples_posthoc.json").write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: {"pairs": v["distinct_order_pairs_with_same_composition_last_action"], "max_oracle_TV": v["max_oracle_TV"],
                      "top": [(round(e["oracle_TV"], 4), round(e["psr_TV"], 4), round(e["static_TV"], 4)) for e in v["examples"]]} for k, v in res.items()}, indent=1))
