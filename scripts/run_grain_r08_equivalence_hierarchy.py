#!/usr/bin/env python
"""R08 amendment companion (packet 14 Part A): symmetry ⇒ lumpability ⇒ predictive equivalence, with the converses
refuted by two exact finite counterexamples, and the equal-prior Bayes-error identity. It reuses the frozen r1 runner's
exact functions read-only and does not edit r1."""
from __future__ import annotations

import json
import sys
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_grain_r08_text_behaviour_quotient as r1  # noqa: E402  (frozen r1 runner, imported only)

OUT = ROOT / "results/grain_gptpro_foundation_v3/F15/equivalence_hierarchy_r1"


def q(x):
    return F(str(x))


def four_state():
    """Supervisor counterexample: states (x, r), x fixed; r-kernel [[.9,.1],[.4,.6]]; emission depends on x only."""
    states = [(x, r) for x in (0, 1) for r in (0, 1)]
    obs = [(0,), (1,)]
    R = {0: [q(.9), q(.1)], 1: [q(.4), q(.6)]}
    K0 = []
    for (x, r) in states:
        p1 = q(.2) + q(.6) * x
        K0.append([[(p1 if o == 1 else 1 - p1) * (R[r][r2] if x2 == x else F(0)) for (x2, r2) in states] for (o,) in obs])
    return {"name": "four_state", "states": states, "labels": [f"x{x}r{r}" for x, r in states], "obs": obs, "K": [K0], "views": {"raw": lambda o: o}}


def trace_not_bisim():
    """s ~ t in every future law, yet {s,t} is not lumpable: s -> ua|ub (.5/.5), t -> w; w emits a|b (.5/.5)."""
    labels = ["s", "t", "ua", "ub", "w", "e"]
    obs = [("x",), ("a",), ("b",), ("e",)]
    K0 = [[[F(0)] * 6 for _ in obs] for _ in labels]
    i = {l: n for n, l in enumerate(labels)}
    o = {v[0]: n for n, v in enumerate(obs)}
    K0[i["s"]][o["x"]][i["ua"]] = q(.5); K0[i["s"]][o["x"]][i["ub"]] = q(.5)
    K0[i["t"]][o["x"]][i["w"]] = F(1)
    K0[i["ua"]][o["a"]][i["e"]] = F(1)
    K0[i["ub"]][o["b"]][i["e"]] = F(1)
    K0[i["w"]][o["a"]][i["e"]] = q(.5); K0[i["w"]][o["b"]][i["e"]] = q(.5)
    K0[i["e"]][o["e"]][i["e"]] = F(1)
    return {"name": "trace_not_bisim", "states": labels, "labels": labels, "obs": obs, "K": [K0], "views": {"raw": lambda v: v}}


def transition_automorphism_error(w, perm):
    """max |T(g s' | g s) - T(s' | s)| on the transition kernel marginalized over observations."""
    S = len(w["states"])
    T = [[sum(w["K"][0][s][oi][s2] for oi in range(len(w["obs"]))) for s2 in range(S)] for s in range(S)]
    return max(abs(T[perm[s]][perm[s2]] - T[s][s2]) for s in range(S) for s2 in range(S))


def automorphism_error(w, perm):
    err = F(0)
    for s in range(len(w["states"])):
        for oi in range(len(w["obs"])):
            for s2 in range(len(w["states"])):
                err = max(err, abs(w["K"][0][perm[s]][oi][perm[s2]] - w["K"][0][s][oi][s2]))
    return err


def main() -> int:
    if OUT.exists():
        raise FileExistsError("equivalence_hierarchy_r1 exists; preserve earlier runs")
    OUT.mkdir(parents=True)
    res = {}
    w = four_state()
    for a in (0,):
        for s, row in enumerate(w["K"][a]):
            assert sum(v for o in row for v in o) == 1
    xblocks = [[0, 1], [2, 3]]
    perm = {0: 1, 1: 0, 2: 3, 3: 2}                                   # g: r -> 1 - r
    pred, dim = r1.krylov_classes_exact(w, "raw", (0,))
    res["four_state"] = {
        "x_block_lumpability": r1.lumpability(w, "raw", (0,), xblocks),
        "r_flip_automorphism_error": float(transition_automorphism_error(w, perm)),
        "r_flip_joint_kernel_error": float(automorphism_error(w, perm)),
        "predictive_classes": r1.labels(w, pred), "future_span_dim": dim,
        "coarsest_bisimulation": r1.labels(w, r1.bisimulation(w, "raw", (0,))),
        "x_blocks_have_distinct_laws": float(r1.tv(r1.future_law(w, "raw", 0, (0,)), r1.future_law(w, "raw", 2, (0,)))),
        "reading": "the x-block quotient is exact (lumpable, predictive) although the r-flip is not a symmetry (error .3): symmetry is sufficient, not necessary"}
    t = trace_not_bisim()
    for s, row in enumerate(t["K"][0]):
        assert sum(v for o in row for v in o) == 1
    pred_t, dim_t = r1.krylov_classes_exact(t, "raw", (0,))
    part = [c for c in pred_t]
    tv_st = max(float(r1.tv(r1.future_law(t, "raw", 0, (0,) * k), r1.future_law(t, "raw", 1, (0,) * k))) for k in range(1, 6))
    res["trace_not_bisim"] = {
        "predictive_classes": r1.labels(t, pred_t), "future_span_dim": dim_t,
        "max_future_TV_s_vs_t_k1to5": tv_st,
        "lumpability_of_predictive_partition": r1.lumpability(t, "raw", (0,), part),
        "coarsest_bisimulation": r1.labels(t, r1.bisimulation(t, "raw", (0,))),
        "reading": "s and t have identical futures (predictively equivalent) but the predictive partition is not lumpable; bisimulation separates them. A predictive-state representation exists without a Markov quotient on that partition"}
    # equal-prior binary Bayes error identity: min error = (1 - TV)/2
    import pandas as pd
    ptv = pd.read_csv(ROOT / "results/grain_gptpro_foundation_v3/F15/text_behaviour_quotient_r1/pair_tv.csv")
    w6 = ptv[(ptv.world == "W6") & (ptv.view == "B") & (ptv.policy == "passive") & (ptv.k == 1)].TV.iloc[0]
    p, qd = 0.2, 0.5                                                     # W6 B laws at k = 1 for x0h0r0 / x1h0r0 (from kernel)
    bayes = 0.5 * min(p, qd) + 0.5 * min(1 - p, 1 - qd)
    res["bayes_error_identity"] = {"W6_B_TV_k1_archived": float(w6), "direct_bayes_error_single_observation": bayes,
                                   "identity_(1-TV)/2": (1 - float(w6)) / 2, "match": abs(bayes - (1 - float(w6)) / 2) < 1e-12,
                                   "W4_anchor_information_nats_per_observation": 0.8317766166719343,
                                   "reading": "distinct laws mean distributional identifiability; a single noisy observation still misclassifies with probability (1-TV)/2 (.35 here); .832 nats is information per observation, not perfect identification"}
    ok = (res["four_state"]["x_block_lumpability"]["lumpable"] and abs(res["four_state"]["r_flip_automorphism_error"] - 0.3) < 1e-15
          and res["four_state"]["x_blocks_have_distinct_laws"] > 0 and tv_st == 0 and not res["trace_not_bisim"]["lumpability_of_predictive_partition"]["lumpable"]
          and res["bayes_error_identity"]["match"])
    res["status"] = "PASS" if ok else "FAIL"
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({"status": res["status"], "four_state": {k: res["four_state"][k] for k in ("r_flip_automorphism_error", "predictive_classes", "coarsest_bisimulation")},
                      "trace": {k: res["trace_not_bisim"][k] for k in ("predictive_classes", "coarsest_bisimulation", "max_future_TV_s_vs_t_k1to5")},
                      "lump_violation": res["trace_not_bisim"]["lumpability_of_predictive_partition"]["max_violation"], "bayes": res["bayes_error_identity"]["direct_bayes_error_single_observation"]}, indent=1))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
