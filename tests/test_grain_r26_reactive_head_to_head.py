"""R26 reactive head-to-head: update ordering, shared streams, zero-reaction identity, no truth leakage, no future access,
deterministic replay, independent metric recomputation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_grain_r26_reactive_head_to_head as rr  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/F06/reactive_head_to_head_r1"
RES = json.loads((OUT / "results.json").read_text())
D = pd.read_csv(OUT / "per_person_runs.csv", float_precision="round_trip")
L = np.load(OUT / "latents_and_stream.npz")


def test_reaction_law_and_update_ordering_from_items():
    g = D[(D.world == "REACT") & (D["mode"] == "naive") & (D.policy == "O")].head(20)
    for r in g.itertuples():
        items = [int(x) for x in r.items.split("|")]
        rt, rs = 0.0, 0.0
        for j in items:
            rs += rt; rt = rr.RHO_R * rt + rr.INC_R * rr.ITEMS[j, 2]
        assert abs(rt - r.r_final) < 1e-12 and abs(rs - r.r_sum) < 1e-12
    z = D[D.world == "ZERO"]
    assert (z.r_final == 0).all() and (z.cost_react == 0).all()


def test_zero_world_naive_aware_identical():
    assert RES["zero_world_naive_aware_identical"] is True


def test_no_truth_leakage_into_planner():
    U = np.zeros(rr.K)       # forces y=1 at every turn for any latent
    for policy in ("S", "O", "OC"):
        for mode, world in (("naive", "REACT"), ("aware", "REACT")):
            a = rr.run_person(-1.2, .4, 1.5, U, policy, mode, world, "T1")["items"]
            b = rr.run_person(2.1, 1.9, -0.7, U, policy, mode, world, "T1")["items"]
            assert a == b


def test_no_future_response_access():
    U1 = np.zeros(rr.K); U2 = np.zeros(rr.K); U2[5:] = 0.999999
    for policy in ("S", "O", "OC"):
        a = rr.run_person(0.3, 1.0, 0.0, U1, policy, "aware", "REACT", "T2")["items"].split("|")
        b = rr.run_person(0.3, 1.0, 0.0, U2, policy, "aware", "REACT", "T2")["items"].split("|")
        assert a[:6] == b[:6]          # choice at turn 5 precedes the response at turn 5


def test_deterministic_replay_and_shared_stream():
    for world, mode, target, policy, u in (("REACT", "aware", "T1", "OC", 3), ("REACT", "naive", "T2", "S", 17), ("ZERO", "naive", "T1", "F", 250)):
        r = rr.run_person(L["tau"][u], L["kap"][u], L["nu"][u], L["U"][u], policy, mode, world, target)
        row = D[(D.world == world) & (D["mode"] == mode) & (D.plan_target == target) & (D.policy == policy) & (D.u == u)].iloc[0]
        assert abs(r["est_T1"] - row.est_T1) < 1e-12 and r["items"] == row["items"]


def test_independent_metric_recomputation():
    for key in ("REACT|naive|T1|O", "REACT|aware|T2|OC", "ZERO|naive|T1|S"):
        world, mode, target, policy = key.split("|")
        g = D[(D.world == world) & (D["mode"] == mode) & (D.plan_target == target) & (D.policy == policy)]
        e = g[f"est_{target}"] - g[f"true_{target}"]
        assert abs(np.sqrt((e ** 2).mean()) - RES["summary"][key]["rmse"]) < 1e-12
        assert np.allclose(g.true_T1.to_numpy(), L["tau"][g.u.to_numpy()])
