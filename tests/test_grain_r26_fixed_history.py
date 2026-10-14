"""Focused tests for packet20 C (R26 fixed-history inference and isolated cost control)."""
import hashlib
import importlib.util
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/grain_gptpro_foundation_v3/F06/fixed_history_inference_r1"
P19 = ROOT / "results/grain_gptpro_foundation_v3/F06/reactive_head_to_head_r1"
PROTOCOL = ROOT / "docs/GRAIN_R26_FIXED_HISTORY_PROTOCOL_V1.md"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def runner():
    spec = importlib.util.spec_from_file_location("r26_fixed", ROOT / "scripts/run_grain_r26_fixed_history.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_protocol_bindings_and_frozen_packet19_code():
    assert sha(PROTOCOL) == "f5dfc5eeecefec3601318c9a9c6b3c81dab0a884845636a1f2d2870f8785bf70"
    for rel, h in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        assert sha(ROOT / rel) == h
    assert json.loads((P19 / "runtime.json").read_text())["code_sha256"] == sha(ROOT / "scripts/run_grain_r26_reactive_head_to_head.py")


def test_historical_response_replay_exact():
    res = json.loads((OUT / "results.json").read_text())
    assert res["c1_records"] == 2400 and res["checks"]["c1_regeneration_failures"]["n_failures"] == 0
    m = runner(); D = pd.read_csv(P19 / "per_person_runs.csv"); L = np.load(P19 / "latents_and_stream.npz")
    for policy, u in (("O", 3), ("OC", 17), ("S", 250)):
        row = D[(D.world == "REACT") & (D["mode"] == "naive") & (D.plan_target == "T1") & (D.policy == policy) & (D.u == u)].iloc[0]
        items = [int(x) for x in row["items"].split("|")]
        ys, gen = m.regenerate(L["tau"][u], L["kap"][u], L["nu"][u], L["U"][u], items)
        nv = m.rescore(items, ys, policy, aware=False)
        assert abs(nv["est_T1"] - row.est_T1) <= 1e-12 and abs(gen["r_final"] - row.r_final) <= 1e-12


def test_histories_unchanged_under_inference_contrast():
    C1 = pd.read_csv(OUT / "c1_per_history.csv", dtype={"responses": str})
    D = pd.read_csv(P19 / "per_person_runs.csv")
    N = D[(D.world == "REACT") & (D["mode"] == "naive") & (D.policy.isin(["S", "O", "OC"]))]
    merged = C1.merge(N[["policy", "plan_target", "u", "items"]], on=["policy", "plan_target", "u"], suffixes=("", "_p19"))
    assert len(merged) == 2400 and (merged["items"] == merged["items_p19"]).all()
    assert (C1.responses.str.len() == 8).all()


def test_posterior_has_no_latent_truth_future_or_planner_access():
    m = runner()
    assert list(inspect.signature(m.rescore).parameters) == ["items", "ys", "policy", "aware"]
    src = inspect.getsource(m.rescore) + inspect.getsource(m.own_action_reaction)
    for banned in ("tau[", "kap", "nu[", "U[", "true_", "op_choose", "scalar_choose", "expit("):
        assert banned not in src
    # causal prefix: the reaction used at turn t depends only on items before t
    rs, _ = m.own_action_reaction([4, 20, 4, 20]); rs2, _ = m.own_action_reaction([4, 20, 4, 21])
    assert rs[:4] == rs2[:4]
    calls = []
    orig_o, orig_s = m.M.op_choose, m.M.scalar_choose
    m.M.op_choose = lambda *a, **k: calls.append(1); m.M.scalar_choose = lambda *a, **k: calls.append(1)
    try:
        m.rescore([4, 20, 4, 20, 4, 20, 4, 20], [1, 0, 1, 1, 0, 1, 0, 1], "O", aware=True)
        m.rescore([4, 20, 4, 20, 4, 20, 4, 20], [1, 0, 1, 1, 0, 1, 0, 1], "S", aware=True)
    finally:
        m.M.op_choose, m.M.scalar_choose = orig_o, orig_s
    assert not calls


def test_deterministic_pairing_and_cost_zero_rule():
    res = json.loads((OUT / "results.json").read_text())
    assert res["checks"]["c2_pairing_failures"]["n_failures"] == 0 and res["checks"]["c2_zero_world_failures"]["n_failures"] == 0
    rule = res["checks"]["cost_zero_rule"]
    assert rule["states"] == rule["argmax_equals_choice_equals_original_react_cost_false"] == 320 and rule["max_abs_utility_rule_error"] < 1e-15
    m = runner(); w = m.M.prior_op()
    for j, y in ((4, 1), (21, 0)):
        w = w * m.M.lik_op(m.M.ITEMS[j], y, .2); w = w / w.sum()
    assert m.op_choose_cost0(w, "T1", True, .2, True) == m._ORIG_OP(w, "T1", True, .2, False)
    u0, u10 = m.utilities(w, "T1", .2, 0.0), m.utilities(w, "T1", .2, .10)
    assert np.allclose(u0 - u10, .02 * m.M.ITEMS[:, 2], atol=1e-15)
    C2 = pd.read_csv(OUT / "c2_per_person.csv")
    assert len(C2) == 800 and set(C2.plan_target) == {"T1", "T2"}


def test_decomposition_identity_and_ci_verdict_labels():
    res = json.loads((OUT / "results.json").read_text())
    for v in res["c1_paired"].values():
        assert v["decomposition_point_identity_error"] < 1e-12
        for k, p in v.items():
            if isinstance(p, dict):
                lo, hi = p["ci"]
                assert p["verdict"].startswith("INCONCLUSIVE") == (lo <= 0 <= hi)
