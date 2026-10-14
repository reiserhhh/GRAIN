"""R08 text/behaviour quotient (protocol V1 + A1): falsifiers F1-F6 and independent structural certificates."""
from __future__ import annotations

import itertools
import json
import sys
from fractions import Fraction as F
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_grain_r08_text_behaviour_quotient as r  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/F15/text_behaviour_quotient_r1"
R = json.loads((OUT / "results.json").read_text())
TV = pd.read_csv(OUT / "pair_tv.csv")


def tvs(world, s, t, view, policy):
    d = TV[(TV.world == world) & (TV.s == s) & (TV.t == t) & (TV.view == view) & (TV.policy == policy)].sort_values("k")
    return d.TV.tolist()


def test_F1_W1_orbit_quotient_is_lumpable_and_construct_futures_equal():
    g = R["W1"]["g_checks"]
    assert g["is_automorphism"] and g["orbit_lumpability_construct"]["lumpable"] and g["construct_one_step_TV_s_vs_gs_max"] == 0
    assert g["raw_one_step_TV_s_vs_gs_max"] > 0.5 and g["raw_law_invariance_max_diff"] > 0          # raw text is NOT invariant
    assert tvs("W1", "x0h0r0", "x0h0r1", "construct", "passive") == [0, 0, 0, 0]
    assert R["W1"]["partitions"]["construct"]["classes"] == R["W1"]["partitions"]["G_orbit_quotient"]["classes"]


def test_F2_conflict_worlds_lose_information_under_orbit_collapse():
    w2, w3 = R["W2"]["g_checks"], R["W3"]["g_checks"]
    assert not w2["is_automorphism"] and not w2["orbit_lumpability_construct"]["lumpable"] and w2["construct_one_step_TV_s_vs_gs_max"] > 0
    assert not w3["orbit_lumpability_construct"]["lumpable"] and w3["construct_one_step_TV_s_vs_gs_max"] == 0   # one-step invariance holds
    assert tvs("W3", "x0h0r0", "x0h0r1", "construct", "passive")[1] > 0                                     # futures differ at k = 2
    l = pd.read_csv(OUT / "merge_losses.csv")
    for w in ("W2", "W3"):
        row = l[(l.world == w) & (l.view == "G_orbit_quotient")].iloc[0]
        assert row["brier_increase_B_t+1"] > 0 or row["brier_increase_B_t+3"] > 0


def test_F3_W6_passive_text_never_reveals_trait_but_B_and_probe_do():
    parts = R["W6"]["partitions"]
    assert ["x0h0r0", "x1h0r0"] in parts["Y_passive"]["classes"] and parts["Y_probed"]["size"] == 8
    assert tvs("W6", "x0h0r0", "x1h0r0", "Y", "passive") == [0, 0, 0, 0]
    assert tvs("W6", "x0h0r0", "x1h0r0", "B", "passive")[0] > 0 and tvs("W6", "x0h0r0", "x1h0r0", "Y", "probe_first")[0] > 0
    # structural certificate independent of Krylov: under a0 the Y-kernel is invariant under the swap x <-> 1-x
    w = r.world_W("W6")
    st = w["states"]
    swap = {i: st.index((1 - x, h, rr)) for i, (x, h, rr) in enumerate(st)}
    ops = r.view_ops(w, "Y", (0,), exact=True)
    assert all(M[swap[s]][swap[s2]] == M[s][s2] for M in ops.values() for s in range(len(st)) for s2 in range(len(st)))


def test_F4_W4_text_alone_nonidentifiable_B_resolves():
    for row in R["W4"]["K2_vs_K1_pi_pairs"]:
        assert row["TV_cprime_k1to4_max"] == 0 and row["TV_joint_k1to4_max"] > 0.1
    assert R["W4"]["population_TV_K1_vs_K2"]["cprime"] == 0 and R["W4"]["population_TV_K1_vs_K2"]["joint"] > 0
    for s in R["W4"]["invariant_vs_directional_scores"]:
        assert s["K1_P_repeat"] == s["K2_P_repeat"] and s["K1_E_next_cprime"] != s["K2_E_next_cprime"]


def test_W4_brute_force_hidden_path_enumeration_matches():
    """Independent formulation: sum over hidden paths explicitly for c'-sequences of length 3."""
    w = r.world_W4()
    lab = w["labels"]

    def law(s0, seq_len=3):
        out = {}
        S = len(w["states"])
        for path in itertools.product(range(S), repeat=seq_len):
            for obs in itertools.product(range(len(w["obs"])), repeat=seq_len):
                p, cur = F(1), s0
                for t in range(seq_len):
                    nxt = path[t]
                    p *= w["K"][0][cur][obs[t]][nxt]
                    cur = nxt
                    if p == 0:
                        break
                if p:
                    key = tuple(w["obs"][o][0] for o in obs)
                    out[key] = out.get(key, F(0)) + p
        return out
    a, b = law(lab.index("K2x0h0")), law(lab.index("K1x1h1"))
    assert a == b and sum(a.values()) == 1


def test_F5_W5_threshold_relation_nontransitive():
    for v in R["W5"]["w5_threshold"].values():
        assert v["nontransitive_threshold_exists"] and v["d01"] < v["d02"] and v["d12"] < v["d02"]


def test_F6_exact_float_agreement_metric_zero_set_and_crosscheck():
    for wn in ("W1", "W2", "W3", "W5", "W6"):
        for info in R[wn]["partitions"].values():
            if "exact_agrees" in info:
                assert info["exact_agrees"], wn
        assert R[wn]["metric"]["zero_iff_bisimilar"], wn
    for info in R["W4"]["partitions"].values():
        assert info["exact_agrees"]
    assert json.loads((OUT / "c2_xor_metric_crosscheck.json").read_text())["status"] == "PASS"


def test_W1_equivariance_independent_loop():
    w = r.world_W("W1")
    st, ob = w["states"], w["obs"]
    for a in (0, 1):
        for i, (x, h, rr) in enumerate(st):
            for j, (c, f, b) in enumerate(ob):
                for k, (x2, h2, r2) in enumerate(st):
                    gi, gj, gk = st.index((x, h, 1 - rr)), ob.index((c, 1 - f, b)), st.index((x2, h2, 1 - r2))
                    assert w["K"][a][gi][gj][gk] == w["K"][a][i][j][k]


def test_validity_gate_fails_closed(monkeypatch):
    real = r.world_W

    def bad(name):
        w = real(name)
        w["K"][0][0][0][0] += F(1, 10)
        return w
    monkeypatch.setattr(r, "world_W", bad)
    with pytest.raises(RuntimeError):
        r.validity_gate()


def test_failed_attempt_preserved():
    assert (ROOT / "results/grain_gptpro_foundation_v3/F15/text_behaviour_quotient_r1_failed_attempt1/FAILURE.md").exists()
