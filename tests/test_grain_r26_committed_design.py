"""Focused tests for packet26 (committed two-observation recovery design). Pre-run + post-run test_output_ tests."""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import f06_committed_design as cd  # noqa: E402
from grain_research import f06_recovery_design as p25  # noqa: E402
from grain_research.gptpro_foundation_v3.f06_gaussian_design import gaussian_terms, source_prior  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/F06/recovery_committed_design_r1"
PROTOCOL = ROOT / "docs/GRAIN_R26_COMMITTED_DESIGN_PROTOCOL_V1.md"
SRC = json.loads((ROOT / "results/grain_gptpro_foundation_v3/F05/source_full_r2/core_20260915/source_model.json").read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_scored_sequence_equals_executed():
    s = p25.author_streams(20260915, 5)
    for arm in ("scalar_a", "recovery"):
        o = cd.run_arm(arm, "distributed", s, SRC)
        assert [p["t"] for p in o["plans"]] == [0] + list(cd.EPOCHS)
        for p in o["plans"]:
            t = p["t"]
            for k, (h, znext) in enumerate(p["committed"]):
                assert o["h"][t + k] == h and o["z"][t + k + 1] == znext
            chosen = p["candidates"][p["chosen"]]
            if t == 0:
                assert chosen == [p["committed"][0][1], p["committed"][0][0]]
            else:
                assert chosen == [p["committed"][0][1], p["committed"][0][0], p["committed"][1][1], p["committed"][1][0]]


def test_intermediate_y_cannot_change_committed_second_action_and_no_future_leakage():
    s = p25.author_streams(20260915, 7)
    base = cd.run_arm("recovery", "distributed", s, SRC)
    t = 9                                              # epoch: commits (h9,z10),(h10,z11); y10 is intermediate
    s2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
    s2["measurement"][10] += 7.0                       # change intermediate y10 only
    alt = cd.run_arm("recovery", "distributed", s2, SRC)
    assert alt["h"][9:11] == base["h"][9:11] and alt["z"][10:12] == base["z"][10:12]
    s3 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
    s3["innovations"][12:] += 5.0; s3["measurement"][12:] -= 4.0   # after the committed horizon of epoch 9
    fut = cd.run_arm("recovery", "distributed", s3, SRC)
    assert fut["h"][:11] == base["h"][:11] and fut["z"][:12] == base["z"][:12]
    assert [p["committed"] for p in fut["plans"] if p["t"] <= 9] == [p["committed"] for p in base["plans"] if p["t"] <= 9]


def test_windows_never_cross_endpoints():
    windows = [(1,)] + [(t + 1, t + 2) for t in cd.EPOCHS]
    for w in windows:
        assert max(w) <= 23
        for e in (7, 15, 23):
            assert not (min(w) <= e < max(w))            # no window spans e and e+1


def test_sequential_quota_feasibility():
    assert cd.feasible_pairs(5, [0.0] * 5) == [(1.0, 1.0)]          # 0 used, slots 5,6 left -> both forced
    assert all(h1 == 0.0 for h1, _ in cd.feasible_pairs(7, [0, 0, 1, 0, 0, 1, 0]))   # h7 forced 0
    assert (0.0, 0.0) not in cd.feasible_pairs(21, [0.0] * 16 + [0, 0, 0, 0, 0])      # block 2 needs 2 in slots 21,22
    for h1, h2 in cd.feasible_pairs(3, [1.0, 0.0, 0.0]):
        hist = [1.0, 0.0, 0.0, h1, h2]
        assert sum(hist) <= 2 and 2 - sum(hist) <= 7 - 5


def test_window_score_equals_direct_fisher_and_eta_lag():
    prior_mean, prior_cov, (q, r) = source_prior(SRC)
    post = {"mean": np.r_[prior_mean, 0.0]}
    z_hist, h_hist = [-1.0, 1.0, -1.0], [0.0, 1.0]
    sc = cd.score(post, z_hist, h_hist, [1.0, -1.0], [1.0, 0.0], SRC, "rho")
    F, _, terms = p25.fisher(prior_mean, [-1, 1, -1, 1, -1], [0, 1, 1, 0, 0], q, r)
    C = np.linalg.inv(np.linalg.inv(prior_cov) + F)
    assert abs(sc - terms["drho"] ** 2 * C[2, 2]) < 1e-15
    t0 = gaussian_terms(np.array([0, 0, .5, 1.0]), np.zeros(5), np.array([0, 0, 1, 0, 0.]), q, r)
    assert t0["forcing"][3] == 1.0 and t0["forcing_eta"][3] == 0 and t0["forcing_eta"][4] != 0


def test_fixed_replays_packet25_and_intervals_finite():
    s = p25.author_streams(20260915, 0)
    r1 = pd.read_csv(ROOT / "results/grain_gptpro_foundation_v3/F06/recovery_target_design_r1/private_budget_metrics.csv")
    for world in ("zero", "distributed"):
        o = cd.run_arm("fixed", world, s, SRC)
        ref = r1[(r1.key == "20260915_0000") & (r1.arm == "fixed") & (r1.world == world)].sort_values("budget")
        for m, (_, rr) in zip(o["metrics"], ref.iterrows()):
            assert abs(m["rho_hat"] - rr.rho_hat) < 1e-12 and abs(m["a_err"] - rr.a_err) < 1e-12
            assert -.99 < m["rho_lo"] <= m["rho_hat"] <= m["rho_hi"] < .99 and np.isfinite(m["rho_width"])


# ------------------------------------------------------------------ outputs
def test_output_gates_and_integrity():
    res = json.loads((OUT / "results.json").read_text())
    assert res["protocol_sha256"] == sha(PROTOCOL)
    for rel, hh in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        assert sha(ROOT / rel) == hh
    g = res["gates"]
    assert g["stream_mismatches"] == 0 and g["fixed_vs_packet25_max_abs"] <= 1e-12
    assert res["integrity"]["quota_violations"] == 0 and res["integrity"]["scored_vs_executed_mismatches"] == 0
    assert res["addon_beta0"]["y_identical_all"] is True
