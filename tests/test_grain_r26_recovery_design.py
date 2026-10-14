"""Focused tests for packet25 B (R26 recovery-parameter design). Pre-run + post-run test_output_ tests."""
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import f06_recovery_design as rd  # noqa: E402
from grain_research.gptpro_foundation_v3.f06_gaussian_design import gaussian_terms  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/F06/recovery_target_design_r1"
PROTOCOL = ROOT / "docs/GRAIN_R26_RECOVERY_DESIGN_PROTOCOL_V1.md"
SRC = json.loads((ROOT / "results/grain_gptpro_foundation_v3/F05/source_full_r2/core_20260915/source_model.json").read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_fisher_equals_direct_finite_difference_gaussian_information():
    psi = np.array([.2, .4, .6, .5]); z = [-1, 1, -1, 1]; h = [1, 0, 1, 0]; q, r = .25, .3
    F, Fcov, _ = rd.fisher(psi, z, h, q, r)
    eps = 1e-6; n = len(z)
    def mu_sig(p):
        t = gaussian_terms(p, np.array(z, float), np.array(h, float), q, r); return t["mean"], t["covariance"]
    m0, S0 = mu_sig(psi); Si = np.linalg.inv(S0)
    dm, dS = [], []
    for k in range(4):
        e = np.zeros(4); e[k] = eps
        mp, Sp = mu_sig(psi + e); mm, Sm = mu_sig(psi - e)
        dm.append((mp - mm) / (2 * eps)); dS.append((Sp - Sm) / (2 * eps))
    Fd = np.array([[dm[i] @ Si @ dm[j] + .5 * np.trace(Si @ dS[i] @ Si @ dS[j]) for j in range(4)] for i in range(4)])
    assert np.allclose(F, Fd, rtol=1e-5, atol=1e-7)


def test_rho_delta_derivative_and_action_lag():
    eta = .7; d = (0.99 * np.tanh(eta + 1e-7) - 0.99 * np.tanh(eta - 1e-7)) / 2e-7
    assert abs(d - 0.99 * (1 - np.tanh(eta) ** 2)) < 1e-7
    t0 = gaussian_terms(np.array([0, 0, .5, 1.0]), np.zeros(5), np.array([0, 0, 1, 0, 0.]), .25, .25)
    assert t0["forcing"][2] == 0 and t0["forcing"][3] == 1.0      # h at t=2 moves state at t=3, not t=2
    assert t0["forcing_eta"][3] == 0 and t0["forcing_eta"][4] != 0  # enters the eta derivative only from t+2


def test_covariance_term_candidate_invariant_and_included():
    prior_mean, prior_cov, (q, r) = rd.source_prior(SRC)
    Fc = [rd.fisher(prior_mean, [1, -1, cz, -cz], [0, 1, ch, 0], q, r)[1] for cz, ch in rd.CANDIDATES]
    assert np.allclose(Fc, Fc[0]) and Fc[0] > 0


def test_quota_rule():
    assert rd.allowed_actions(7, [0] * 7) == (0.0,)
    assert rd.allowed_actions(5, [0] * 5) == (1.0,)                 # 2 slots left (5,6), quota 2 -> forced
    assert rd.allowed_actions(3, [1, 1, 0]) == (0.0,)               # quota used
    assert rd.allowed_actions(2, [0, 1]) == (0.0, 1.0)
    assert rd.allowed_actions(23, [0] * 23) == (0.0,)


def test_no_future_leakage_and_matched_quota():
    s = rd.author_streams(20260915, 3)
    base = rd.run_arm("recovery", "distributed", s, SRC)
    s2 = dict(s); s2["innovations"] = s["innovations"].copy(); s2["measurement"] = s["measurement"].copy()
    s2["innovations"][12:] += 5.0; s2["measurement"][12:] -= 3.0
    alt = rd.run_arm("recovery", "distributed", s2, SRC)
    assert base["h"][:12] == alt["h"][:12] and base["z"][:13] == alt["z"][:13]   # choice after y_11 uses y_0..y_11 only
    for arm in rd.ARMS:
        assert rd.quota_ok(rd.run_arm(arm, "zero", s, SRC)["h"])


def test_first_condition_prior_and_streams_match_archive():
    for arm in rd.ARMS:
        assert rd.first_condition(arm, SRC) in (-1.0, 1.0)
    s = rd.author_streams(20260915, 0)
    arch = json.load(gzip.open(ROOT / "results/grain_gptpro_foundation_v3/F06/gaussian_full_r1/authors/20260915_application_0000.json.gz"))
    assert arch["author_id"] == s["author_id"] and arch["truth"]["rho"] == s["rho"]
    assert np.array_equal(np.array(arch["truth"]["innovations"]), s["innovations"])


# ------------------------------------------------------------------ outputs
def test_output_gates_quota_and_pairing():
    res = json.loads((OUT / "results.json").read_text())
    assert res["protocol_sha256"] == sha(PROTOCOL)
    for rel, hh in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        assert sha(ROOT / rel) == hh
    g = res["gates"]
    assert g["stream_mismatches"] == 0 and g["archived_fixed_replay_max_abs"] <= 1e-10
    assert res["integrity"]["quota_violations"] == 0 and res["neural_calls"] == 0
    assert res["authors_completed"] == res["authors_planned"] or res["status"] == "PARTIAL"
