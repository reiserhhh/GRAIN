"""Focused tests for the R24 target repair (protocol V2, personal_law_r2)."""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import run_grain_r24_personal_law as r1  # noqa: E402
import run_grain_r24_personal_law_r2 as r2  # noqa: E402
from grain_research.gptpro_foundation_v3.f08_estimators import FEATURES  # noqa: E402
from grain_research.gptpro_foundation_v3.f13_binary_ensemble import SIGMA2, EnsembleHistory  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/F13/personal_law_r2"
P_XOR = np.array([0.25, 0.75, 0.75, 0.25])


def miss_prob(mech, y, last):
    return {"MCAR": 0.4, "MNAR": 0.6 if y == 1 else 0.2, "MAR": 0.6 if last == 1 else 0.2, "none": 0.0}[mech]


def brute_force(p, obs_vals, flags, t, mech):
    """Exhaustive enumeration over y in {0,1}^t with the full selection likelihood (observed-step factors included)."""
    num = den = 0.0
    for ys in itertools.product((0, 1), repeat=t):
        if any(flags[i] and ys[i] != obs_vals[i] for i in range(t)):
            continue
        pr, last = 0.25, None
        for i in range(2, t):
            s = 2 * ys[i - 2] + ys[i - 1]
            pr *= p[s] if ys[i] else 1 - p[s]
        for i in range(t):
            q = miss_prob(mech, ys[i], last)
            pr *= (1 - q) if flags[i] else q
            if flags[i]:
                last = ys[i]
        num += pr * p[2 * ys[t - 2] + ys[t - 1]]
        den += pr
    return num / den


def test_two_step_mnar_example():
    assert r2.integrated_sel(P_XOR, [0, 0], [False, True], 2, "MNAR") == pytest.approx(0.625, abs=1e-15)
    assert r1.true_integrated(P_XOR, [0, 0], [False, True], 2) == pytest.approx(0.5, abs=1e-15)


def test_complete_history_equality():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 4)
    ys = list(rng.integers(0, 2, 20))
    for mech in ("none", "MCAR", "MAR", "MNAR"):
        for t in range(2, 20):
            v = r2.integrated_sel(p, ys, [True] * 20, t, mech)
            assert v == r1.true_integrated(p, ys, [True] * 20, t)
            assert v == pytest.approx(p[2 * ys[t - 2] + ys[t - 1]], abs=1e-12)


@pytest.mark.parametrize("mech", ["MCAR", "MAR", "MNAR"])
def test_exhaustive_enumeration_and_ignorability(mech):
    rng = np.random.default_rng({"MCAR": 2, "MAR": 3, "MNAR": 4}[mech])
    diffs = []
    for _ in range(25):
        p = rng.uniform(0.05, 0.95, 4)
        t = int(rng.integers(3, 9))
        ys = [int(v) for v in rng.integers(0, 2, t)]
        flags = [bool(v) for v in rng.random(t) > 0.45]
        obs = [y if f else 0 for y, f in zip(ys, flags)]
        sel = r2.integrated_sel(p, obs, flags, t, mech)
        assert sel == pytest.approx(brute_force(p, obs, flags, t, mech), abs=1e-12)
        diffs.append(abs(sel - r1.true_integrated(p, obs, flags, t)))
    if mech in ("MCAR", "MAR"):
        assert max(diffs) < 1e-12
    else:
        assert max(diffs) > 1e-3


def test_variance_identities_gap_formula_and_counterexamples():
    rng = np.random.default_rng(5)
    for _ in range(50):
        k = 4
        w = rng.dirichlet(np.ones(k)); m = rng.uniform(0, 1, k)
        L = rng.normal(size=(k, k)); C = L @ L.T * 0.01
        pbar = w @ m
        i1 = w @ np.diag(C) + w @ (m - pbar) ** 2
        i2 = w @ C @ w
        D = 0.5 * sum(w[i] * w[j] * ((m[i] - m[j]) ** 2 + C[i, i] + C[j, j] - 2 * C[i, j]) for i in range(k) for j in range(k))
        assert i1 - i2 == pytest.approx(D, abs=1e-14) and D >= -1e-15
    w = np.array([0.5, 0.5])
    def gap(m, C):
        return w @ np.diag(C) + w @ (m - w @ m) ** 2 - w @ C @ w
    assert gap(np.array([0.1, 0.9]), np.zeros((2, 2))) == pytest.approx(0.16)            # zero variances alone: D>0
    assert gap(np.array([0.5, 0.5]), np.diag([0.01, 0.01])) == pytest.approx(0.005)      # constant means alone: D>0
    assert gap(np.array([0.5, 0.5]), np.full((2, 2), 0.01)) == pytest.approx(0.0, abs=1e-18)  # equal means + Var(p1-p2)=0
    m, C = np.array([0.1, 0.9]), np.diag([0.01, 0.01])
    assert w @ np.diag(C) + w @ (m - w @ m) ** 2 == pytest.approx(0.17) and w @ C @ w == pytest.approx(0.005)
    # sampling verification of I1 and I2 on the numeric example
    r = np.random.default_rng(6)
    P = m + r.normal(size=(400_000, 2)) * 0.1
    S = r.integers(0, 2, 400_000)
    assert P[np.arange(len(P)), S].var() == pytest.approx(0.17, rel=0.01)
    assert (P @ w).var() == pytest.approx(0.005, rel=0.02)


def _masked_ensemble():
    records, _truth = r1.personal_world("H1", 20260915)
    fits, inits, _h, _m, _u = r1.source_side("H1", 20260915, records, 0)
    ens = EnsembleHistory(fits, inits)
    rng = np.random.default_rng(9)
    for t in range(30):
        seen = rng.random() > 0.4
        ens.observe(float(rng.integers(0, 2)) if seen else 0.5, 1.0 if seen else 0.0)
    return ens


def test_b2_and_m_full_covariance_by_explicit_maps_and_sampling():
    ens = _masked_ensemble()
    values, weights, personal = ens.per_state()
    w = weights[0]
    c_b2, c_m = r2.working_covariances(ens)
    assert np.max(np.abs(np.diag(c_b2) - personal[0, 1])) < 1e-12 and np.max(np.abs(np.diag(c_m) - personal[0, 2])) < 1e-12
    # independent explicit loops
    cov = SIGMA2 * np.linalg.inv(ens.gram[0] + np.eye(4))
    loop_b2 = np.array([[sum(FEATURES[s, i] * cov[i, j] * FEATURES[u, j] for i in range(4) for j in range(4)) for u in range(4)] for s in range(4)])
    assert np.max(np.abs(loop_b2 - c_b2)) < 1e-14
    n = ens.counts[0].sum(1); raw = (ens.counts[0, :, 1] + 1) / (n + 2); rv = raw * (1 - raw) / (n + 3); P = ens.projector[0]
    loop_m = np.array([[sum(P[j, s] * P[j, u] * rv[j] for j in range(4)) for u in range(4)] for s in range(4)])
    assert np.max(np.abs(loop_m - c_m)) < 1e-14
    # sampling through the working linear maps
    rng = np.random.default_rng(10)
    coef = np.linalg.solve(ens.gram[0] + np.eye(4), ens.rhs[0] + ens.source_coef[0])
    draws = rng.multivariate_normal(coef, cov, size=300_000) @ FEATURES.T
    assert np.allclose(draws.mean(0), values[0, 1], atol=5e-3)
    assert (draws @ w).var() == pytest.approx(w @ c_b2 @ w, rel=0.02)
    raws = raw + rng.normal(size=(300_000, 4)) * np.sqrt(rv)
    proj = ens.m_mean[0] + (raws - ens.m_mean[0]) @ P
    assert (proj @ w).var() == pytest.approx(w @ c_m @ w, rel=0.02)
    # off-diagonal terms matter for this actual state
    assert abs(w @ c_b2 @ w - (w ** 2) @ np.diag(c_b2)) > 1e-4


def test_working_estimators_do_not_depend_on_true_law():
    records, truth = r1.personal_world("H2", 20260917)
    fits, inits, hypers, _m, _u = r1.source_side("H2", 20260917, records, 3)
    author, group = next((k, g) for k, g in r1.ordered_rows(records).items() if g[0]["role"] == "application")
    ys = [int(r["observation"]) for r in group]
    seen = r1.mask(ys, "MNAR", np.random.default_rng(3))
    a, ea = r2.score_author_r2(ys, truth[author], seen, fits, inits, hypers, False, "MNAR")
    b, eb = r2.score_author_r2(ys, np.array([0.9, 0.1, 0.9, 0.1]), seen, fits, inits, hypers, False, "MNAR")
    for c in a:
        for k in ("width", "state", "personal", "source", "pers_fw", "width_fw", "brier"):
            assert a[c][k] == b[c][k], (c, k)
    point_cols = [i for i, name in enumerate(r2.EV_COLS) if name.endswith(("_point", "_state", "_personal", "_source", "_pers_fw")) or name.startswith(("w", "b2_C", "m_C"))]
    assert np.array_equal(ea[:, point_cols], eb[:, point_cols], equal_nan=True)


def test_gate_cold_verification_and_r1_preserved():
    g = json.loads((OUT / "gate.json").read_text())
    assert g["status"] == "PASS" and g["unmatched"] == 0 and max(g["max_abs_diff"].values()) == 0.0
    assert g["validation_equal"] and g["source_side_equal"]
    cv = json.loads((OUT / "cold_verification.json").read_text())
    assert cv["status"] == "PASS" and cv["rows_compared"] == 25200
    rt = json.loads((OUT / "runtime.json").read_text())
    assert rt["r1_files_unchanged"] and rt["neural_calls"] == 0
    r1dir = ROOT / "results/grain_gptpro_foundation_v3/F13/personal_law_r1"
    for rel, h in rt["r1_file_hashes"].items():
        data = (r1dir / rel).read_bytes()
        if hashlib.sha256(data).hexdigest() == h:
            continue
        # only a declared, append-only correction notice may follow the recorded bytes
        marker = b"\n\n## Packet-12 correction notice"
        assert rel == "decision.md" and marker in data, rel
        assert hashlib.sha256(data[: data.index(marker)]).hexdigest() == h


def test_complete_history_equality_and_analysis_hashes():
    comp = json.loads((OUT / "complete_history_equality.json").read_text())
    assert comp["weights_one_hot"] and comp["max_abs_targets_lat_ign_sel_fw"] == 0.0
    for c in ("b1", "b2", "m", "eb"):
        assert comp[c]["max_abs_state"] == 0.0 and comp[c]["max_abs_half_L_minus_FW"] <= 1e-12
    a = json.loads((OUT / "analysis.json").read_text())
    for f, h in a["output_sha256"].items():
        assert hashlib.sha256((OUT / f).read_bytes()).hexdigest() == h, f
