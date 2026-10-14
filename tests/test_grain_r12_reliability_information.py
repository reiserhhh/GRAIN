"""Focused tests for packet 28 (R12 reliability / diversity / information; FU2-c uncertainty recovery).
Pre-run algebra tests + test_output_ tests. Tests check implementation only, not a scientific outcome."""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import r12_reliability_information as m  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/F04/reliability_information_r1"
PROTOCOL = ROOT / "docs/GRAIN_R12_RELIABILITY_INFORMATION_PROTOCOL_V1.md"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.mark.parametrize("n,k", [(8, 4), (16, 4), (32, 4), (64, 4), (32, 2), (32, 8)])
def test_balanced_disjoint_halves(n, k):
    z = m.design_z(n, k); A, B = m.halves(n)
    assert not set(A) & set(B) and len(A) == len(B) == n // 2 and sorted(np.r_[A, B]) == list(range(n))
    for h in (A, B):
        vals, cnt = np.unique(np.round(z[h], 12), return_counts=True)
        assert len(vals) == k and len(set(cnt)) == 1
    assert abs(z[A].mean() - z[B].mean()) < 1e-12


def test_k1_refused_not_scored():
    with pytest.raises(m.Unidentified):
        m.design_z(32, 1)
    with pytest.raises(m.Unidentified):
        m.slope_var(0.0)
    assert m.information(np.ones(16))["rank"] == 1


def test_analytic_variance_and_spearman_brown_identity():
    for n, k in [(8, 4), (32, 2), (32, 8)]:
        z = m.design_z(n, k); A, B = m.halves(n)
        w = m.slope_weights(z[A])
        assert abs(m.SIGMA2 * (w @ w) - m.slope_var(m.szz(z[A]))) < 1e-14          # Var(b_hat|b) = sigma2/Szz exactly
        assert abs(w @ z[A] - 1) < 1e-12 and abs(w.sum()) < 1e-12                    # unbiased, intercept-free
        assert abs(m.szz(z) - m.szz(z[A]) - m.szz(z[B])) < 1e-12                     # equal half means
        r = m.half_reliability(m.slope_var(m.szz(z[A])), m.slope_var(m.szz(z[B])))
        assert abs(m.full_reliability(m.szz(z)) - m.spearman_brown(r)) < 1e-12
    # unequal halves: SB of the geometric half formula no longer equals the full-slope formula
    vA, vB = 0.02, 0.2
    r = m.half_reliability(vA, vB)
    full = m.VAR_B / (m.VAR_B + 1 / (1 / vA + 1 / vB))
    assert abs(m.spearman_brown(r) - full) > 1e-3


def test_equal_szz_invariance():
    target = m.szz(m.design_z(32, 4)[:16])
    rel = []
    for k in (2, 4, 8):
        z = m.equal_szz(m.design_z(32, k), target); A, B = m.halves(32)
        assert abs(m.szz(z[A]) - target) < 1e-10 and abs(m.szz(z[B]) - target) < 1e-10
        rel.append(m.full_reliability(m.szz(z)))
    assert max(rel) - min(rel) < 1e-12
    assert abs(m.equal_szz(m.design_z(32, 2), target).max() - np.sqrt(target / 16)) < 1e-12   # support changes


def test_ols_slopes_match_lstsq():
    rng = np.random.default_rng(1); z = m.design_z(16, 4); Y = rng.normal(size=(5, 16))
    ref = [np.linalg.lstsq(np.column_stack([np.ones(16), z]), y, rcond=None)[0][1] for y in Y]
    assert np.allclose(m.ols_slopes(Y, z), ref)


def test_paired_cluster_bootstrap_stratified():
    idx = m.stratified_indices(400, 3, 50, np.random.default_rng(2))
    assert idx.shape == (50, 1200)
    for s in range(3):
        blk = idx[:, s * 400:(s + 1) * 400]
        assert blk.min() >= s * 400 and blk.max() < (s + 1) * 400
    X = np.arange(1200.0)
    assert np.array_equal(X[idx], X[idx])            # the same indices serve every design (pairing is by construction)


def test_log_grid_curvature():
    x = np.log2([8, 16, 32, 64])
    D, C = m.divided_differences(x, x ** 2)
    assert np.allclose(C, 1.0) and np.allclose(D, [7, 9, 11])
    D, C = m.divided_differences(x, 3 * x + 1)
    assert np.allclose(C, 0) and np.allclose(D, 3)
    D, C = m.divided_differences(np.log2([2, 4, 8]), np.array([[1.0, 2.0, 4.0], [0.0, 0.0, 0.0]]))
    assert C.shape == (2, 1) and np.isclose(C[0, 0], 0.5)


# ------------------------------------------------------------------ outputs (after the frozen run)
def test_output_bindings_gates_and_frozen_roots():
    res = json.loads((OUT / "results.json").read_text())
    assert res["protocol_sha256"] == sha(PROTOCOL) and res["neural_calls"] == 0
    for rel, hh in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        assert sha(ROOT / rel) == hh
    g = res["gates"]
    assert g["k1_refused"] and g["crn_shared_noise"] and g["equal_szz_max_dev"] < 1e-10
    assert g["analytic_vs_old_curve_max_abs"] < 0.01
    assert g["fu2c_replay_max_abs"] <= 1e-9 and g["fu2c_old_retest_ci_max_abs"] <= 1e-9


def test_output_deterministic_scoring_replay():
    res = json.loads((OUT / "results.json").read_text())
    arr = np.load(OUT / "author_arrays.npz", allow_pickle=False)
    designs = [d for d in res["design_table"] if d["scored"]]
    for d in designs:
        did = d["design_id"]
        for s in range(3):
            bA, bB = arr[f"{did}__bA"][s], arr[f"{did}__bB"][s]
            r = float(np.corrcoef(bA, bB)[0, 1])
            assert abs(r - res["per_seed"][f"{did}|seed{s}"]["half_r_empirical"]) < 1e-12
