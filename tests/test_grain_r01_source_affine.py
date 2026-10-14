"""Focused tests for packet 27 (R01 common shift versus z-dependent action). Pre-run synthetic tests + test_output_ tests.
Tests check algebra/implementation only; they do not encode any scientific outcome."""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import r01_source_affine as a  # noqa: E402
from grain_research import r01_source_crossed as x  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/P05/source_affine_action_r1"
PROTOCOL = ROOT / "docs/GRAIN_R01_SOURCE_AFFINE_PROTOCOL_V1.md"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _case(seed, n=9, p=3, D=40):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, p)) + 0.4
    U = rng.normal(size=(n, D))
    Dn = rng.normal(size=(n, D)) + rng.normal(size=D)            # shared donor component
    return Z, U, Dn


@pytest.mark.parametrize("kind", ["full", "Z_rank_deficient_with_constant", "U_rank_deficient", "least_squares"])
def test_centered_sylvester_equals_joint_primal(kind):
    Z, U, Dn = _case(3)
    lam = 1 / len(Z)
    if kind == "Z_rank_deficient_with_constant":
        Z = np.hstack([Z[:, :2], Z[:, :1], np.ones((len(Z), 1))])  # duplicated column and 1 in col(Z)
    if kind == "U_rank_deficient":
        U[4] = U[1]; U[7] = 2 * U[2] - U[5]
    if kind == "least_squares":
        lam = 0.0
    Q, V, _ = x.orth_basis(U)
    T = (Dn @ Q) @ V.T
    B, c = a.fit_affine(Z, V, T, lam)
    Bj, cj = a.fit_affine_joint(Z, V, T, lam) if lam > 0 or kind == "least_squares" else (None, None)
    P = (c[None, :] + Z @ B) @ V.T; Pj = (cj[None, :] + Z @ Bj) @ V.T
    assert np.abs(P - Pj).max() < 1e-9 * max(1.0, np.abs(T).max())
    if lam > 0:
        assert np.abs(B - Bj).max() < 1e-9 and np.abs(c - cj).max() < 1e-9      # unique under the slope penalty
    assert a.affine_grad(Z, V, T, B, c, lam) < 1e-9 * max(1.0, np.abs(T).max())
    # c identity when T = D_Q V'
    assert np.abs(c - (a.common_coef(Dn @ Q) - Z.mean(0) @ B)).max() < 1e-9
    if kind == "U_rank_deficient":
        assert V.shape[1] < len(U)


def test_basis_rotation_invariance():
    Z, U, Dn = _case(4)
    Q, V, _ = x.orth_basis(U); lam = 1 / len(Z)
    R, _ = np.linalg.qr(np.random.default_rng(5).normal(size=(V.shape[1], V.shape[1])))
    Q2, V2 = Q @ R, V @ R
    T = Dn @ U.T
    B, c = a.fit_affine(Z, V, T, lam); B2, c2 = a.fit_affine(Z, V2, T, lam)
    assert np.allclose((c + Z @ B) @ Q.T, (c2 + Z @ B2) @ Q2.T, atol=1e-10)
    Bl, Bl2 = a.fit_linear(Z, V, T, lam), a.fit_linear(Z, V2, T, lam)
    assert np.allclose(Bl @ Q.T, Bl2 @ Q2.T, atol=1e-10)


def test_linear_equals_frozen_packet24_solver():
    Z, U, Dn = _case(6)
    Q, V, _ = x.orth_basis(U); T = Dn @ U.T
    assert np.array_equal(a.fit_linear(Z, V, T, 1 / 9), x.fit_full(Z, V, T, 1 / 9))


def test_origin_translation_invariance_affine_not_linear():
    Z, U, Dn = _case(7)
    Q, V, _ = x.orth_basis(U); T = Dn @ U.T; lam = 1 / len(Z)
    delta = np.array([0.7, -1.1, 0.3]); zq = np.array([0.2, 0.5, -0.4]); vq = V[2]
    B, c = a.fit_affine(Z, V, T, lam); B2, c2 = a.fit_affine(Z - delta, V, T, lam)
    assert abs(a.predict(zq, B, c, vq) - a.predict(zq - delta, B2, c2, vq)) < 1e-9
    Bl, Bl2 = a.fit_linear(Z, V, T, lam), a.fit_linear(Z - delta, V, T, lam)
    assert abs(zq @ Bl @ vq - (zq - delta) @ Bl2 @ vq) > 1e-6


def test_orientation_identity():
    rng = np.random.default_rng(8)
    B = rng.normal(size=(3, 5)); c = rng.normal(size=5); z = rng.normal(size=3); v = rng.normal(size=5)
    assert abs(a.oddness(z, B, c, v) - 2 * c @ v) < 1e-12
    assert abs(a.oddness(z, B, c, v, sigma=1.0)) < 1e-12
    assert abs(a.oddness(z, B, np.zeros(5), v)) < 1e-12          # LINEAR is odd


def test_projection_shares_and_orthogonal_identity():
    rng = np.random.default_rng(9)
    Z = rng.normal(size=(12, 3)) + 0.2
    Zc = Z - Z.mean(0)
    T = np.ones((12, 1)) @ rng.normal(size=(1, 12)) + Zc @ rng.normal(size=(3, 12))
    s = a.projection_shares(Z, T)
    assert abs(s["P1Z_share"] - 1) < 1e-10 and s["orthogonal_identity_error"] < 1e-10
    assert s["PZ_share"] < 1 and s["affine_minus_linear_projection_gain"] > 0
    s2 = a.projection_shares(Z, rng.normal(size=(12, 12)))
    assert s2["orthogonal_identity_error"] < 1e-10 and 0 < s2["residual_share"] < 1


def test_split_excludes_held_cell_in_both_roles():
    cells = [f"c{k}" for k in range(12) for _ in range(3)]
    for hc in sorted(set(cells)):
        tr, h = a.split(cells, hc)
        assert len(h) == 3 and len(tr) == 33 and not set(tr) & set(h)
        assert all(cells[k] != hc for k in tr)


def test_cold_score_matches_direct_predictions():
    Z, U, Dn = _case(10)
    Q, V, _ = x.orth_basis(U); T = Dn @ U.T; lam = 1 / len(Z); s = 4.0
    B, c = a.fit_affine(Z, V, T, lam); Bl = a.fit_linear(Z, V, T, lam); dbar = a.common_coef(Dn @ Q)
    Ua = np.random.default_rng(11).normal(size=(5, U.shape[1])); Za = np.random.default_rng(12).normal(size=(5, 3))
    got = a.cold_score({"Q": Q, "B_linear": Bl, "B_affine": B, "c_affine": c, "dbar": dbar}, Za, Ua, s)
    Va = Ua @ Q
    assert np.allclose(got["AFFINE"], [a.predict(Za[i], B, c, Va[i]) / 2 for i in range(5)])
    assert np.allclose(got["LINEAR"], [Za[i] @ Bl @ Va[i] / 2 for i in range(5)])
    assert np.allclose(got["COMMON"], Va @ dbar / 2)


# ------------------------------------------------------------------ outputs (after the frozen run)
def test_output_bindings_replays_and_gates():
    res = json.loads((OUT / "results.json").read_text())
    assert res["protocol_sha256"] == sha(PROTOCOL) and res["neural_calls"] == 0
    for rel, hh in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        assert sha(ROOT / rel) == hh
    g = res["gates"]
    assert g["linear_heldcell_replay_max_abs"] <= 1e-9 and g["linear_supplement_replay_max_abs"] <= 1e-9
    assert g["linear_application_replay_max_abs"] <= 1e-9
    assert g["affine_vs_joint_fold0_max_abs"] <= 1e-8 and g["max_stationarity_grad"] <= 1e-8
    assert g["c_identity_max_abs"] <= 1e-8 and g["origin_invariance_affine_max_abs"] <= 1e-8
    assert g["orientation_identity_max_abs"] <= 1e-8 and g["projection_identity_max"] <= 1e-8
    assert g["cold_replay_max_abs"] <= 1e-10 and g["held_exclusion_ok"]


def test_output_prediction_table_complete():
    P = pd.read_csv(OUT / "private_heldcell_predictions.csv")
    assert P.groupby(["query", "input"]).size().eq(36).all()
    for m in ("ZERO", "COMMON", "LINEAR", "AFFINE"):
        assert np.isfinite(P[f"pred_{m}"]).all()
    A = pd.read_csv(OUT / "private_application_predictions.csv")
    assert A.groupby(["query", "input"]).size().eq(72).all()
