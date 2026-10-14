"""Packet 30 B tests (implementation only; not scientific support)."""
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import r13_freedom_components as m  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/F07/freedom_components_r1"
Z = np.linspace(-1, 1, 6)
TR = dict(e=.3, O=.2, ACR=.05, W=.1, SLOPE=.25, AR=.04, CR=.03, A=1.0, R=.1)


def world(seed, nA=3000, nO=2, nR=3):
    rng = np.random.default_rng(seed); nC = 6
    g = lambda *s: rng.standard_normal(s)
    return (np.sqrt(TR["A"]) * g(nA, 1, 1, 1) + np.sqrt(TR["SLOPE"]) * g(nA, 1, 1, 1) * Z[None, :, None, None] + np.sqrt(TR["W"]) * g(nA, nC, 1, 1)
            + np.sqrt(TR["R"]) * g(1, 1, 1, nR) + np.sqrt(TR["AR"]) * g(nA, 1, 1, nR) + np.sqrt(TR["CR"]) * g(1, nC, 1, nR) + np.sqrt(TR["ACR"]) * g(nA, nC, 1, nR)
            + np.sqrt(TR["O"]) * g(nA, nC, nO, 1) + np.sqrt(TR["e"]) * g(nA, nC, nO, nR))


def test_component_recovery_large_world():
    f = m.estimate(world(0), Z)
    for c in ("e", "O", "ACR", "W", "SLOPE", "AR", "A"):
        assert abs(f["components"][c] - TR[c]) < 0.15 * TR[c] + 0.02, c
    assert f["ms"]["ss_identity_error"] < 1e-12


def test_rank_deficient_design_flagged_not_zero():
    f = m.estimate(world(1, nA=300, nO=1), Z)
    for c in ("e", "O", "ACR", "W"):
        assert f["flags"][c] == "UNIDENTIFIED" and f["components"][c] is None
    for c in ("A", "SLOPE", "AR"):
        assert f["flags"][c] != "UNIDENTIFIED"
    g = m.g_coefficients(f)
    assert g["G_level"] is not None          # the level-error combination stays identifiable


def test_mixed_target_covariance_not_sum_of_variances():
    lam, kap, rho, err = 0.8, 0.8, 0.3, 0.2
    vx = lam ** 2 + kap ** 2 + 2 * lam * kap * rho + err
    cov = (lam + kap * rho) + (lam * rho + kap)
    a, fid = m.aligned(cov, 2 + 2 * rho, vx)
    naive = (lam ** 2 + kap ** 2) / vx
    assert abs(fid - naive) > 1e-3 and 0 < fid <= 1
    # Monte Carlo check of Cov(x,t)^2/(Var t Var x)
    rng = np.random.default_rng(3); th = rng.standard_normal(200000); s = rho * th + np.sqrt(1 - rho ** 2) * rng.standard_normal(200000)
    x = lam * th + kap * s + np.sqrt(err) * rng.standard_normal(200000)
    assert abs(np.corrcoef(x, th + s)[0, 1] ** 2 - fid) < 0.01


def test_truth_isolation_of_fitter():
    for fn in (m.mean_squares, m.estimate, m.g_coefficients, m.scores, m.combo):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"truth", "theta", "beta", "T", "true_components"}, fn.__name__


def test_output_checks_and_paired_contrasts():
    res = json.loads((OUT / "results.json").read_text())
    c = res["checks"]
    assert c["ss_identity_max"] < 1e-10 and c["closed_form_max_abs"] < 1e-8 and c["g_recompute_max_abs"] < 1e-10
    assert c["negative_control_unidentified_all"] and c["negative_control_never_zero"] and c["main_all_identifiable"]
    CN = pd.read_csv(OUT / "paired_contrasts.csv")
    assert set(CN.contrast) == {"1-0", "0.5-0", "1-0.5"} and CN.groupby(["seed", "mechanism", "reader_regime", "contrast"]).size().eq(9).all()
    TRr = pd.read_csv(OUT / "target_recovery.csv")
    assert len(TRr) == 60
