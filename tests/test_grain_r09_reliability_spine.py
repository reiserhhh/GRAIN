"""R09 reliability spine: oracle formula vs independent closed form, limits, identification, ICC, design invariants."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import reliability_spine as rsp  # noqa: E402


def _par(sd2=.05, sw2=rsp.SW2, se2=rsp.SE2, Sig=None):
    return {"Sig": rsp.SIG if Sig is None else Sig, "sd2": sd2, "rho": rsp.RHO, "sw2": sw2, "se2": se2}


def test_random_intercept_closed_form_not_prior_over_prior_plus_posterior():
    sa2, se2, n = 1.0, 4.0, 2          # R = 1/3; the incorrect prior/(prior+posterior) gives .6
    post = 1 / (1 / sa2 + n / se2)
    R = rsp.reliability(sa2, [post])
    assert abs(R - n * sa2 / (n * sa2 + se2)) < 1e-12
    assert abs(R - 1 / 3) < 1e-12 and abs(sa2 / (sa2 + post) - .6) < 1e-12   # the wrong formula differs


def test_generic_conditioning_matches_closed_form_intercept():
    # tiny: only coordinate-1 rows at one condition, no d/state -> y = a1 + B11 c1 + B12 c2 + e with c fixed (1,1)
    Sig = np.diag([1.3, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12])
    par = _par(sd2=1e-12, sw2=1e-12, se2=.7, Sig=Sig)
    rows = np.flatnonzero((rsp.R_K == 0) & (rsp.R_CI == 3))[:5]
    _, pc = rsp.posterior(par, (0, 1), rows)
    assert abs(pc[0, 0] - 1 / (1 / 1.3 + 5 / .7)) < 1e-8


def test_limits_no_data_zero_noiseless_one_unidentified_below_one():
    _, pc0 = rsp.posterior(_par(), (0, 1), [])
    assert all(abs(rsp.reliability(rsp.SIG[j, j], [pc0[j, j]])) < 1e-12 for j in range(6))
    tiny = _par(sd2=1e-10, sw2=1e-10, se2=1e-10)
    _, pc1 = rsp.posterior(tiny, (0, 1), np.arange(rsp.NROW))
    assert all(rsp.reliability(rsp.SIG[j, j], [pc1[j, j]]) > 1 - 1e-6 for j in range(6))
    only11 = np.flatnonzero(rsp.R_CI == 3)
    _, pc2 = rsp.posterior(tiny, (0, 1), only11)
    assert rsp.reliability(rsp.SIG[0, 0], [pc2[0, 0]]) < .9 and rsp.reliability(rsp.SIG[2, 2], [pc2[2, 2]]) < .9


def test_monte_carlo_corr2_matches_R_for_correct_model():
    rng = np.random.default_rng(1); par = _par(); pair = (0, 1)
    Kg, pc = rsp.posterior(par, pair, np.arange(rsp.NROW))
    V = rsp.V_of(par, pair)
    Lv = np.linalg.cholesky(V + 1e-12 * np.eye(rsp.NROW))
    n = 4000
    theta = rng.multivariate_normal(rsp.MU, rsp.SIG, n)
    # simulate full data exactly from the model: y = X beta + Zt(theta-mu) + rest, rest ~ N(0, V - Zt Sig Zt')
    Vrest = V - rsp.ZT @ rsp.SIG @ rsp.ZT.T
    Lr = np.linalg.cholesky(Vrest + 1e-12 * np.eye(rsp.NROW))
    Y = (rsp.X @ rsp.BETA_TRUE)[None, :] + (theta - rsp.MU) @ rsp.ZT.T + rng.standard_normal((n, rsp.NROW)) @ Lr.T
    est = (Y - rsp.X @ rsp.BETA_TRUE) @ Kg[:6].T
    for j in (0, 2):
        R = rsp.reliability(rsp.SIG[j, j], [pc[j, j]])
        r2 = np.corrcoef(est[:, j], theta[:, j])[0, 1] ** 2
        assert abs(r2 - R) < .03
    del Lv


def test_icc_formulas_known_cases():
    rng = np.random.default_rng(2)
    t = rng.normal(0, 1, 5000)
    M = np.c_[t + rng.normal(0, 1, 5000), t + rng.normal(0, 1, 5000)]
    a, c = rsp.icc_two_way(M)
    assert abs(c - .5) < .03 and abs(a - .5) < .03
    a2, c2 = rsp.icc_two_way(M + np.array([0, 3.]))    # constant rater shift lowers agreement only
    assert abs(c2 - c) < 1e-9 and a2 < a - .2


def test_design_invariants():
    assert rsp.NROW == 96 and rsp.X.shape == (96, 10) and np.linalg.matrix_rank(rsp.X) == 10
    assert set(map(tuple, rsp.HELD)).isdisjoint(set(map(tuple, rsp.OBS)))
    for p in range(4):
        assert sorted(rsp.P_ORDERS[p]) == list(range(24))
    assert np.allclose(rsp.G.sum(1), 0)


# ---------------- run-output checks (results produced by scripts/run_grain_r09_reliability_spine.py)
import json  # noqa: E402

import pandas as pd  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/F04/reliability_spine_r1"


def _res():
    return json.loads((OUT / "results.json").read_text())


def test_run_counts_roles_and_fit_success():
    R = _res(); P = pd.read_csv(OUT / "application_person_metrics.csv")
    assert R["rows_multichannel"] == 181440 <= 500000 and R["neural_calls"] == 0
    assert len(P) == 3 * 360 and set(P.world) == {"LOW", "HIGH", "CURVED"}
    for k, v in R["fits"].items():
        assert v["attempts"][0]["success"] and not v["flags"]["variance_below_1e-6"]
        assert abs(v["sd2_hat"] - v["sd2_true"]) > 0           # a real estimate, not the declared value
    assert (P.u >= 300).all()                                  # only application persons are scored


def test_known_R_matches_monte_carlo_corr2():
    R = _res()
    for w in ("LOW", "HIGH"):
        for nm in ("a1", "a2", "B11", "B12", "B21", "B22"):
            assert abs(R["known_model_reliability"][w][nm] - R["mc_check_corr2"][w][nm]) < .06


def test_independent_recomputation_of_one_application_estimate():
    R = _res(); P = pd.read_csv(OUT / "application_person_metrics.csv")
    f = R["fits"]["HIGH|2"]
    par = {"Sig": np.array(f["Sig_hat"]), "sd2": f["sd2_hat"], "rho": f["rho_hat"], "sw2": f["sw2_hat"], "se2": f["se2_hat"]}
    beta = np.array(f["beta_hat"])
    data = rsp.generate("HIGH", 2)
    for u in (300, 317, 419):
        pair = rsp.pair_of(u)
        Kg, _ = rsp.posterior(par, pair, np.arange(rsp.NROW))
        th = beta[:6] + (Kg @ (data["Y"][u] - rsp.X @ beta))[:6]
        row = P[(P.world == "HIGH") & (P.seed == 2) & (P.u == u)].iloc[0]
        assert np.allclose(th, [row[f"fit_{n}"] for n in ("a1", "a2", "B11", "B12", "B21", "B22")], atol=1e-9)
        assert np.allclose(data["theta"][u], [row[f"true_{n}"] for n in ("a1", "a2", "B11", "B12", "B21", "B22")], atol=1e-12)


def test_threshold_bracket_kappa_recomputed():
    D = pd.read_csv(OUT / "threshold_bracket_rows.csv"); R = _res()["threshold_bracket"]
    assert len(D) == 600
    x, y = D.h1.to_numpy(), D.h2.to_numpy()
    po = np.mean(x == y); pe = sum(np.mean(x == c) * np.mean(y == c) for c in range(3))
    assert abs((po - pe) / (1 - pe) - R["half_kappa"]) < 1e-12 and "not identified" in R["note"]
