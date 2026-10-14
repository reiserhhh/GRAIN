"""Focused tests for packet21 B (R05 paired four-component identification).
Algebra/estimator tests (pre-run) and output tests (post-run, prefixed test_output_)."""
import hashlib
import inspect
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import r05_four_component as m  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/F03/paired_four_component_identification_r1"
PROTOCOL = ROOT / "docs/GRAIN_R05_PAIRED_FOUR_COMPONENT_PROTOCOL_V1.md"
SIG1 = np.diag([.36, .25, .49, .16])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_rank_and_null_witness():
    assert m.design_summary(m.L_NOMINAL[[0, 1, 2]])["rank"] == 2
    assert m.design_summary(m.L_NOMINAL[[0, 1, 2, 3]])["rank"] == 3
    assert m.design_summary(m.L_NOMINAL)["rank"] == 4
    v = m.NULL_WITNESS
    for rows in ([0, 1, 2], [0, 1, 2, 3]):
        L = m.L_NOMINAL[rows]
        assert np.allclose(L @ v, 0)
        assert np.max(np.abs(m.implied_cov(L, SIG1 + .37 * np.outer(v, v)) - m.implied_cov(L, SIG1))) < 1e-15
    assert np.max(np.abs(m.implied_cov(m.L_NOMINAL, SIG1 + .37 * np.outer(v, v)) - m.implied_cov(m.L_NOMINAL, SIG1))) > .1


def test_invertible_coordinate_equivalence_leaky_clamp_and_changed_D():
    L3 = m.L_NOMINAL.copy(); L3[3] = [0, 1, 1, .3]
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, .3], [0, 0, 0, .7]])
    assert np.allclose(m.L_NOMINAL @ T, L3, atol=1e-15)
    assert np.allclose(m.implied_cov(m.L_NOMINAL, T @ SIG1 @ T.T), m.implied_cov(L3, SIG1), atol=1e-15)
    assert not np.allclose(T @ SIG1 @ T.T, SIG1)            # named components differ
    LD = m.L_NOMINAL.copy(); LD[3] = [0, 1, 1.5, 0]         # seminar-18 claim: changed independent D row
    TD, err = m.basis_transform(m.L_NOMINAL, LD)
    assert err < 1e-12 and np.allclose(m.implied_cov(m.L_NOMINAL, TD @ SIG1 @ TD.T), m.implied_cov(LD, SIG1), atol=1e-14)
    assert np.allclose(m.restriction_residuals(m.implied_cov(LD, SIG1)), 0, atol=1e-15)
    G = np.random.default_rng(3).normal(size=(4, 4)) + 2 * np.eye(4)
    assert np.allclose(m.implied_cov(m.L_NOMINAL @ np.linalg.inv(G), G @ SIG1 @ G.T), m.implied_cov(m.L_NOMINAL, SIG1), atol=1e-12)


def test_observable_C_restriction():
    L4 = m.L_NOMINAL.copy(); L4[2] = [0, 1.75, 1, 1]
    r = m.restriction_residuals(m.implied_cov(L4, SIG1))
    assert np.allclose(r, [(1.75 ** 2 - 1) * .25, .75 * .25, .75 * .25, .75 * .25, .75 * .25])
    _, err = m.basis_transform(m.L_NOMINAL, L4)
    assert err > .5                                          # C != B cannot be absorbed by coordinates
    assert np.allclose(m.restriction_residuals(m.implied_cov(m.L_NOMINAL, SIG1)), 0)


def test_replicate_noise_correction_and_alternative_formula():
    d, _ = m.simulate(SIG1, m.L_NOMINAL, np.zeros(5), .8, 4000, 1, 3, [99, 1])
    Y = d["Y_src"]
    C, noise, _ = m.between_cov(Y)
    assert np.max(np.abs(C - m.between_cov_crossrep(Y))) < 1e-12
    assert np.allclose(noise, .64, atol=.03)
    assert np.max(np.abs(C - m.implied_cov(m.L_NOMINAL, SIG1))) < .06
    d0, _ = m.simulate(np.eye(4) * 1e-12, m.L_NOMINAL, np.zeros(5), .8, 4000, 1, 3, [99, 2])
    C0, _, _ = m.between_cov(d0["Y_src"])
    assert np.max(np.abs(C0)) < .03                          # pure noise: between covariance ~ 0 after correction


def test_shared_authors_across_arms():
    d, t = m.simulate(SIG1, m.L_NOMINAL, np.zeros(5), 0.0, 50, 10, 3, [99, 3])
    Y = d["Y_src"]
    assert np.allclose(Y[:, 0, 0] - Y[:, 1, 0], t["theta_src"][:, 0])     # A-B is the same person's choice
    assert np.allclose(Y[:, 4, 1], t["theta_src"][:, 1])                   # E is the same person's interpretation
    assert np.allclose(Y[:, 1], Y[:, 2])                                   # B and C share I+S+R when noise-free


def test_no_application_truth_access():
    for f in (m.between_cov, m.between_cov_crossrep, m.summary_stats, m.scores, m.bootstrap, m.sigma_hat):
        params = set(inspect.signature(f).parameters)
        assert not params & {"theta", "truth", "theta_app", "theta_src", "Sigma_true", "L_true"}
    d, t = m.simulate(SIG1, m.L_NOMINAL, np.zeros(5), .8, 100, 40, 3, [99, 4])
    s1 = m.scores(d["Y_app"], d["Y_src"].mean(axis=(0, 2)))
    t["theta_app"][:] = 0                                                  # truth mutation cannot change scores
    assert np.array_equal(s1, m.scores(d["Y_app"], d["Y_src"].mean(axis=(0, 2))))


def test_deterministic_whole_person_bootstrap():
    d, _ = m.simulate(SIG1, m.L_NOMINAL, np.zeros(5), .8, 60, 1, 3, [99, 5])
    a, ia = m.bootstrap(d["Y_src"], 20, [7, 7], block=7)
    b, ib = m.bootstrap(d["Y_src"], 20, [7, 7], block=20)
    assert np.array_equal(ia, ib) and all(np.array_equal(a[k], b[k]) for k in a)
    st = m.summary_stats(d["Y_src"][ia[3]])
    assert np.allclose(st["sigma_vec"], a["sigma_vec"][3])


# ------------------------------------------------------------------ outputs (post-run)
def test_output_protocol_bindings_and_counts():
    assert sha(PROTOCOL) == "46139a8b2062d574923c15e48263024da43b5729d89ee7ba67f9f4a0e62cf221"
    for rel, h in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        assert sha(ROOT / rel) == h
    res = json.loads((OUT / "results.json").read_text())
    assert res["scalar_observations"] == 81000 and res["neural_calls"] == 0 and res["identity_max_err"] < 1e-12


def test_output_witnesses_replay():
    w = json.loads((OUT / "analytic_witnesses.json").read_text())
    assert [w["designs"][k]["rank"] for k in ("ABC", "ABCD", "ABCDE", "ABDE")] == [2, 3, 4, 4]
    assert w["designs"]["ABC"]["cov_change_under_Sigma_plus_0.1vv'"] < 1e-15 and w["designs"]["ABCDE"]["cov_change_under_Sigma_plus_0.1vv'"] > .01
    assert w["W3_leaky_clamp"]["implied_cov_nominal(TSigmaT')_minus_true"] < 1e-15 and w["W3_leaky_clamp"]["solved_equals_declared"] < 1e-12
    assert w["changed_D_style_loading_delta_.5"]["implied_cov_misfit"] < 1e-14
    assert max(abs(x) for x in w["W4_C_restriction_violated"]["restriction_residuals_population"]) > .1


def test_output_negative_eigenvalues_reported_not_clipped():
    per = json.loads((OUT / "per_world_seed.json").read_text())
    for v in per.values():
        assert len(v["eigen_C_hat"]) == 5 and len(v["eigen_Sigma_hat"]) == 4
        assert abs(v["ABC_only"]["fitted_cov_change_Sigma_hat_plus_0.1vv'"]) < 1e-12
