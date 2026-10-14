"""Focused tests for R24 personal laws (protocol V1 + A1)."""
from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_grain_r24_personal_law as rr  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/F13/personal_law_r1"


def test_eb_posterior_matches_scipy_beta():
    stats = pytest.importorskip("scipy.stats")
    alpha, beta = np.array([2.3, 7.1, 0.9, 50.0]), np.array([5.2, 1.4, 3.3, 150.0])
    x, n = np.array([3, 0, 7, 11]), np.array([9, 4, 7, 30])
    m, v = rr.eb_state(alpha, beta, x, n)
    for s in range(4):
        d = stats.beta(alpha[s] + x[s], beta[s] + n[s] - x[s])
        assert m[s] == pytest.approx(d.mean(), abs=1e-14)
        assert v[s] == pytest.approx(d.var(), abs=1e-14)


def test_tau2_matches_exact_rational_formula():
    xs = [3, 9, 1, 14, 6]
    ns = [10, 12, 8, 16, 11]
    counts = np.zeros((5, 4, 2))
    for i, (x, n) in enumerate(zip(xs, ns)):
        counts[i, 0] = [n - x, x]
        counts[i, 1:] = [1, 1]
    _a, _b, tau2 = rr.eb_hyper(counts)
    N = sum(ns)
    pbar = Fraction(sum(xs), N)
    S = sum(Fraction(n) * (Fraction(x, n) - pbar) ** 2 for x, n in zip(xs, ns))
    denom = N - Fraction(sum(n * n for n in ns), N)
    exact = max(Fraction(0), (S - pbar * (1 - pbar) * (len(ns) - 1)) / denom)
    assert tau2[0] == pytest.approx(float(exact), rel=1e-12)
    assert exact > 0


def test_homogeneous_reduction_on_constructed_counts():
    counts = np.zeros((40, 4, 2))
    counts[:, :, 1] = 3
    counts[:, :, 0] = 9          # every author has p_hat = .25 in every state
    alpha, beta, tau2 = rr.eb_hyper(counts)
    assert np.all(tau2 == 0) and np.allclose(alpha + beta, 1e6)
    m, v = rr.eb_state(alpha, beta, np.array([5, 0, 2, 9]), np.array([6, 3, 2, 9]))
    assert np.max(np.abs(m - 0.25)) < 1e-5 and np.max(v) < 1e-6


def test_true_integrated_equals_latent_on_complete_history():
    p_u = np.array([0.1, 0.8, 0.6, 0.3])
    ys = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1]
    for t in range(2, len(ys)):
        assert rr.true_integrated(p_u, ys, [True] * len(ys), t) == pytest.approx(p_u[2 * ys[t - 2] + ys[t - 1]], abs=1e-12)


def test_source_isolation_and_role_disjointness():
    side = {(h["world"], h["seed"]): h for h in json.loads((OUT / "source_side.json").read_text())}
    records, _truth = rr.personal_world("H1", 20260915)
    roles = {}
    for r in records:
        roles.setdefault(r["author_id"], set()).add(r["role"])
    assert all(len(v) == 1 for v in roles.values())
    source = [r for r in records if r["role"] == "source"]
    groups = rr.ordered_rows(source)
    stats = rr.author_statistics([groups[a] for a in sorted(groups)])
    a, b, t2 = rr.eb_hyper(stats["counts"])
    h = side[("H1", 20260915)]
    assert np.allclose(a, h["eb_alpha"], rtol=0, atol=1e-12) and np.allclose(t2, h["tau2_by_state"], rtol=0, atol=1e-15)
    everything = rr.ordered_rows(records)
    all_stats = rr.author_statistics([everything[k] for k in sorted(everything)])
    assert not np.allclose(rr.eb_hyper(all_stats["counts"])[2], t2)   # including non-source rows would change the fit


def test_prediction_before_observation():
    records, truth = rr.personal_world("H2", 20260916)
    fits, inits, hypers, _m, _u = rr.source_side("H2", 20260916, records, 3)
    author, group = next((k, g) for k, g in rr.ordered_rows(records).items() if g[0]["role"] == "application")
    ys = [int(r["observation"]) for r in group]
    flipped = ys[:-1] + [1 - ys[-1]]
    for seen in ([True] * 48, [i % 3 != 1 for i in range(48)]):
        a = rr.score_author(ys, truth[author], seen, fits, inits, hypers, True)
        b = rr.score_author(flipped, truth[author], seen, fits, inits, hypers, True)
        for c in a:
            for k in ("se_lat", "se_int", "width", "cover_lat", "cover_int", "personal", "source", "state"):
                assert a[c][k] == b[c][k], (c, k)
            assert a[c]["brier"] != b[c]["brier"]


def test_replay_gate_and_scope_of_eb_rows():
    rep = json.loads((OUT / "replay_check.json").read_text())
    assert rep["status"] == "PASS" and rep["rows"] == 600 and max(rep["max_abs_diff"].values()) == 0.0
    m = pd.read_csv(OUT / "author_metrics.csv")
    eb = m[m.candidate == "eb"]
    assert set(eb.mechanism) == {"none", "MCAR"}
    assert eb.groupby(["world", "mechanism"]).size().eq(600).all()
    assert m.groupby(["world", "mechanism", "candidate"]).size().eq(600).all()
    assert m.author_id.nunique() == 1800


def test_analysis_hashes_match():
    a = json.loads((OUT / "analysis.json").read_text())
    for f, h in a["output_sha256"].items():
        assert hashlib.sha256((OUT / f).read_bytes()).hexdigest() == h, f
