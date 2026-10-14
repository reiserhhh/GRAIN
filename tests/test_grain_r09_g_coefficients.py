"""Focused tests for packet20 D (R09 G coefficients from saved REML moments)."""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results/grain_gptpro_foundation_v3/F04/generalizability_coefficients_r1"
PROTOCOL = ROOT / "docs/GRAIN_R09_G_COEFFICIENT_PROTOCOL_V1.md"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_protocol_bindings_and_amendment_recorded():
    assert sha(PROTOCOL) == "77e37e56f8681b77b3908720a902fd1281cde8a8675ad88bebcd731c944739ac"
    for rel, h in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        assert sha(ROOT / rel) == h
    res = json.loads((OUT / "results.json").read_text())
    assert sha(ROOT / "docs/GRAIN_R09_G_COEFFICIENT_PROTOCOL_V1_AMENDMENT_1.md") in res["amendment"]
    assert (ROOT / "results/grain_gptpro_foundation_v3/F04/generalizability_coefficients_r1_failed_attempt1/FAILED_REASON.md").exists()
    assert res["fits"] == 0 and res["neural_calls"] == 0


def test_composites_reproduce_universe_score():
    ch = json.loads((OUT / "composite_checks.json").read_text())
    assert all(v["loading_err"] <= 1e-12 for v in ch.values())
    assert all(v["facet_cancel_err"] <= 1e-12 for k, v in ch.items() if k.startswith("full48"))


def test_closed_form_matches_monte_carlo_for_one_composite():
    from grain_research import reliability_spine as rs
    import run_grain_r09_g_coefficients as g
    w = g.composites(rs.rows_half(0, 0))["B12"]
    vt, vd, vs, ve = g.parts(w, rs.SIG, .15, rs.RHO, rs.SW2, rs.SE2)
    rng = np.random.default_rng(7); n = 4000
    th = rng.multivariate_normal(np.zeros(6), rs.SIG, n); d = rng.normal(0, np.sqrt(.15), (n, 24))
    S = np.stack([rng.multivariate_normal(np.zeros(rs.NROW), rs.state_cov(rs.RHO, rs.SW2 / (1 - rs.RHO ** 2), rs.pair_of(u))) for u in range(n)])
    y = th @ rs.ZT.T + d @ rs.ZD.T + S + rng.normal(0, np.sqrt(rs.SE2), (n, rs.NROW))
    comp = y @ w
    assert abs(np.var(comp) - (vt + vd + vs + ve)) / (vt + vd + vs + ve) < .06
    assert abs(np.corrcoef(comp, th[:, 3])[0, 1] ** 2 - vt / (vt + vd + vs + ve)) < .05


def test_universe_ordering_and_labels():
    d = pd.read_csv(OUT / "g_coefficients.csv")
    assert (d.G_rel_fixed_wording >= d.G_rel_random_wording - 1e-15).all()
    assert (d.Phi_random_wording == "NOT_ESTIMATED").all()
    assert np.allclose(d.Phi_fixed_wording, d.G_rel_fixed_wording)
