"""Packet 31 tests (implementation only, not scientific support)."""
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import r12_informative_contexts as m  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/F04/informative_contexts_r1"
Q = {"A": np.array([0, 1.0, 0]), "B": np.array([0, 0, 1.0]), "AB": np.array([0, 1.0, 1.0]) / np.sqrt(2)}


def test_block_equals_aggregated_information():
    for d in ("D1", "D2", "D3", "D2_relabel"):
        for t2 in (0.0, 0.25, 0.9):
            c = m.design(d)
            assert np.abs(m.info(c, 1.0, t2) - m.info_dense(c, 1.0, t2)).max() < 1e-10


def test_rowspace_refusal_and_witness():
    I = m.info(m.design("D1"), 1.0, 0.25)
    r, ok, wit, rank, _ = m.gls_risk(I, Q["B"])
    assert r is None and not ok and rank == 2 and abs(wit @ Q["B"]) > 0.5 and np.linalg.norm(I @ wit) < 1e-9
    assert m.gls_risk(I, Q["A"])[0] is not None


def test_exact_tau0_equality_and_floor():
    for q in Q.values():
        assert abs(m.gls_risk(m.info(m.design("D2"), 1.0, 0.0), q)[0] - m.gls_risk(m.info(m.design("D3"), 1.0, 0.0), q)[0]) < 1e-14
        assert m.gls_risk(m.info(m.design("D3"), 1.0, 0.25), q)[0] < m.gls_risk(m.info(m.design("D2"), 1.0, 0.25), q)[0]


def test_relabel_identity_scores_and_information():
    rng = np.random.default_rng(0); Y = rng.normal(size=(50, 32))
    s2, I2 = m.gls_scores(Y, m.design("D2"), 1.0, 0.25); sr, Ir = m.gls_scores(Y, m.design("D2_relabel"), 1.0, 0.25)
    assert np.array_equal(s2, sr) or np.abs(s2 - sr).max() < 1e-12
    assert np.abs(I2 - Ir).max() < 1e-12


def test_source_fitter_truth_isolation_and_recovery():
    assert set(inspect.signature(m.source_fit).parameters) == {"Y", "ctx"}
    rng = np.random.default_rng(1); c = m.design("D3"); U = 4000
    th = rng.normal(size=(U, 3)); d = rng.normal(size=(U, 8)) * 0.5; e = rng.normal(size=(U, 32))
    Y = np.zeros((U, 32))
    for k in c:
        Y[:, k["slots"]] = (th @ k["phi"])[:, None] + d[:, [k["dslot"]]] + e[:, k["slots"]]
    f = m.source_fit(Y, c)
    assert abs(f["sigma2"] - 1) < 0.03 and abs(f["tau2_raw"] - 0.25) < 0.03


def test_output_checks_and_contrasts():
    res = json.loads((OUT / "results.json").read_text())
    ch = res["checks"]
    assert ch["dense_vs_formula_max"] < 1e-10 and ch["relabel_score_max_abs"] < 1e-12 and ch["tau0_D2_D3_risk_max"] < 1e-14
    assert ch["unidentified_never_scored"] and ch["independent_risk_recompute_max"] < 1e-12
    PC = pd.read_csv(OUT / "paired_contrasts.csv")
    rel = PC[PC.contrast == "D2-D2_relabel"]
    assert (rel.point.abs() < 1e-12).all()
