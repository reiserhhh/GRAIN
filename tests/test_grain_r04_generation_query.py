"""Packet 32 tests (implementation only; not scientific support)."""
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import r04_generation_query as m  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/F01/generation_query_r1"


def test_distinct_vs_replica_rank():
    Rd = np.stack([m.SENS[q][0] for q in m.PAIRS["distinct"]]); Rr = np.stack([m.SENS[q][0] for q in m.PAIRS["replica"]])
    assert np.linalg.matrix_rank(Rd) == 2 and np.linalg.matrix_rank(Rr) == 1


def test_realization_gain_cancels_in_ratio():
    """Noise-free: rho = alpha/bG does not depend on K (generation-side gain), only on theta and language constants."""
    U = 3; th = np.array([0.5, 0.5, 0.5]); K = np.array([0.6, 1.0, 1.7]); L = np.zeros(U, int); g, a = m.crossed(U)
    Z = np.zeros((U, m.T, 3)); Zq = np.zeros((U, m.T, 2))
    st = m.author_ols(m.generate(th, K, L, g, a, "distinct", Z, Zq), g, a)
    rho = st["alpha"] / st["bG"]
    assert np.abs(rho - rho[0]).max() < 1e-10 and np.abs(st["bG"][:, 0] / K - st["bG"][0, 0] / K[0]).max() < 1e-10


def test_reader_side_vs_generation_side_separation_noise_free():
    """Reader-side lam enters bA additively (no K); generation-side audience effect scales with K."""
    U = 4; th = np.zeros(U); K = np.array([0.5, 1.0, 1.5, 2.0]); L = np.zeros(U, int); g, a = m.crossed(U)
    st = m.author_ols(m.generate(th, K, L, g, a, "distinct", np.zeros((U, m.T, 3)), np.zeros((U, m.T, 2))), g, a)
    rs = m.reader_side({**st, "S": np.zeros_like(st["S"])})
    assert abs(rs[1]["lambda_hat"] - 0.4) < 1e-9 and abs(rs[0]["lambda_hat"]) < 1e-9


def test_estimator_truth_isolation():
    for fn in (m.author_ols, m.ratio, m.calibrate, m.reader_side, m.score):
        assert not set(inspect.signature(fn).parameters) & {"theta", "K", "truth", "W_TH", "loadings"}, fn.__name__


def test_output_checks():
    res = json.loads((OUT / "results.json").read_text())
    c = res["checks"]
    assert c["witness_conf_max_abs"] < 1e-10 and c["witness_readerside_max_abs"] < 1e-10 and c["witness_language_max_abs"] < 1e-10
    assert c["conf_theta_refused"] and c["independent_recompute_max_abs"] < 1e-9
    PC = pd.read_csv(OUT / "paired_contrasts.csv")
    assert {"distinct-replica", "anchor-noanchor"} <= set(PC.contrast)
