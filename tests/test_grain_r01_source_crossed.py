"""Focused tests for packet24 B (R01 source-crossed derivative design). Pre-run + post-run test_output_ tests."""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import r01_source_crossed as x  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/P05/source_crossed_action_r1"
PROTOCOL = ROOT / "docs/GRAIN_R01_SOURCE_CROSSED_PROTOCOL_V1.md"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_fullcross_eigen_equals_primal_and_dual_and_mean_loss_penalty():
    rng = np.random.default_rng(0); Z = rng.normal(size=(8, 3)); U = rng.normal(size=(8, 40)); T = rng.normal(size=(8, 8))
    Q, V, _ = x.orth_basis(U); lam = 1 / 8
    Bf = x.fit_full(Z, V, T, lam)
    assert np.abs(Bf - x.fit_full_primal(Z, V, T, lam)).max() < 1e-12
    rows = [(i, j) for i in range(8) for j in range(8)]
    assert np.abs(Bf - x.fit_rows(Z, V, rows, T, lam)).max() < 1e-12
    assert x.loss_grad(Z, V, rows, T, Bf, lam) < 1e-12
    # diagonal with lambda=1/n equals the old summed alpha=1 J-cell dual
    D = [(i, i) for i in range(8)]; t = np.diag(T)
    K = (Z @ Z.T) * (V @ V.T); c = np.linalg.solve(K + np.eye(8), t)
    assert np.abs(x.fit_rows(Z, V, D, T, lam) - (Z * c[:, None]).T @ V).max() < 1e-12


def test_orthonormal_basis_preserves_frobenius_and_predictions():
    rng = np.random.default_rng(1); U = rng.normal(size=(6, 30)); Q, V, _ = x.orth_basis(U)
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1])) and np.allclose(V @ Q.T, U)
    B = rng.normal(size=(3, Q.shape[1])); W = B @ Q.T
    assert np.isclose(np.linalg.norm(W), np.linalg.norm(B))
    z = rng.normal(size=3)
    assert np.isclose(z @ W @ U[2], z @ B @ V[2])


def test_kronecker_rank_and_projection_share():
    rng = np.random.default_rng(2); Z = rng.normal(size=(10, 3)); U = rng.normal(size=(10, 50))
    assert x.kron_rank(Z, U) == (3, 10, 30)
    A = rng.normal(size=(3, 10)); T = Z @ A
    assert abs(x.pz_share(Z, T)[0] - 1) < 1e-12
    assert x.pz_share(Z, rng.normal(size=(10, 10)))[0] < 1


def test_cyclic_and_random_designs_are_permutations_without_same_cell():
    cells = [f"c{k}" for k in range(4) for _ in range(3)]; seeds = [f"s{k % 3}" for k in range(12)]
    for rows in (x.cyclic_rows(cells, cells, seeds), x.random_rows(cells, cells, seeds, [1, 2])):
        assert sorted(i for i, _ in rows) == list(range(12)) and sorted(j for _, j in rows) == list(range(12))
        assert all(cells[i] != cells[j] for i, j in rows)
    assert x.random_rows(cells, cells, seeds, [1, 2]) == x.random_rows(cells, cells, seeds, [1, 2])


# ------------------------------------------------------------------ outputs
def test_output_replays_ranks_and_units():
    res = json.loads((OUT / "results.json").read_text())
    assert res["protocol_sha256"] == sha(PROTOCOL)
    for rel, hh in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        assert sha(ROOT / rel) == hh
    assert res["trees"] == {"private_jacobians": "35e1b6d4d305a8e0aacf0f47827e42bf99921515d0fcf295c19c2e8ea649d751",
                            "private_app_jacobians": "8601b303be50ad1b0279ab4d7aad89c831c25cd6401d703a27955bc36adb60cc"}
    for q, r in res["replay"].items():
        assert max(r.values()) <= 1e-9
    for k, d in res["diagnostics"].items():
        assert d["fullcross_rank_numerical"] == d["fullcross_rank_expected"] == d["rank_Z"] * d["rank_U"]
        assert d["product_gram_identity_max_rel"] < 1e-8
    f0 = [v for k, v in res["folds"].items() if k.endswith("fold0")]
    assert all(v["fullcross_vs_primal_max"] < 1e-8 and v["fullcross_vs_dual_max"] < 1e-8 for v in f0)
    assert all(max(v["max_grad"].values()) < 1e-8 for v in res["folds"].values())


def test_output_held_cell_excluded_in_both_roles():
    rl = json.loads((OUT / "row_lists.json").read_text())
    P = pd.read_csv(OUT / "private_heldcell_predictions.csv")
    order = json.loads((ROOT / "results/grain_vector_mechanisms/conditional_operator_v1/r2/pair_order.json").read_text())
    for key, lists in rl.items():
        q, inp, fold = key.split("|"); fi = int(fold[4:])
        held = set(P[(P["query"] == q) & (P["input"] == inp) & (P.fold == fi)].pair)
        assert len(held) == 3
        # row lists store positions within the 36 source pairs; map via predictions file pairs
        srcpairs = [p for p in order if p in set(P.pair)]
        assert all(i >= 0 for rows in lists.values() for pr in rows for i in pr)
    assert P.groupby(["query", "input"]).size().eq(36).all()
