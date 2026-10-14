"""Focused tests for packet23 (R08 raw-vector / VM05 GRAIN history comparators). Pre-run + post-run test_output_ tests."""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import r08_raw_grain_history as g  # noqa: E402
from grain_research.vector_mechanisms.external_material import raw_author_features  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/F14/raw_grain_history_r1"
PROTOCOL = ROOT / "docs/GRAIN_R08_RAW_GRAIN_HISTORY_PROTOCOL_V1.md"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_affine_ridge_normal_equations():
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(300, 60)), rng.normal(size=(300, 30)) + 2.0
    sc = g.fit_scaler(X); Xs = g.apply_scaler(X, sc)
    B, b = g.fit_affine_ridge(Xs, Y)
    gB, gb = g.ridge_gradients(Xs, Y, B, b)
    assert gB < 1e-12 and gb < 1e-12


def test_zero_sd_column_handled():
    X = np.c_[np.random.default_rng(1).normal(size=(50, 3)), np.ones(50)]
    sc = g.fit_scaler(X)
    assert sc["zero_sd"].tolist() == [False, False, False, True] and np.all(g.apply_scaler(X, sc)[:, 3] == 0)


def test_named_mean_std_equals_upstream_raw_author_features():
    names = json.loads((g.PRE / "manifest.json").read_text())["feature_names"]
    u = np.random.default_rng(2).normal(size=(5, 36, 24))
    assert np.allclose(g.named_mean_std(u, names), raw_author_features(u, names), atol=1e-15)


def test_prefix_features_ignore_future_and_static_is_permutation_invariant():
    rng = np.random.default_rng(3)
    P = rng.normal(size=(36, 30)); names = json.loads((g.PRE / "manifest.json").read_text())["feature_names"]; G = rng.normal(size=(36, 24))
    f0 = [g.feat_raw_ordered(P[:t]) for t in range(25, 36)]; s0 = [g.feat_grain_static(G[:t], names) for t in range(25, 36)]
    P2, G2 = P.copy(), G.copy(); P2[30:] += 9; G2[30:] += 9
    for i, t in enumerate(range(25, 36)):
        same = np.array_equal(f0[i], g.feat_raw_ordered(P2[:t])) and np.array_equal(s0[i], g.feat_grain_static(G2[:t], names))
        assert same == (t <= 30)
    pre = P[:27]; o = g.permuted_order(27, 2, 11)
    assert np.allclose(g.feat_raw_static(pre[o]), g.feat_raw_static(pre), atol=1e-12)
    assert np.array_equal(o, g.permuted_order(27, 2, 11))


def test_keyed_from_raw_formula_and_pca_determinism():
    rng = np.random.default_rng(4)
    V = rng.normal(size=(40, 1024)).astype(np.float32); D = rng.normal(size=(1024, 30))
    k = g.keyed_from_raw(V, D)
    assert np.allclose(k, (V.astype(float) / np.linalg.norm(V.astype(float), axis=1, keepdims=True)) @ D)
    p1, p2 = g.fit_pca(g.unit_normalise(V)), g.fit_pca(g.unit_normalise(V))
    assert np.array_equal(p1["W"], p2["W"])


# ------------------------------------------------------------------ outputs
def test_output_gates_and_arms():
    res = json.loads((OUT / "results.json").read_text())
    assert res["protocol_sha256"] == sha(PROTOCOL)
    for rel, hh in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        assert sha(ROOT / rel) == hh
    a1 = res["A1_raw_bridge"]
    assert a1["rows_compared"] == 22464 and (a1["rows_over_tol"] == 0) == ("RAW_STATIC" in res["arms_executed"])
    assert res["A3_representation_audit"]["clean"] == ("GRAIN_STATIC" in res["arms_executed"] or "GRAIN_STATIC" in res["arms_not_executed"])
    assert set(res["arms_executed"]) | set(res["arms_not_executed"]) == {"KEYED_STATIC", "RAW_STATIC", "GRAIN_STATIC", "RAW_ORDERED"}
    assert res["neural_calls"] == 0 and res["cold_replay_max_abs"] <= 1e-12
    rr = res["reference_replay_vs_packet22"]
    assert rr["rows"] == 1320 and max(rr["prefix_mean_max"], rr["coupled_hist_max"]) <= 1e-12
    for f in res["fits"].values():
        assert f["max_grad_B"] < 1e-10 and f["max_grad_b"] < 1e-10 and f["rows"] == 4433


def test_output_private_files_and_pairing():
    for f in OUT.glob("*"):
        if f.name.startswith("._") or f.name == "common_record_r1":
            continue
        assert f.name.startswith("private_") or f.suffix == ".json" or f.name == "decision.md" or f.name.endswith(".md")
    H = pd.read_csv(OUT / "private_held_event_losses.csv")
    assert len(H) == 1320 and H.groupby("author").size().eq(11).all()
