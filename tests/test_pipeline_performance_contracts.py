#!/usr/bin/env python3
"""Independent numerical and cohort contracts for the fast paths; no timing thresholds."""

import numpy as np
import pytest
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits

from examples.handbook_study import corpus
from grain_core import factor_discovery as fd
from grain_core import liquid_instrument as li
from grain_core import riccati_mobius as rm


@pytest.mark.parametrize("method", ["gmm_soft", "kmeans_tree"])
def test_batched_native_paths_match_single_rows_across_boundaries(method):
    items = li.episodes(corpus("discovery", 30), 0)
    texts = [r[2] for _, rows in items for r in rows]
    with threadpool_limits(limits=1):
        atlas = li.fit_atlas(texts, 4, method)
        items[85][1][1] = (1, 0, "zzunknownwordzz")  # adjacent to batch boundary
        paths = li.native_paths(atlas, items)
        assert [len(p) for p in paths] == [len(rows) for _, rows in items]
        for idx in [0, 84, 85, 86, -1]:
            expected = np.vstack(
                [li.locate(atlas, [t])[0] for _, _, t in items[idx][1]]
            )
            np.testing.assert_allclose(paths[idx], expected, atol=1e-13)
        assert paths[85][1][-1] == 1


def test_source_attribution_matches_unflattened_index_contraction():
    rng = np.random.default_rng(5)
    p = rng.dirichlet(np.ones(7), size=23)
    effect = rng.normal(size=(23, 23, 7))
    i, j = np.tril_indices(23, -1)
    effect[i, j] = 0
    obs = {
        "readout": p,
        "reset": p,
        "missing_effect": effect,
        "all_missing": p,
        "reverse_endpoint": p[0],
        "middle_swap_endpoint": None,
    }
    signed = li.trajectory(p, obs)["source_node_to_readout_node"]
    np.testing.assert_allclose(signed, np.einsum("ka,ktb->tab", p, effect), atol=1e-13)


def test_reset_memo_preserves_replicates_and_distinct_order(monkeypatch):
    first, second = [[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]

    def fake(req, transport):
        assert req["paths"] == [first, second, first[:1]]
        assert req["mode"] == "score"
        return {"paths": ["ab", "ba", "a"]}

    monkeypatch.setattr(li, "call", fake)
    assert li._score_reset_paths([first, second, first[:1], first], {}, "local") == [
        "ab",
        "ba",
        "a",
        "ab",
    ]


def test_worker_batch_is_not_a_statistical_batch(monkeypatch):
    doc = corpus("discovery", 8)
    texts = [text for e in doc["episodes"] for text in e["events"]]
    calls = []

    def fake(req, transport):
        calls.append(transport)
        out = []
        for p in req["paths"]:
            n, d = np.shape(p)
            out.append(
                {
                    "readout": p,
                    "reset": p,
                    "missing_effect": np.zeros((n, n, d)).tolist(),
                    "reverse_endpoint": p[0],
                    "middle_swap_endpoint": None,
                    "all_missing": p,
                }
            )
        return {"paths": out}

    monkeypatch.setattr(li, "call", fake)
    with threadpool_limits(limits=1):
        ref = {
            "version": li.VERSION,
            "worker": li.digest(li.WORKER.read_text()),
            "width": 0,
            "conditions": ["departure", "detour"],
            "atlas": li.fit_atlas(texts, 4, "gmm_soft"),
            "weights": {},
            "axes": {
                s: {} for s in ("native", "fixed_half", "liquid", "anchored_signed")
            },
        }
    # Both IDs and authors may recur across documents. Legacy batch-relative
    # outputs must still be calculated on each document's own cohort.
    docs = [
        dict(doc, episodes=doc["episodes"][:8]),
        dict(doc, episodes=doc["episodes"][4:12]),
    ]
    expected = [li.run("score", x, ref, "local") for x in docs]
    calls.clear()
    assert li.score_batch(docs, ref) == expected
    assert calls == ["local"]
    huge = dict(
        doc,
        episodes=[
            dict(doc["episodes"][0], id=str(i), events=["river"] * 128)
            for i in range(30)
        ],
    )
    with pytest.raises(ValueError, match="budget"):
        li.score_batch([huge], ref)
    assert calls == ["local"]


def dense_oracle(y, z, rates, budget, w):
    n = len(y)
    eye = np.eye(n)
    a, b = [], []
    for t in range(n - 1):
        for s, r, e in zip((np.zeros(n), y, z), rates, budget, strict=True):
            row = eye[t + 1] - r * eye[t]
            value = s[t + 1] - r * s[t]
            a.extend([row, -row])
            b.extend([e + value, e - value])
    if w is not None:
        for row, mid, hw in zip(eye, (y + z) / 2, w, strict=True):
            a.extend([row, -row])
            b.extend([mid + hw, -mid + hw])
    ans = []
    for sign in (1, -1):
        results = [
            linprog(
                sign * row,
                A_ub=np.array(a) if a else None,
                b_ub=np.array(b) if b else None,
                bounds=[(-2, 2)] * n,
                method="highs",
            )
            for row in eye
        ]
        if any(r.status == 2 for r in results):
            return None
        assert all(r.success for r in results)
        ans.append([sign * r.fun for r in results])
    return np.array(ans)


@pytest.mark.parametrize("n", [1, 4, 15])
@pytest.mark.parametrize("has_width", [False, True])
def test_sparse_polytope_matches_independent_dense_oracle(n, has_width):
    rng = np.random.default_rng(17 + n)
    y, z = rng.normal(0, 0.12, (2, n))
    rates = (-0.3, 0.0, 0.8)
    for budget in ((0.4, 0.5, 0.6), (0.0, 0.0, 0.0)):
        w = np.full(n, 0.6) if has_width else None
        expected = dense_oracle(y, z, rates, budget, w)
        for last in (False, True):
            actual = rm.solve_axis_trajectory_lp(
                y, z, *rates, *budget, w, c_bound=2.0, endpoint_only=last
            )
            if expected is None:
                assert actual is None
            else:
                np.testing.assert_allclose(
                    actual, expected[:, -1:] if last else expected, atol=1e-9
                )


def test_spectrum_acceleration_preserves_null_and_strict_ties(monkeypatch):
    rng = np.random.default_rng(4)
    a = rng.normal(size=(32, 9))
    b = a * 0.6 + rng.normal(size=a.shape)
    for strata in (np.zeros(32, int), np.arange(32)):
        fast = fd.fit_stable_crossview(a, b, n_permutations=23, strata=strata)
        with monkeypatch.context() as m:
            m.setattr(np.linalg, "eigvalsh", lambda op: np.linalg.eigh(op)[0])
            oracle = fd.fit_stable_crossview(a, b, n_permutations=23, strata=strata)
        assert fast.n_factors == oracle.n_factors
        np.testing.assert_allclose(
            fast.permutation_thresholds, oracle.permutation_thresholds, atol=1e-12
        )
        np.testing.assert_array_equal(
            fast.rotated_directions, oracle.rotated_directions
        )
        if len(np.unique(strata)) == len(a):
            assert fast.n_factors == 0
