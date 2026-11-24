#!/usr/bin/env python3
"""Native parity and intervention contracts; no neural forward in these unit tests."""

import numpy as np
import pandas as pd
import pytest
from grain_core import liquid_instrument as li
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits


def texts():
    nouns = ("garden", "piano", "river", "paper", "bread", "wheel")
    return [
        f"I like the {a} and the {b}. Today the {a} changed."
        for a in nouns
        for b in nouns
    ]


@pytest.mark.parametrize("method", ["kmeans_tree", "gmm_soft"])
def test_source_nodes_match_published_geometry(method):
    frame = pd.DataFrame(
        {"slice_text": texts(), "user_id": [str(i % 6) for i in range(36)]}
    )
    frame["token_count"] = [len(li.native.tokenize(t)) for t in texts()]
    cfg = li.native.SuicaFactorConfig(
        method=method,
        svd_dims=16,
        tfidf_min_df=2,
        root_k=3,
        child_k=3,
        min_node_slices=4,
        gmm_k=3,
    )
    with threadpool_limits(limits=1):
        atlas = li.fit_atlas(texts(), 3, method)
        p, _ = li.locate(atlas, texts())
        _, x, _, _ = li.native.fit_slice_vectors(frame, cfg)
        if method == "gmm_soft":
            old = (
                GaussianMixture(
                    n_components=3,
                    covariance_type="diag",
                    max_iter=120,
                    reg_covar=1e-5,
                    random_state=42,
                )
                .fit(x)
                .predict_proba(x)
            )
            np.testing.assert_allclose(p[:, :-1], old, atol=1e-6)
        else:
            y, tab = li.native.build_tree_labels(x, frame, cfg)
            assert [atlas["labels"][j] for j in p.argmax(1)] == list(
                tab["node_id"].to_numpy()[y]
            )
        unseen, ok = li.locate(atlas, ["zzqqxxnonexistent"])
        assert not ok[0] and unseen[0, -1] == 1 and p[:, -1].sum() == 0


def test_cutters_keep_all_tokens_and_do_not_refit_on_application():
    text = " ".join(["green river"] * 48)
    a = li.cut(text, 0, {"kind": "rind", "terms": ["green"]})
    assert [len(v.split()) for v in a] == [48, 48]
    assert [len(v.split()) for v in li.cut(text, 16, {})] == [16] * 6
    items = [({}, [(0, 0, t) for t in texts()])]
    policy = li.fit_policy(items, "semantic")
    before = li.digest(policy)
    assert " ".join(li.cut(texts()[0], 0, policy)) == " ".join(
        li.native.tokenize(texts()[0])
    )
    assert before == li.digest(policy)


def test_signed_origin_account_and_missing_are_distinct_from_node_shares():
    p = np.array([[0.8, 0.2], [0.1, 0.9]])
    delta = np.array([[[0.2, -0.2], [0.04, -0.04]], [[0, 0], [-0.1, 0.1]]])
    obs = {
        "readout": p,
        "reset": p,
        "missing_effect": delta,
        "reverse_endpoint": p[0],
        "all_missing": p - delta.sum(0),
        "middle_swap_endpoint": None,
    }
    o = li.trajectory(p, obs)
    np.testing.assert_allclose(
        np.array(o["source_node_to_readout_node"]).sum(1), delta.sum(0)
    )
    np.testing.assert_allclose(o["joint_minus_sum"], 0, atol=1e-12)
    delta[1, 0, 0] = 0.5
    with pytest.raises(ValueError, match="future"):
        li.trajectory(p, obs)


def test_no_join_across_episode_boundaries():
    items = [({"author": "u", "view": "A", "condition": c}, []) for c in ("a", "b")]
    x = [np.array([[1.0, 0], [1.0, 0]]), np.array([[0.0, 1], [0.0, 1]])]
    stats = li.participant_features(items, x, ["a", "b", "c"])["u", "A"]
    np.testing.assert_allclose(stats["dynamic"][4:6], 0)
    np.testing.assert_allclose(stats["hybrid"][:4], [0.5, -0.5, -0.5, 0.5])
    assert np.isnan(stats["hybrid"][4:]).all()


def test_no_case_or_clock_ambiguity():
    e = {
        "id": "1",
        "author": "a",
        "carrier": "c",
        "condition": "c",
        "view": "A",
        "events": ["Plain text."],
    }
    d = {"episodes": [e, e], "language": "en", "clock": "opportunity"}
    with pytest.raises(ValueError, match="duplicate"):
        li.episodes(d, 0)
    d.update(episodes=[e], clock="seconds")
    with pytest.raises(ValueError, match="clock"):
        li.episodes(d, 0)
