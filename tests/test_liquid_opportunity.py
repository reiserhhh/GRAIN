#!/usr/bin/env python3
import numpy as np
import pytest
from grain_core import factor_discovery as fd
from grain_core import liquid_instrument as li
from grain_core import liquid_opportunity as op


def example(n, conditions=5, seed=16):
    rng = np.random.default_rng(seed)
    ref = {"source_digest": "fixture"}
    labels = list("abcd")
    out = {
        "input_digest": "fixture",
        "reference_digest": li.digest(ref),
        "node_names": labels,
        "episodes": {},
    }
    means = rng.normal(size=(conditions, len(labels)))
    for i in range(n):
        author = rng.normal(scale=0.15, size=len(labels))
        for v in ("A", "B"):
            for c in range(conditions):
                logits = (
                    means[c] + author + rng.normal(scale=0.1, size=(4, len(labels)))
                )
                x = np.exp(logits)
                x /= x.sum(1, keepdims=True)
                out["episodes"][f"{v}-{i}-{c}"] = {
                    "author": str(i),
                    "view": v,
                    "condition": f"c{c}",
                    "slots": [[j, 0] for j in range(4)],
                    **{k: x.tolist() for k in op.CHANNELS},
                }
    return ref, out


def test_cold_response_slopes_match_original_on_eligible_design():
    r, src = example(20)
    ref = op.fit(src, r)
    assert ref["status"] == "fitted"
    _, held = example(2, seed=10)
    held["input_digest"] = "held"
    d = op.score(held, ref)
    f = op.frame(src, "native")
    m = fd.fit_opportunity_model(f, src["node_names"], set(f.user_id))
    legacy = fd.build_family_features(op.frame(held, "native"), m)
    for fam in ("static", "dynamic", "hybrid"):
        gold = legacy[fam].iloc[0]
        for k, v in d["channels"]["native"][fam][0]["features"].items():
            np.testing.assert_allclose(v, gold[k], atol=1e-12)
    assert len(d["channels"]["liquid"]["hybrid"][0]["features"]) == 12
    with pytest.raises(ValueError, match="original source"):
        op.fit(held, r)


def test_insufficient_support_is_an_explicit_result():
    r, src = example(4, 3)
    ref = op.fit(src, r)
    assert ref["status"] == "insufficient_source"
    assert op.score(src, ref)["source_authors_per_condition"] == {
        "c0": 4,
        "c1": 4,
        "c2": 4,
    }
    src["node_names"] = list("efgh")
    with pytest.raises(ValueError, match="different reader"):
        op.score(src, ref)
