#!/usr/bin/env python3
"""Source exclusion, adjoint equivalence and cancellation in counteraction errors."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.source_common_counteraction import (
    apply_readers,
    compile_readers,
    direct_actions,
    errors,
    select_contexts,
)


def fixture():
    """Small non-isotropic paired contexts with no model or private data dependency."""
    r = np.random.default_rng(47)
    h, k = 7, 11
    w = {
        "gate": r.normal(size=(k, h)),
        "up": r.normal(size=(k, h)),
        "down": r.normal(size=(h, k)),
        "gamma": r.uniform(0.1, 2, h),
        "epsilon": 0.02,
    }
    bank = {
        "x": r.normal(size=(8, h)),
        "index": np.repeat(np.arange(4), 2),
        "query": np.tile(["neuroticism", "confirmation_seeking"], 4),
    }
    return bank, r.normal(size=(2, h)), r.normal(size=(2, 4, h)), w


@pytest.mark.parametrize("cohort,index", [("source4", 0), ("source4", 3), ("new8", 7)])
def test_source_common_compilation(cohort, index):
    bank, shared, d, w = fixture()
    x, members = select_contexts(bank, cohort, index)
    assert len(x) == (6 if cohort == "source4" else 8)
    if cohort == "source4":
        assert index not in members
    v = compile_readers(x, shared, w)
    actions, contrast = apply_readers(v, d)
    np.testing.assert_allclose(actions, direct_actions(x, shared, d, w), atol=1e-11)
    np.testing.assert_allclose(
        contrast, np.einsum("mh,fh->mf", v, d[0] - d[1]), atol=1e-11
    )
    if cohort == "source4":
        bank["x"][bank["index"] == index] += 1e6
        same, _ = select_contexts(bank, cohort, index)
        np.testing.assert_array_equal(compile_readers(same, shared, w), v)


def test_error_identity_retains_native_parity():
    p, c, o, n = [
        np.array(x)
        for x in (
            [1.0, 4.0, -2.0],
            [2.0, 1.0, -4.0],
            [-3.0, 2.0, 2.0],
            [-0.9, 3.0, -2.0],
        )
    ]
    e = errors(p, c, o, n)
    np.testing.assert_allclose(
        e["source_error"] + e["missing_operator"] + e["parity_remainder"],
        e["total_error"],
    )
    assert e["parity_remainder"][0] != 0


def test_invalid_source_inventory_and_runtime_schema():
    bank, shared, d, _ = fixture()
    bank["index"][0] = 1
    with pytest.raises(ValueError):
        select_contexts(bank, "new8", 0)
    with pytest.raises(ValueError):
        apply_readers(shared, d[:, :3])
