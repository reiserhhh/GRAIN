#!/usr/bin/env python3
"""Source-query split, source exclusion, direct operator identity and error gain."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.source_common_counteraction import (
    compile_readers,
    direct_actions,
    select_contexts,
)
from grain_research.vector_mechanisms.source_query_correction import (
    apply_correction,
    error_account,
    query_readers,
)
from tests.test_grain_source_common_counteraction import fixture


@pytest.mark.parametrize("cohort,index", [("new8", 0), ("source4", 2)])
def test_query_and_common_compilation(cohort, index):
    bank, shared, d, weights = fixture()
    v, contexts, members = query_readers(bank, cohort, index, shared, weights)
    x, old_members = select_contexts(bank, cohort, index)
    p = apply_correction(v, d)
    assert members == old_members
    np.testing.assert_allclose(
        p["common_v"], compile_readers(x, shared, weights), atol=1e-12
    )
    np.testing.assert_allclose(
        p["contrast"], p["actions"][0] - p["actions"][1], atol=1e-12
    )
    for q in range(2):
        np.testing.assert_allclose(
            p["actions"][q],
            direct_actions(contexts[q], shared, d, weights)[q],
            atol=1e-11,
        )
    if cohort == "source4":
        bank["x"][bank["index"] == index] *= 1000
        other, _, _ = query_readers(bank, cohort, index, shared, weights)
        np.testing.assert_array_equal(v, other)


def test_identical_query_reader_has_zero_correction():
    _, shared, d, _ = fixture()
    p = apply_correction(np.repeat(shared[None], 2, axis=0), d)
    np.testing.assert_array_equal(p["correction"], 0)
    with pytest.raises(ValueError):
        apply_correction(np.zeros((2, 2, 7)), d[:, :2])


def test_exact_error_and_squared_error_change():
    r = np.random.default_rng(48)
    vals = r.normal(size=(5, 24))
    e = error_account(*vals)
    np.testing.assert_allclose(
        e["new_error"],
        e["source_error"] + e["query_error"] + e["parity_error"],
        atol=1e-14,
    )
    np.testing.assert_allclose(
        e["new_error"] ** 2 - e["old_error"] ** 2,
        e["mse_change_cross"] + e["mse_change_self"],
        atol=1e-14,
    )
