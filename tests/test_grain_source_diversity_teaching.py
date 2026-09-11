#!/usr/bin/env python3
"""Disjoint selection, source-only fixed teaching and publication metadata tests."""

import numpy as np
import pandas as pd
import pytest

from grain_research.vector_mechanisms import source_diversity_teaching as f


def material():
    """Synthetic24-author view0, four sources and eight recipients."""
    rows = [
        {
            "author_id": str(i),
            "source_group": "mbti_only" if i % 2 else "big5_only",
            "view": 0,
            "unread_label": i,
        }
        for i in range(24)
    ]
    return rows, {str(i) for i in range(4)}, {str(i) for i in range(4, 12)}


def test_selection_disjoint_balanced_and_label_order_invariant():
    rows, sources, recipients = material()
    chosen, counts = f.select(rows, sources, recipients)
    assert counts == {"mbti_only": 6, "big5_only": 6}
    assert len(chosen) == 8
    assert not {r["author_id"] for r in chosen} & (sources | recipients)
    changed = [r | {"unread_label": -999} for r in reversed(rows)]
    other, _ = f.select(changed, sources, recipients)
    assert [r["author_id"] for r in chosen] == [r["author_id"] for r in other]
    assert sum(r["source_group"] == "mbti_only" for r in chosen) == 4


def test_membership_errors_do_not_silently_replace_authors():
    rows, sources, recipients = material()
    for data, s, r in (
        (rows + rows[:1], sources, recipients),
        (rows[:-1], sources, recipients),
        (rows, sources, recipients | {"0"}),
    ):
        with pytest.raises(ValueError):
            f.select(data, s, r)


def model_fixture():
    """Synthetic fixed original model and eight additional input positions."""
    rng = np.random.default_rng(880)
    old = {
        "x": rng.normal(size=(28, 7)),
        "weights": np.ones(28) * 3,
        "regularizer": 0.02,
        "units": np.arange(1, 6.0),
        "fixed_frame": rng.normal(size=(5, 9)),
    }
    points = np.concatenate((old["x"][f.previous.ORIGINAL], rng.normal(size=(8, 7))))
    return rng, old, points, rng.normal(size=(12, 5))


def test_matched12_solve_and_target_linearity():
    _, old, x, y = model_fixture()
    m = f.fit(old, x, y)
    assert m["coefficients"] == 60
    assert m["normal_residual_relative"] < 1e-10
    np.testing.assert_array_equal(m["weights"], np.full(12, 3))
    np.testing.assert_array_equal(m["units"], old["units"])
    np.testing.assert_allclose(f.fit(old, x, 2 * y)["beta"], 2 * m["beta"], atol=1e-10)
    k = f.previous.base.gram(x, m["v"], m["kinds"], x, m["v"], m["kinds"])
    np.testing.assert_allclose((k + np.eye(12) * 0.02 / 9) @ m["beta"], y, atol=1e-10)


def test_reject_partial_or_reordered_or_nonfinite_teaching():
    _, old, x, y = model_fixture()
    with pytest.raises(ValueError):
        f.fit(old, x[:8], y[:8])
    with pytest.raises(AssertionError):
        f.fit(old, x[::-1], y)
    with pytest.raises(ValueError):
        f.fit(old, x, y * np.nan)


def test_expanded_field_derivative():
    rng, old, x, y = model_fixture()
    m = f.fit(old, x, y)
    a, d = rng.normal(size=(2, 7))
    eps = 1e-5
    _, grad = f.previous.base.field(m, a, d)
    values = [f.previous.base.field(m, a + s * eps * d) for s in (-1, 1)]
    np.testing.assert_allclose(grad[0], (values[1] - values[0]) / (2 * eps), atol=1e-8)


def test_format_adapter_preserves_original4_names_in_old_rows():
    from scripts.run_grain_source_diversity_numeric import adapt_tables

    before = pd.DataFrame({"variant": list(f.previous.VARIANTS), "authors": [8, 8]})
    tables = [pd.concat((before, before), ignore_index=True) for _ in range(6)]
    tables += [before.copy() for _ in range(4)]
    tables[7]["observations"] = 4
    adapted = adapt_tables(tables, [2] * 6)
    for t in adapted[:6]:
        pd.testing.assert_frame_equal(t.iloc[:2].reset_index(drop=True), before)
        assert tuple(t.iloc[2:].variant) == f.VARIANTS
    assert set(adapted[6].authors) == {12}
    assert set(adapted[7].observations) == {12}
