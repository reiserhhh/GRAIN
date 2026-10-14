#!/usr/bin/env python3
"""Common coordinate changes preserve functions but not bare intercepts/slopes."""

import numpy as np
import pytest

from grain_research.condition_coordinates import (
    reexpress_response,
    selective_repeat_expectation,
)


def test_function_and_reference_integral_are_preserved():
    rng = np.random.default_rng(3)
    a, b, c = rng.normal(size=5), rng.normal(size=(5, 3)), rng.normal(size=3)
    t = np.array([[2.0, 0.3, -0.1], [0.0, 1.0, 0.4], [0.0, 0.0, 0.6]])
    z = rng.normal(size=(100, 3))
    anew, bnew = reexpress_response(a, b, c, t)
    np.testing.assert_allclose(a + z @ b.T, anew + (c + z @ t.T) @ bnew.T, atol=1e-13)
    np.testing.assert_allclose(a + b @ z.mean(0), anew + bnew @ (c + t @ z.mean(0)))


def test_shift_mixes_slope_into_intercept():
    a, b = reexpress_response([0.0], [[0.3]], [-0.4], [[0.4]])
    np.testing.assert_allclose(a, [0.3])
    np.testing.assert_allclose(b, [[0.75]])


def test_singular_condition_map_rejected():
    with pytest.raises(ValueError, match="not invertible"):
        reexpress_response([0], [[1, 2]], [0, 0], [[1, 0], [0, 0]])


def test_scope_changes_only_context_input():
    import json

    from scripts.run_grain_condition_accumulation import ROOT, build_bank
    from scripts.run_grain_condition_scope import condition_spans

    bank = build_bank(
        json.loads((ROOT / "configs/grain_condition_accumulation.json").read_text())
    )
    spans = condition_spans(bank)
    assert spans.condition_text.nunique() == 32
    for row in spans.itertuples():
        assert row.full_text[row.start : row.end] == row.condition_text
    paired = bank.merge(spans, on="case_id").groupby(
        ["query", "partition", "condition", "context_variant"]
    )
    assert paired.condition_text.nunique().eq(1).all()


def test_selective_repetition_exact_enumeration():
    from itertools import product

    for p in (0.2, 0.5, 0.8):
        expected = 0.0
        for values in product((-1, 1), repeat=6):
            positives = sum(v == 1 for v in values)
            probability = p**positives * (1 - p) ** (6 - positives)
            reported_mean = (sum(values) + 18 * max(values)) / 24
            expected += probability * reported_mean
        np.testing.assert_allclose(
            selective_repeat_expectation(2 * p - 1), expected, atol=1e-14
        )
    assert selective_repeat_expectation(0.0) == pytest.approx(0.7265625)
