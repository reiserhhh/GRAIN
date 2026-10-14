#!/usr/bin/env python3
"""Exact parity identities without model calls."""

import pytest

from scripts.analyze_grain_math_evidence_response_parity import parity, summarize


@pytest.mark.parametrize("native,plus,minus", [(6, 2, 2), (3, 4, 2), (-2, 1, -1)])
def test_exact_parity_reconstructs_responses(native, plus, minus):
    row = parity(native, plus, minus)
    assert row["reconstruction_error"] < 1e-12
    assert native + row["even_action"] + row["odd_action"] == pytest.approx(plus)
    assert native + row["even_action"] - row["odd_action"] == pytest.approx(minus)


def test_common_compression_is_even_not_odd():
    row = parity(6, 2, 2)
    assert row["even_action"] == -4
    assert row["odd_action"] == 0
    assert summarize([row])["both_signs_smaller"] == 1
