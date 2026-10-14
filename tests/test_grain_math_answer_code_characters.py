#!/usr/bin/env python3
"""Known raw-code/concept/mixed four-cell worlds for exact finite contrasts."""

import itertools

import pytest

from scripts.analyze_grain_math_answer_code_characters import characters


@pytest.mark.parametrize(
    "coefficients", [(0, 2, 0, 0), (0, 0, 2, 0), (1, 2, 3, 4), (0, 0, 0, 0)]
)
def test_full_code_characters_reconstruct(coefficients):
    a, b, c, d = coefficients
    cells = {
        (s, k): a + b * s + c * k + d * s * k
        for s, k in itertools.product((1, -1), repeat=2)
    }
    values = characters(cells)
    assert [
        values[k]
        for k in ("constant", "source_code", "receiver_code", "code_interaction")
    ] == list(coefficients)
    assert values["reconstruction_error"] == 0


def test_missing_cell_cannot_be_filled_with_marginals():
    with pytest.raises(ValueError):
        characters({(1, 1): 1, (1, -1): -1, (-1, 1): 1})
