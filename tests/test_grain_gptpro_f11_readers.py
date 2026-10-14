"""Fixed reader variance decomposition and degenerate correlation behavior."""

import numpy as np

from scripts.run_grain_gptpro_f11_reader_interface import (
    correlation,
    crossed_decomposition,
)


def test_constant_reader_shift_is_not_material_interaction():
    y = np.column_stack([np.arange(10), np.arange(10) + 4])
    parts = crossed_decomposition(y)
    np.testing.assert_allclose(parts, [8.25, 4, 0])


def test_material_dependent_reader_disagreement():
    x = np.arange(-5, 6, dtype=float)
    parts = crossed_decomposition(np.column_stack([x, -x]))
    np.testing.assert_allclose(parts, [0, 0, np.mean(x * x)])


def test_degenerate_correlation_remains_missing():
    assert correlation(np.ones(10), np.arange(10)) is None
