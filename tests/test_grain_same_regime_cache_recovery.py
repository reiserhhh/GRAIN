#!/usr/bin/env python3
"""The exact export-type fault is tested without model or source data."""

import numpy as np
import pytest

from scripts.run_grain_same_regime_reader_hm274_recovery import cache_export_values


def test_kv_mapping_preserves_every_value_and_array_identity():
    k, v = np.zeros((2, 3), dtype=float), np.ones((2, 3), dtype=float)
    a, b = cache_export_values({"values": v, "keys": k})
    assert a is k and b is v
    np.testing.assert_array_equal(a - b, -np.ones((2, 3)))


@pytest.mark.parametrize(
    "exports",
    [
        {"keys": np.zeros((2, 3))},
        {"keys": np.zeros((2, 3)), "values": np.zeros((2, 4))},
        {"keys": np.zeros((2, 3)), "values": np.full((2, 3), np.nan)},
        {"keys": "keys", "values": "values"},
    ],
)
def test_bad_cache_export_is_not_silently_imputed(exports):
    with pytest.raises(ValueError):
        cache_export_values(exports)
