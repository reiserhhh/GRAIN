#!/usr/bin/env python3
"""Regression tests for the failed cold export stream separation."""

import numpy as np
import pytest

from scripts.complete_grain_expression_response_hm119 import append_batch


def test_distinct_batches_append_without_history_shadowing():
    """A shorter final batch produces compatible token/head output streams."""
    storage = {name: [] for name in ("k", "v", "readings")}
    for size in (8, 3):
        k = np.full((size, 2, 9, 2, 128), 1.0)
        v = k * 2
        r = np.ones((size, 2, 2, 6))
        append_batch(storage, k, v, r)
    assert np.stack(storage["k"]).shape == (11, 2, 9, 2, 128)
    assert np.stack(storage["v"]).shape == (11, 2, 9, 2, 128)
    assert np.stack(storage["readings"]).shape == (11, 2, 2, 6)
    assert np.all(np.stack(storage["v"]) == 2)


def test_mismatched_streams_are_not_committed():
    """Refuse a corrupted export before appending any output stream."""
    storage = {name: [] for name in ("k", "v", "readings")}
    with pytest.raises(ValueError):
        append_batch(
            storage,
            np.zeros((2, 2, 9, 2, 128)),
            np.zeros((3, 2, 9, 2, 128)),
            np.zeros((2, 2, 2, 6)),
        )
    assert all(not value for value in storage.values())
