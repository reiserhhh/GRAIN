#!/usr/bin/env python3
"""Reproduce the full-mask false rejection without changing scientific cuts."""

import numpy as np

from grain_research.construction import math_native_access as M
from scripts.recover_grain_native_access_hm131 import suffix_support


def test_different_body_padding_same_all_off_suffix_support():
    a = {
        "input_ids": list(range(8)),
        "history": [2, 3, 4],
        "current": [7],
        "padding": [4],
    }
    b = {**a, "padding": []}
    assert not np.array_equal(M.permissions(a, 3, 11, 0), M.permissions(b, 3, 11, 0))
    np.testing.assert_array_equal(
        suffix_support(a, [[0, 1, 2]], 11), suffix_support(b, [[0, 1, 2]], 11)
    )


def test_recovery_does_not_hide_changed_question_permissions():
    row = {
        "input_ids": list(range(8)),
        "history": [2, 3, 4],
        "current": [7],
        "padding": [],
    }
    off = suffix_support(row, [[0, 1, 2]], 11)
    altered = off.copy()
    altered[0, 1, 2] = True
    assert not np.array_equal(off, altered)
