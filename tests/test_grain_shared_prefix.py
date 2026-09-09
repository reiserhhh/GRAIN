#!/usr/bin/env python3
"""Exact shared-prefix grouping must not merge target-specific contexts."""

import pytest

from grain_research.vector_mechanisms.shared_prefix import context_groups


def test_context_group_keeps_different_system_prefix_separate():
    rows = [
        {"input_ids": [1, 2, 3, 10, 20], "mask": [False, False, False, True, False]},
        {
            "input_ids": [1, 2, 3, 11, 12, 20],
            "mask": [False, False, False, True, True, False],
        },
        {"input_ids": [1, 8, 3, 10, 20], "mask": [False, False, False, True, False]},
    ]
    groups = context_groups(rows)
    assert [len(g) for _, g in groups] == [2, 1]
    assert groups[0][0] == [1, 2, 3]


def test_invalid_cache_partition_rejected():
    with pytest.raises(ValueError):
        context_groups([{"input_ids": [1, 2], "mask": [True, True]}])
    with pytest.raises(ValueError):
        context_groups([{"input_ids": [1, 2], "mask": [True, False]}])
