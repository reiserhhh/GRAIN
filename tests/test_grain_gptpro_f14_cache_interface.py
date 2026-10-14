#!/usr/bin/env python3
"""Regression tests for checks previously declared True without execution."""

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3.f14_cache_interface import check_interface


def data():
    """Two-author aligned synthetic numeric cache, with no text values."""
    arrays = {k: np.ones((4, 2)) for k in ("g", "h", "semantic", "scores")}
    arrays.update(
        authors=np.array([0, 0, 1, 1]), reference_authors=np.array([0, 0, 1, 1])
    )
    addresses = np.array([[0, 0, 0, 10], [0, 1, 10, 20], [1, 0, 0, 10], [1, 1, 10, 20]])
    return arrays, addresses


def test_valid_and_same_author_wrong_address():
    arrays, left = data()
    assert check_interface(arrays, left, left, [0], [1], 2)["units"] == 4
    wrong = left.copy()
    wrong[[0, 1]] = wrong[[1, 0]]
    with pytest.raises(ValueError, match="material address mismatch"):
        check_interface(arrays, left, wrong, [0], [1], 2)


def test_nonfinite_actual_values():
    arrays, addresses = data()
    arrays["g"][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        check_interface(arrays, addresses, addresses, [0], [1], 2)


@pytest.mark.parametrize("source,held", [([0], [0, 1]), ([0], []), ([0, 0], [1])])
def test_partitions(source, held):
    arrays, addresses = data()
    with pytest.raises(ValueError):
        check_interface(arrays, addresses, addresses, source, held, 2)


def test_duplicate_and_empty_span():
    arrays, addresses = data()
    addresses[1] = addresses[0]
    with pytest.raises(ValueError, match="duplicate material"):
        check_interface(arrays, addresses, addresses, [0], [1], 2)
    arrays, addresses = data()
    addresses[0, 3] = addresses[0, 2]
    with pytest.raises(ValueError, match="empty span"):
        check_interface(arrays, addresses, addresses, [0], [1], 2)
