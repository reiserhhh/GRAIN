#!/usr/bin/env python3
"""Source-only sensitivity views cannot turn application gains into rank selection."""

import numpy as np
import pytest

from scripts.run_grain_math_compensation_resolution import prefix_view


def fixture():
    """A small source bank with one unresolved tail and a matched control."""
    meta = {
        "role": "source",
        "application_nodes_loaded": 0,
        "rank": 3,
        "resolved_against_proxy": 2,
    }
    bank = {
        "basis": np.eye(6)[:, :3],
        "random_basis": np.eye(6)[:, 3:],
        "resolved_against_proxy": np.array([True, True, False]),
    }
    return meta, bank


def test_prefix_is_source_fixed_and_dimension_matched():
    meta, bank = fixture()
    w, r, rank = prefix_view(meta, bank)
    assert rank == 2 and w.shape == r.shape == (6, 2)
    np.testing.assert_array_equal(w, bank["basis"][:, :2])


def test_application_resolution_not_allowed():
    meta, bank = fixture()
    with pytest.raises(ValueError, match="source-only"):
        prefix_view({**meta, "role": "application"}, bank)


def test_noncontiguous_or_selected_rank_is_rejected():
    meta, bank = fixture()
    with pytest.raises(ValueError, match="contiguous"):
        prefix_view(
            meta, {**bank, "resolved_against_proxy": np.array([True, False, True])}
        )
    with pytest.raises(ValueError, match="contiguous"):
        prefix_view({**meta, "resolved_against_proxy": 1}, bank)
