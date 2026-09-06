"""Synthetic coverage of fixed dimension-matched projection controls."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.external_random_subspaces import (
    orthogonal_partition,
    random_subspace_grid,
)


def test_partition_is_complementary_deterministic_and_label_free():
    first, second = orthogonal_partition(48, 15, 94100)
    assert first.shape == (15, 48) and second.shape == (33, 48)
    joined = np.vstack([first, second])
    np.testing.assert_allclose(joined @ joined.T, np.eye(48), atol=1e-14)
    np.testing.assert_array_equal(first, orthogonal_partition(48, 15, 94100)[0])
    assert not np.allclose(first, orthogonal_partition(48, 15, 94101)[0])
    with pytest.raises(ValueError, match="nonempty"):
        orthogonal_partition(48, 48, 731)


def test_complete_fixed_random_condition_family():
    tasks = ["big5", "EI_cont", "SN_cont", "TF_cont", "JP_cont"]
    rows = random_subspace_grid(tasks)
    assert len(rows) == len({r["id"] for r in rows}) == 200
    assert {r["seed"] for r in rows} == set(range(94100, 94120))
    for task in tasks:
        for rank in (15, 33):
            assert sum(r["task"] == task and r["rank"] == rank for r in rows) == 20
