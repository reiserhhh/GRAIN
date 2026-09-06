"""Outcome-independent, dimension-matched orthogonal partition controls."""

import numpy as np


def orthogonal_partition(dimensions, first_rank, seed):
    """Return complementary row bases from a seeded Haar orthogonal matrix."""
    if not 0 < first_rank < dimensions:
        raise ValueError("Require two nonempty complementary subspaces")
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(dimensions, dimensions)))
    signs = np.where(np.diag(r) < 0, -1, 1)
    q *= signs
    return q[:, :first_rank].T, q[:, first_rank:].T


def random_subspace_grid(tasks):
    """Fix every seed/rank/task cell; never search for a winning projection."""
    return [
        {
            "id": f"{task}_seed{seed}_rank{rank}",
            "task": task,
            "seed": seed,
            "rank": rank,
        }
        for task in tasks
        for seed in range(94100, 94120)
        for rank in (15, 33)
    ]
