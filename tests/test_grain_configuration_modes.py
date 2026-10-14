#!/usr/bin/env python3
"""Level-clone decomposition must not invent independent dynamic evidence."""

from itertools import product

import numpy as np
import pytest

from grain_research.construction import math_configuration_kernel as K
from grain_research.construction import math_configuration_modes as M


def test_kernel_expansion_matches_explicit_tensor_blocks():
    rng = np.random.default_rng(14)
    s, a = rng.normal(size=(7, 3, 2)), rng.normal(size=(5, 3, 2))
    sm, am = s.mean(1), a.mean(1)
    ff = [s[:, j] @ a[:, j].T for j in range(3)]
    mf = [sm @ a[:, j].T for j in range(3)]
    fm = [s[:, j] @ am.T for j in range(3)]
    mm = [sm @ am.T] * 3
    blocks = M.tensor_blocks(M.component_grams(ff, mf, fm, mm))
    ts, ta = [], []
    for bits in product((0, 1), repeat=3):
        for x, mean, tensors in ((s, sm, ts), (a, am, ta)):
            args = [x[:, j] - mean if bit else mean for j, bit in enumerate(bits)]
            tensors.append(
                np.einsum("ni,nj,nk->ijk", *[v - v.mean(0) for v in args]) / len(x)
            )
    direct = np.array([[(x * y).sum() for y in ta] for x in ts])
    assert np.allclose(blocks, direct)
    assert blocks.sum() == pytest.approx(K.interaction_inner(ff))


def test_all64_cells_group_into_four_signed_cells():
    rng = np.random.default_rng(15)
    s, a = rng.normal(size=(12, 3, 4)), rng.normal(size=(9, 3, 4))
    grams = M.substitution_kernels(s, a, [1, 2, 3])
    blocks = M.tensor_blocks(M.component_grams(*grams))
    grouped = M.grouped_cells(*grams)
    assert np.allclose(
        grouped,
        [
            [blocks[0, 0], blocks[0, 1:].sum()],
            [blocks[1:, 0].sum(), blocks[1:, 1:].sum()],
        ],
    )


def test_constant_parcels_have_no_departure_component():
    rng = np.random.default_rng(16)
    s = np.repeat(rng.normal(size=(8, 1, 3)), 3, axis=1)
    a = np.repeat(rng.normal(size=(7, 1, 3)), 3, axis=1)
    cells = M.grouped_cells(*M.substitution_kernels(s, a, [1, 2, 3]))
    assert np.max(abs(cells.ravel()[1:])) < 1e-12


def test_signed_components_are_not_information_percentages():
    rng = np.random.default_rng(17)
    s, a = rng.normal(size=(16, 3, 3)), rng.normal(size=(12, 3, 3))
    blocks = M.tensor_blocks(
        M.component_grams(*M.substitution_kernels(s, a, [1, 1, 1]))
    )
    assert blocks.min() < 0 and blocks.max() > 0
