#!/usr/bin/env python3
"""Input-program contracts and a bag-order non-identification witness."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.configuration_extraction import (
    ARMS,
    InputProgram,
    query_pool,
    token_features,
)


@pytest.mark.parametrize("arm", ARMS)
def test_reference_and_single_row_scores_do_not_depend_on_test_batch(arm):
    rng = np.random.default_rng(52)
    bags = rng.normal(size=(12, 5, 6))
    token = rng.normal(size=(12, 2, 8))
    y = np.column_stack([bags[:, :, 0].mean(1), token[:, 0, 0]])
    model = InputProgram(arm, rng.normal(size=(2, 6)), width=16).fit(bags, token, y)
    centers = [c.copy() for c in model.centers]
    batch = model.predict(bags[10:], token[10:])
    solo = model.predict(bags[10:11], token[10:11])
    np.testing.assert_allclose(batch[0], solo[0], atol=1e-12)
    for before, after in zip(centers, model.centers, strict=True):
        np.testing.assert_array_equal(before, after)


def test_token_baseline_is_an_explicit_weighted_lookup():
    table = np.array([[1.0, 0], [0, 1.0], [-1.0, 0]])
    got = token_features([0, 1], table, [0])
    np.testing.assert_allclose(got[0], [0.5, 0.5])
    assert got[1, 0] > 0.99
    with pytest.raises(ValueError):
        token_features([3], table, [0])


def test_fixed_weighted_pool_keeps_ordered_pole_direction_distinct():
    v = np.array([[1.0, 0], [-1.0, 0]])
    np.testing.assert_allclose(query_pool(v, [1.0, 0], 8, absolute=True), [0, 0])
    assert query_pool(v, [1.0, 0], 8, absolute=False)[0] > 0.99


def test_kernel_bag_can_distinguish_equal_means_but_not_order():
    poles = np.eye(2)
    train = np.array(
        [[[1.0, 0], [-1.0, 0]], [[0, 1.0], [0, -1.0]], [[2.0, 0], [-2.0, 0]]]
    )
    targets = np.column_stack([np.arange(3), np.arange(3)])
    model = InputProgram("e5_distribution", poles, width=64).fit(train, None, targets)
    blocks = model.raw_blocks(train, None)
    np.testing.assert_allclose(blocks[0][0], blocks[0][1])
    assert np.linalg.norm(blocks[1][0] - blocks[1][1]) > 0.01
    reversed_blocks = model.raw_blocks(train[:, ::-1], None)
    np.testing.assert_allclose(blocks[1], reversed_blocks[1])
    # An order-sensitive target has different values at these identical inputs.
    assert train[0, 0, 0] != train[0, -1, 0]


def test_identical_seed_and_input_produce_identical_program():
    rng = np.random.default_rng(21)
    bags = rng.normal(size=(10, 4, 5))
    y = rng.normal(size=(10, 2))
    poles = rng.normal(size=(2, 5))
    a = InputProgram("e5_distribution", poles, width=8).fit(bags, None, y)
    b = InputProgram("e5_distribution", poles, width=8).fit(bags, None, y)
    np.testing.assert_array_equal(a.predict(bags, None), b.predict(bags, None))


def test_instance_functional_exactly_reconstructs_levels_and_removal():
    rng = np.random.default_rng(13)
    bags = rng.normal(size=(12, 5, 6))
    y = rng.normal(size=(12, 2))
    poles = rng.normal(size=(2, 6))
    model = InputProgram("e5_mean", poles).fit(bags, None, y)
    weights, offset = model.linear_instance_map()
    all_scores = bags @ weights.T
    np.testing.assert_allclose(
        all_scores.mean(1) + offset, model.predict(bags, None), atol=1e-12
    )
    removed = [0, 2]
    keep = [1, 3, 4]
    identity = (len(removed) / 5) * (
        all_scores[:, keep].mean(1) - all_scores[:, removed].mean(1)
    )
    actual = model.predict(bags[:, keep], None) - model.predict(bags, None)
    np.testing.assert_allclose(identity, actual, atol=1e-12)


def test_direct_residual_readout_is_a_difference_of_same_ridge_programs():
    rng = np.random.default_rng(141)
    bags = rng.normal(size=(24, 5, 6))
    poles = rng.normal(size=(2, 6))
    c = rng.normal(size=24)
    grain = rng.normal(size=24)
    direct = InputProgram("e5_mean", poles).fit(
        bags, None, np.column_stack([c, c - grain])
    )
    projected_grain = InputProgram("e5_mean", poles).fit(
        bags, None, np.column_stack([grain, grain])
    )
    p = direct.predict(bags, None)
    pg = projected_grain.predict(bags, None)[:, 0]
    np.testing.assert_allclose(p[:, 0] - p[:, 1], pg, atol=1e-12)
    derived = p[:, 0] - grain
    np.testing.assert_allclose(p[:, 1] - derived, grain - pg, atol=1e-12)
