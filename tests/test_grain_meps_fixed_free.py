#!/usr/bin/env python3
"""Synthetic tests only: private MEPS data are never needed by CI."""

import numpy as np
import pandas as pd
import pytest

from grain_research.vector_mechanisms.meps_fixed_free import (
    aggregate_windows,
    correspondence,
    holm,
    length_matched,
    normalize,
    weighted_node_stat,
)


def test_identity_and_flat_similarity():
    identity = np.eye(6)
    np.testing.assert_allclose(weighted_node_stat(identity, np.ones(6)), [1, 1])
    np.testing.assert_allclose(
        weighted_node_stat(np.ones((6, 6)), np.ones(6)), [0, 0.5]
    )
    result = correspondence(
        identity, np.ones(6), np.ones(6), draws=40, permutations=199
    )
    assert result["gap_permutation_p"] < 0.05
    assert result["gap_ci_low"] == 1


def test_node_bootstrap_excludes_copies_of_same_author():
    s = np.array([[1, 0.2, 0.3], [0.1, 1, 0.4], [0.7, 0.6, 1]])
    w = np.array([3, 1, 0])
    expected = []
    for a in (s, s.T):
        expected.append(np.average([1 - a[0, 1], 1 - a[1, 0]], weights=[3, 1]))
    np.testing.assert_allclose(weighted_node_stat(s, w), [np.mean(expected), 1])


def test_shape_and_nonfinite_rejected():
    with pytest.raises(ValueError):
        weighted_node_stat(np.ones((2, 3)), [1, 1])
    with pytest.raises(ValueError):
        weighted_node_stat(np.eye(3), [3, 0, 0])
    with pytest.raises(ValueError):
        normalize([0, 0])


def test_length_selection_independent_of_cosine():
    s = np.eye(4)
    gap, rank, error = length_matched(s, [10, 10, 100, 100], [10, 10, 100, 100], 1)
    assert (gap, rank, error) == (1, 1, 0)


def test_holm_and_reproducibility():
    np.testing.assert_allclose(holm([0.04, 0.001, 0.02]), [0.04, 0.003, 0.04])
    s = np.random.default_rng(8).normal(size=(8, 8))
    a = correspondence(s, np.ones(8), np.ones(8), draws=10, permutations=19)
    assert a == correspondence(s, np.ones(8), np.ones(8), draws=10, permutations=19)


def test_speaker_isolation_turn_pooling_and_no_fake_assistant():
    rows = []
    for role, seq, start, end, tokens in [
        ("user", 0, 0, 3, 3),
        ("user", 0, 3, 4, 1),
        ("user", 2, 0, 1, 1),
        ("assistant", 1, 0, 10, 10),
        ("system", 3, 0, 1, 1),
    ]:
        rows.append(
            {
                "participant_code": "a",
                "view_kind": "free_chat",
                "condition": "free_chat",
                "role": role,
                "turn_sequence": seq,
                "start": start,
                "end": end,
                "content_tokens": tokens,
            }
        )
    frame = pd.DataFrame(rows)
    vectors = np.array([[1.0, 0], [0, 1], [0, 1], [-1, 0], [0, -1]])
    meta, pooled = aggregate_windows(frame, vectors)
    i = meta.index[(meta.role == "user") & (meta.pooling == "token")][0]
    np.testing.assert_allclose(pooled[i], normalize([3, 2]))
    j = meta.index[(meta.role == "user") & (meta.pooling == "equal_turn")][0]
    np.testing.assert_allclose(pooled[j], normalize(normalize([3, 1]) + [0, 1]))
    assert meta.loc[i, "turns"] == 2
    assert meta.loc[i, "chars"] == 5
    only, _ = aggregate_windows(frame.iloc[:3], vectors[:3])
    assert "assistant" not in set(only.role)
