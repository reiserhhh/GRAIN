#!/usr/bin/env python3
"""Small correctness floor for query-first author evidence experiments."""

import numpy as np
import pytest

from scripts.run_grain_real_author_path import (
    choose_comments,
    select_membership,
    variants,
)


def test_query_poles_select_both_directions():
    v = np.array([[1.0, 0.0], [-1.0, 0.0], [0.9, 0.1], [-0.9, 0.1]] + [[0.0, 1.0]] * 8)
    selected, control, scores = choose_comments(v, [[-1.0, 0.0], [1.0, 0.0]], [5] * 12)
    assert selected == [0, 1, 2, 3]
    assert not set(selected) & set(control)
    assert scores[0] > 0 > scores[1]


def test_matching_uses_length_not_score_direction():
    v = np.array([[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 8)
    lengths = [5] * 4 + [2, 3, 7, 8, 20, 30, 40, 50]
    _, c, _ = choose_comments(v, [[-1.0, 0.0], [1.0, 0.0]], lengths)
    assert sum(lengths[i] for i in c) == 20


def test_evidence_order_and_counts():
    texts = [str(i) for i in range(12)]
    rows = variants(texts, [0, 3, 6, 9], [1, 4, 7, 10])
    assert list(map(len, rows.values())) == [12, 8, 8, 4, 4]
    assert rows["selected_only"] == ["0", "3", "6", "9"]
    assert rows["remove_selected"] == [
        str(i) for i in range(12) if i not in [0, 3, 6, 9]
    ]


def test_invalid_vectors_rejected():
    with pytest.raises(ValueError):
        choose_comments(np.ones((4, 2)), np.ones((2, 2)), [2] * 4)


def test_new_membership_excludes_prior_authors_and_keeps_all_held():
    registry = {
        "author_ids": np.array(["a", "b", "c", "d", "e", "f"]),
        "source_group": np.array(["mbti_only"] * 5 + ["bridge"]),
        "all_arms_ready_reference_free": np.array([True] * 4 + [False, True]),
    }
    rows = select_membership(registry, {"a", "b"}, [("TF_cont", "mbti_only", 2)], True)
    assert {r["author_id"] for r in rows} == {"c", "d"}
    assert {r["split"] for r in rows} == {"held_author"}
