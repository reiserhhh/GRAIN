"""Predeclared operands, paired random controls, and exact factorial witnesses."""

import pytest

from scripts.run_grain_query_binding import (
    MASKS,
    anchored_terms,
    matched_random,
    word_intervals,
)
from scripts.run_grain_query_coordinate import material


def test_operands_are_question_local_and_explicit():
    for row in material():
        intervals = word_intervals(row)
        for value in intervals.values():
            a, b = value["span"]
            assert row["user"][a:b].strip() == value["text"]
            assert row["intervals"]["C"][0] <= a < b <= row["intervals"]["C"][1]


def test_matched_random_is_deterministic_and_nonoverlapping():
    values = matched_random(list(range(9)), [3, 4], "paired-id")
    assert values == matched_random(list(range(9)), [3, 4], "paired-id")
    assert len(values) == 2 and not set(values) & {3, 4}
    with pytest.raises(ValueError):
        matched_random([1, 2, 3], [1, 2], "too-few")


def test_interaction_terms_recover_known_nonadditivity():
    values = {m: 7 + 2 * m[0] + 3 * m[1] * m[2] - m[0] * m[1] * m[2] for m in MASKS}
    terms = anchored_terms(values)
    assert terms == {"N": 2, "K": 0, "R": 0, "NK": 0, "NR": 0, "KR": 3, "NKR": -1}
