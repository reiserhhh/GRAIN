#!/usr/bin/env python3
"""Synthetic checks of the separately declared post-outcome description."""

import numpy as np
import pytest

from scripts.describe_grain_author_predictive_link_hm257 import describe


def fixture():
    """Keep a signed normalizer reversal in every complete observation account."""
    fields = np.arange(240, dtype=float).reshape(48, 5) / 240
    actions = {name: np.full((48, 5), -8.0) for name in ("stranger", "order", "carry")}
    rows = [
        {
            "unit": u,
            "operation": name,
            "target": q,
            "token_position": t,
            "numerator": 1.0,
            "normalizer": 3.0,
            "nats": -2.0,
        }
        for u in range(48)
        for name in actions
        for q in "ENACO"
        for t in range(4)
    ]
    return fields, fields * 0.25, actions, np.full(48, 4), rows


def test_all_directions_parts_and_quarters_are_preserved():
    """Descriptive output neither picks a positive sector nor drops the denominator."""
    directions, terms, quarters, error = describe(*fixture())
    assert (len(directions), len(terms), len(quarters)) == (900, 30, 2880)
    assert error == 0
    assert all(row["normalizer_changes_mean_sign"] for row in terms)
    assert all(row["nats"] == -2.0 for row in quarters)


@pytest.mark.parametrize("duplicate", [False, True])
def test_missing_or_duplicate_token_is_not_complete(duplicate):
    """A count-only completion cannot hide omission or reused evidence."""
    fields, choice, actions, lengths, rows = fixture()
    rows = rows + rows[:1] if duplicate else rows[1:]
    with pytest.raises(ValueError):
        describe(fields, choice, actions, lengths, rows)
