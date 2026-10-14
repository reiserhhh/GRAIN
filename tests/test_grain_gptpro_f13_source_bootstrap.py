#!/usr/bin/env python3
"""Bootstrap identities, chronology and source-only fitting safeguards."""

import pytest

from scripts.run_grain_gptpro_f13_dynamic_source_bootstrap import resample_source


def test_duplicate_draws_are_distinct_groups_with_same_observations():
    people = {
        "u": [{"author_id": "u", "event_id": "e", "role": "source", "t": 7, "y": 0.3}]
    }
    rows = resample_source(people, [0, 0])
    assert len({r["event_id"] for r in rows}) == 2
    assert len({r["author_id"] for r in rows}) == 2
    assert [r["t"] for r in rows] == [7, 7]
    assert [r["y"] for r in rows] == [0.3, 0.3]
    assert people["u"][0]["author_id"] == "u"


@pytest.mark.parametrize("role,duplicate", [("application", False), ("source", True)])
def test_illegal_input_rejected(role, duplicate):
    row = {"author_id": "u", "event_id": "e", "role": role}
    with pytest.raises(ValueError):
        resample_source({"u": [row, row] if duplicate else [row]}, [0])
