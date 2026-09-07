"""Common-prefix material, ownership orientation and future-token safeguards."""

import numpy as np
import pytest

from scripts.run_grain_entity_state import material, verify_common_prefix
from scripts.run_grain_query_coordinate import material as parent_material


def test_both_people_and_three_criteria_share_literal_entity_prefix():
    rows = material(parent_material())
    assert len(rows) == 48
    for group in {r["group_id"] for r in rows}:
        for level in (-1, 1):
            subset = [r for r in rows if r["group_id"] == group and r["level"] == level]
            assert {r["role_sign"] for r in subset} == {-1, 1}
            for entity in ("Alex", "Blair"):
                own = [r for r in subset if r["entity"] == entity]
                assert len(own) == 3
                assert len({r["user"][: r["entity_span"][1]] for r in own}) == 1
                assert all(
                    r["user"][slice(*r["entity_span"])].strip() == entity for r in own
                )


def test_shared_state_check_rejects_changed_prefix_or_state():
    rows = [
        {"case_id": c, "group_id": "g", "level": 1, "entity": "Alex", "criterion": c}
        for c in ("event", "trait", "venue")
    ]
    layouts = {
        c: {"N": [1], "ids": [1, 2, j]}
        for j, c in enumerate(("event", "trait", "venue"))
    }
    states = {c: np.ones((3, 4)) for c in layouts}
    assert (
        max(
            r["max_state_difference"]
            for r in verify_common_prefix(rows, layouts, states)
        )
        == 0
    )
    states["trait"][1, 0] = 2
    with pytest.raises(ValueError):
        verify_common_prefix(rows, layouts, states)
    states["trait"][1, 0] = 1
    layouts["trait"]["ids"][0] = 7
    with pytest.raises(ValueError):
        verify_common_prefix(rows, layouts, states)
