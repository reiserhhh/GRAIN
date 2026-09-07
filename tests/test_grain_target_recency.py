"""P1M material identity, deduplication and recording-order safeguards."""

import pandas as pd
import pytest

from scripts.run_grain_entity_state import material as entity_material
from scripts.run_grain_query_coordinate import material as coordinate_material
from scripts.run_grain_target_recency import material, reorder, summarize


def test_action_order_reversal_preserves_header_and_ownership():
    text = "Room open.\nAlex initiates.\nBlair waits."
    assert reorder(text, "reversed") == "Room open.\nBlair waits.\nAlex initiates."
    with pytest.raises(ValueError):
        reorder("Room open.\nAlex initiates.", "reversed")


def test_all_parent_prompts_replay_and_room_is_not_duplicated():
    parent = entity_material(coordinate_material())
    rows = material(parent)
    assert len(rows) == 112
    old = {r["case_id"]: r for r in parent}
    assert sum(r["target_kind"] == "object" for r in rows) == 16
    for r in rows:
        assert r["user"][slice(*r["anchor_span"])].strip() == r["anchor_text"]
        if r["target_kind"] == "person" and r["record_order"] == "original":
            assert r["user"] == old[r["parent_case"]]["user"]
    for group in {r["group_id"] for r in rows}:
        assert (
            sum(r["group_id"] == group and r["target_kind"] == "object" for r in rows)
            == 4
        )


def test_venue_comparison_counts_one_object_against_two_people():
    rows = material(entity_material(coordinate_material()))
    base = pd.DataFrame([{**r, "margin": float(r["role_sign"])} for r in rows])
    lookup = {
        (r["group_id"], r["level"], r["entity"], r["criterion"], r["record_order"]): r[
            "case_id"
        ]
        for r in rows
    }
    effects = pd.DataFrame(
        [
            {
                **r,
                "margin": float(-r["role_sign"]),
                "patch": patch,
                "donor_case": lookup[
                    r["group_id"],
                    -r["level"],
                    r["entity"],
                    r["criterion"],
                    r["record_order"],
                ],
            }
            for r in rows
            for patch in ("anchor", "EQD")
        ]
    )
    tables = summarize(base, effects)
    assert len(tables["venue_comparison"]) == 24
    assert (tables["venue_comparison"].object_minus_person == 0).all()
    assert len(tables["order_contrasts"]) == 84
