"""Crossed evidence materials and exact finite composition identity."""

import numpy as np
import pandas as pd

from scripts.run_grain_attribute_routing import decompose, material, summarize
from scripts.run_grain_entity_state import material as entity_material
from scripts.run_grain_query_coordinate import material as coordinate_material
from scripts.run_grain_target_recency import material as recency_material


def panel():
    """Build authored test records without any human data."""
    return material(recency_material(entity_material(coordinate_material())))


def test_truth_cross_keeps_actions_and_target():
    rows = panel()
    assert len(rows) == 80
    assert sum(r["target_kind"] == "object" for r in rows) == 16
    for r in rows:
        assert r["user"][slice(*r["anchor_span"])].strip() == r["anchor_text"]
        assert ("The room was open." in r["evidence"]) == (r["room_sign"] == 1)
        assert r["expected_sign"] == (
            r["room_sign"] if r["target_kind"] == "object" else r["role_sign"]
        )


def test_state_and_readout_terms_can_cancel():
    result = decompose(1.0, 2.0, 3.0, 5.0, 8.0)
    assert result == {
        "total": 1.0,
        "state_departure": -3.0,
        "readout_interaction": 4.0,
        "identity_error": 0.0,
    }
    assert (
        result["total"] ** 2
        == result["state_departure"] ** 2
        + result["readout_interaction"] ** 2
        + 2 * result["state_departure"] * result["readout_interaction"]
    )


def test_complete_summary_retains_irrelevant_axes():
    rows = panel()
    base = pd.DataFrame(
        [{**r, "margin": float(r["role_sign"] + 2 * r["room_sign"])} for r in rows]
    )
    index = base.set_index("case_id").margin
    lookup = {
        (r["group_id"], r["level"], r["entity"], r["criterion"], r["room_sign"]): r[
            "case_id"
        ]
        for r in rows
    }
    effects = []
    for r in rows:
        for label, b, t in (
            ("B", -r["level"], r["room_sign"]),
            ("R", r["level"], -r["room_sign"]),
            ("BR", -r["level"], -r["room_sign"]),
        ):
            donor = lookup[r["group_id"], b, r["entity"], r["criterion"], t]
            for patch in ("anchor", "EQD"):
                effects.append(
                    {
                        **r,
                        "patch": patch,
                        "intervention": label,
                        "native_donor_case": donor,
                        "margin": index[donor],
                    }
                )
        effects.append(
            {
                **r,
                "patch": "anchor",
                "intervention": "ADD",
                "native_donor_case": "",
                "margin": float(-r["role_sign"] - 2 * r["room_sign"]),
            }
        )
    result = summarize(base, pd.DataFrame(effects))
    assert len(result["native_contrasts"]) == 80
    assert len(result["effect_summary"]) == 21
    assert len(result["interactions"]) == 80
    assert np.allclose(result["interactions"].identity_error, 0)
