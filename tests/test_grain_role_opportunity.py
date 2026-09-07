"""Synthetic material, semantic recoding, donor safety and direction isolation."""

import copy
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from grain_research.vector_mechanisms.role_opportunity import (
    build_material,
    check_donor_layout,
    code_answers,
    fit_directions,
    summarize,
    validate_material,
)


def test_complete_crossed_material_and_owner_polarity():
    rows = build_material()
    assert len(rows) == 156
    counts = Counter(r["family"] for r in rows)
    assert counts["trait"] == 48 and counts["owner"] == 24
    for r in rows:
        if r["family"] == "owner":
            assert r["expected_sign"] == r["level"] * (
                1 if r["owner"] == "target" else -1
            )


@pytest.mark.parametrize(
    "q,c,answer", [(1, 1, "A"), (1, -1, "B"), (-1, 1, "B"), (-1, -1, "A")]
)
def test_question_code_mapping(q, c, answer):
    assert code_answers(q, c)[0] == answer


def test_rejects_missing_and_duplicate_material():
    rows = build_material()
    with pytest.raises(ValueError):
        validate_material(rows[:-1])
    with pytest.raises(ValueError):
        validate_material(rows[:-1] + [rows[0]])


def test_donor_never_changes_question_or_layout():
    check_donor_layout([0, 1, 2, 3], [1, 2], [0, 9, 8, 3], [1, 2])
    with pytest.raises(ValueError, match="outside"):
        check_donor_layout([0, 1, 2, 3], [1, 2], [0, 9, 8, 4], [1, 2])
    with pytest.raises(ValueError, match="layout"):
        check_donor_layout([0, 1, 2, 3], [1, 2], [0, 9, 3], [1])


def test_direction_uses_only_development_endpoint_with_equal_norms():
    rows = build_material()
    states = {
        r["case_id"]: np.array([[r["level"], 2 * r["level"], 1.0], [999, 888, 777]])
        for r in rows
        if r["split"] == "development"
    }
    directions, info = fit_directions(rows, states)
    assert info["shared_norm"] == pytest.approx(np.sqrt(20))
    for v in directions.values():
        assert np.linalg.norm(v) == pytest.approx(info["shared_norm"])
    np.testing.assert_array_equal(directions["self"], [2, 4, 0])


def test_native_summary_separates_opportunity_and_owner():
    data = []
    gains = {"free": 2.0, "assigned": 0.5, "neutral": 1.5, "none": 1.0}
    for r in build_material():
        data.append(
            copy.deepcopy(r)
            | {
                "margin": gains[r["condition"]] * r["expected_sign"],
                "candidate_mass": 0.99,
            }
        )
    tables = summarize(pd.DataFrame(data), pd.DataFrame())
    contrast = tables["opportunity_contrasts"].set_index("contrast")
    assert contrast.loc["free_minus_assigned", "mean"] == pytest.approx(3)
    owners = tables["native_summary"].query("family == 'owner'")
    assert (owners.expected_direction_rate == 1).all()
    assert owners.query("owner == 'companion'")["mean"].iloc[0] == -2
