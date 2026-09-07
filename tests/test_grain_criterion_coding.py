"""Polarity/coding factorial material and algebra, no model execution."""

import pytest

from scripts.run_grain_criterion_coding import contrast_coefficients, material
from scripts.run_grain_entity_state import material as entity_material
from scripts.run_grain_query_coordinate import material as coordinate_material


def test_complete_cross_and_common_prefix():
    rows = material(entity_material(coordinate_material()))
    assert len(rows) == 192
    groups = {}
    for r in rows:
        groups.setdefault((r["group_id"], r["level"], r["entity"]), []).append(r)
        assert r["expected_sign"] == r["polarity"] * r["code"] * (
            1 if r["criterion"] == "venue" else r["role_sign"]
        )
    assert len(groups) == 16
    for group in groups.values():
        assert len(group) == 12
        assert len({r["user"][: r["entity_span"][1]] for r in group}) == 1


def test_factorial_separates_semantics_from_yes_and_token_bias():
    cells = {(p, c): 2 + 3 * p + 5 * c + 7 * p * c for p in (-1, 1) for c in (-1, 1)}
    coeff, error = contrast_coefficients(cells)
    assert coeff == {"constant": 2.0, "polarity": 3.0, "code": 5.0, "semantic": 7.0}
    assert error == 0
    cells.pop((1, 1))
    with pytest.raises(ValueError, match="Missing"):
        contrast_coefficients(cells)
