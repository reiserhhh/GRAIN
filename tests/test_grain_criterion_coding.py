"""Polarity/coding factorial material and algebra, no model execution."""

import pandas as pd
import pytest

from scripts.audit_grain_criterion_coding import parity_replay
from scripts.run_grain_criterion_coding import (
    contrast_coefficients,
    material,
    summarize,
)
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


def test_independent_parity_replay_and_tamper_detection():
    rows = material(entity_material(coordinate_material()))
    keys = ("group_id", "level", "entity", "criterion", "polarity", "code")
    lookup = {tuple(r[k] for k in keys): r["case_id"] for r in rows}
    base = pd.DataFrame(
        [
            {
                **r,
                "margin": 2
                + r["role_sign"] * (3 + 5 * r["code"] + 7 * r["code"] * r["polarity"]),
            }
            for r in rows
        ]
    )
    index = base.set_index("case_id").margin
    effects = []
    for r in rows:
        donor = lookup[
            r["group_id"],
            -r["level"],
            r["entity"],
            r["criterion"],
            r["polarity"],
            r["code"],
        ]
        for patch in ("N", "EQD"):
            effects.append(
                {**r, "patch": patch, "donor_case": donor, "margin": index[donor]}
            )
    result = summarize(base, pd.DataFrame(effects))
    tables = [result[k] for k in ("role_contrasts", "coefficients", "parity_summary")]
    assert parity_replay(*tables) < 1e-12
    with pytest.raises(ValueError, match="Incomplete"):
        parity_replay(tables[0].iloc[:-1], *tables[1:])
    tables[1].loc[0, "semantic"] += 1
    with pytest.raises(ValueError, match="Parity"):
        parity_replay(*tables)
