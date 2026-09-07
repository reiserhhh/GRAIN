"""Finite factorial decomposition and disjoint intervention partitions."""

import pytest

from scripts.run_grain_evidence_conduit import (
    MASKS,
    conduit_material,
    factorial_interactions,
    partitions,
)


def test_complete_material_families():
    core, opportunity = conduit_material()
    assert len(core) == 60 and len(opportunity) == 36
    assert len({r["case_id"] for r in core + opportunity}) == 96
    assert {r["variant"] for r in core} == {
        "event_target",
        "event_companion",
        "trait_code1",
        "trait_code-1",
        "venue",
    }


def test_partition_covers_suffix_without_overlap():
    p = partitions(10, [3, 4, 5])
    assert p == {"E": [3, 4, 5], "Q": [6, 7, 8], "D": [9]}
    assert sorted(i for indices in p.values() for i in indices) == list(range(3, 10))
    with pytest.raises(ValueError):
        partitions(4, [1, 2, 3])


def test_factorial_recovers_all_interactions_not_just_subtraction():
    values = {
        (e, q, d): 7
        + 2 * e
        + 3 * q
        + 4 * d
        + 5 * e * q
        + 6 * e * d
        + 8 * q * d
        + 9 * e * q * d
        for e, q, d in MASKS
    }
    terms = factorial_interactions(values)
    assert terms == {"E": 2, "Q": 3, "D": 4, "EQ": 5, "ED": 6, "QD": 8, "EQD": 9}
    assert values[(1, 1, 1)] - values[(0, 0, 0)] == sum(terms.values())
    values.pop((1, 0, 1))
    with pytest.raises(ValueError):
        factorial_interactions(values)
