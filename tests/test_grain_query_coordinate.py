"""Order manipulation preserves clauses and never loses boundary-token accounting."""

from collections import Counter

import pytest

from scripts.run_grain_query_coordinate import material, semantic_partition


def test_material_exhaustive_and_only_clause_order_changes():
    rows = material()
    assert len(rows) == len({r["case_id"] for r in rows}) == 48
    for row in rows:
        for key, (a, b) in row["intervals"].items():
            assert row["user"][a:b].strip() == row["clauses"][key]
    for parent in {r["parent_case"] for r in rows}:
        pair = [r for r in rows if r["parent_case"] == parent]
        assert Counter(pair[0]["user"].split()) == Counter(pair[1]["user"].split())
        assert pair[0]["clauses"] == pair[1]["clauses"]
        assert pair[0]["evidence"] == pair[1]["evidence"]


def test_partition_keeps_boundaries_and_rejects_overlap():
    parts = semantic_partition(list(range(2, 12)), [3, 4], [7, 8])
    assert parts["R"] == [2, 5, 6, 9, 10, 11]
    assert sorted(i for p in parts.values() for i in p) == list(range(2, 12))
    with pytest.raises(ValueError):
        semantic_partition(list(range(2, 12)), [3, 4], [4, 5])
    with pytest.raises(ValueError):
        semantic_partition(list(range(2, 12)), [0], [4, 5])
