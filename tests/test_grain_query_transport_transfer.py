"""Fresh transport material and unchanged operator scoring contracts."""

from collections import Counter

from scripts.run_grain_query_transport_transfer import material


def test_fresh_complete_material():
    rows = material()
    assert len(rows) == len({r["case_id"] for r in rows}) == 48
    assert Counter(r["form"] for r in rows) == {"canonical": 24, "rewrite": 24}
    assert Counter(r["variant"] for r in rows) == {
        "event_target": 16,
        "trait_code1": 16,
        "venue": 16,
    }
    assert all(r["group_id"].startswith("f_") for r in rows)
    for group in {r["group_id"] for r in rows}:
        evidence = {r["level"]: r["evidence"] for r in rows if r["group_id"] == group}
        assert Counter(evidence[1].split()) == Counter(evidence[-1].split())
