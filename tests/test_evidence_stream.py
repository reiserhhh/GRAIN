#!/usr/bin/env python3
"""Validate incremental ingest, coverage diagnostics, and actor-matched queries."""

from io import StringIO

import pytest
from grain_core.evidence_panel import (
    Coverage,
    EvidenceEvent,
    EvidencePanel,
    Opportunity,
    run_queries,
)
from grain_core.evidence_stream import (
    combine_sources,
    namespace_source,
    read_panel,
    text_csv,
    write_stream,
)
from grain_core.evidence_text import TextRecord, iter_native_text_evidence


def test_mixed_stream_is_single_pass_and_matches_in_memory_queries():
    rows = [
        EvidenceEvent("a1", "before", "u", "home", "text", [1]),
        Opportunity("a1", "after", "u", "menu", "pick", True, None),
        EvidenceEvent("a2", "after", "u", "away", "text"),
        Coverage("after", True, "home"),
    ]
    channels = {"text": ["value"]}
    expected = EvidencePanel.from_records(channels, iter(rows))
    src = StringIO()
    write_stream(src, channels, iter(rows))
    src.seek(0)
    actual = read_panel(src)
    queries = [
        {
            "id": "q",
            "op": "participation_flow",
            "before": "before",
            "after": "after",
            "target": "home",
        }
    ]
    assert run_queries(actual, queries) == run_queries(expected, queries)
    assert actual.diagnostics() == expected.diagnostics()
    assert actual.diagnostics()["channels"] == {"text": {"records": 2, "scored": 1}}


def test_ndjson_reports_physical_line_and_rejects_oversized_record():
    text = '{"schema_version":"grain.evidence-stream.v1","channels":{}}\n\n{"type":"ignored"}\n'
    with pytest.raises(ValueError, match="line 3: unknown"):
        read_panel(StringIO(text))
    with pytest.raises(ValueError, match="line 1: record exceeds"):
        read_panel(StringIO(text), max_line=40)


def test_csv_uses_a_declared_map_and_preserves_quoted_newlines():
    columns = {"id": "report", "actor": "user", "text": "body"}
    constants = {"window": "a", "context": "essays"}
    src = StringIO('report,user,body\n1,u,"a\nb"\n')
    assert list(text_csv(src, columns=columns, constants=constants)) == [
        TextRecord("1", "a", "u", "essays", "a\nb")
    ]
    with pytest.raises(ValueError, match="column count"):
        list(
            text_csv(
                StringIO("report,user,body\n1,u\n"),
                columns=columns,
                constants=constants,
            )
        )


def test_frozen_adapter_only_consumes_the_requested_batch(monkeypatch):
    import grain_core.evidence_text as module

    calls = []

    class Reader:
        def __init__(self, atlas):
            calls.append(atlas)

        def locate(self, batch):
            import numpy as np

            return np.zeros((len(batch), 1)), [True] * len(batch)

    def source():
        for i in range(5):
            calls.append(str(i))
            yield TextRecord(str(i), "a", "person", "writing", "Example")

    monkeypatch.setattr(module, "_AtlasLocator", Reader)
    atlas = {"labels": ["node"]}
    _, events = iter_native_text_evidence(atlas, source(), batch_size=2)
    assert calls == []
    assert next(events).id == "0" and calls == ["0", "1", atlas]
    assert len(list(events)) == 4 and calls.count(atlas) == 1


def test_coverage_unknown_flow_still_lists_observed_destinations():
    rows = [
        EvidenceEvent("1", "a", "u", "home", "c"),
        EvidenceEvent("2", "a", "u", "other", "c"),
        EvidenceEvent("3", "b", "u", "other", "c"),
        EvidenceEvent("4", "b", "u", "other", "c"),
        EvidenceEvent("5", "b", "new", "home", "c"),
    ]
    p = EvidencePanel({"c": ["x"]}, rows)
    v = p.participation_flow("a", "b", "home")
    assert v["prior_roster_current_states"] == {"target_coverage_unknown": 1}
    assert v["not_in_prior_observed_roster"] == 1
    assert v["destinations_without_target_records"]["other"] == {
        "previously_observed": 1,
        "target_coverage_unconfirmed": 1,
    }


def test_paired_actions_use_same_people_and_keep_unknown_outcome_interval():
    p = EvidencePanel.from_records(
        {"c": ["reading"]},
        [
            Opportunity("1", "a", "u", "x", "act", True, False),
            Opportunity("2", "b", "u", "y", "act", True, True),
            Opportunity("3", "b", "u", "y", "act", True, None),
            Opportunity("4", "a", "v", "x", "act", None, None),
            Opportunity("5", "b", "v", "y", "act", True, False),
            Opportunity("6", "b", "new", "y", "act", True, True),
            EvidenceEvent("x", "a", "u", "x", "c", [1]),
            EvidenceEvent("y", "b", "u", "y", "c", [3]),
        ],
    )
    c = p.paired_opportunities("a", "b", "x", "y", "act", channel="c")
    assert c["before_logged_actors"] == 2 and c["after_logged_actors"] == 3
    assert c["paired_logged_actors"] == 2 and c["paired_known_eligible"]["actors"] == 1
    assert c["paired_known_eligible"]["known_eligible_delta_bounds"] == [0.5, 1.0]
    assert c["with_paired_coordinates"]["mean_coordinate_shift"] == [2]


def test_independent_sources_do_not_pair_by_local_actor_id():
    row = Opportunity("e", "w", "001", "menu", "choose", True, False)

    def panel(link):
        a, ea = namespace_source(
            "s1",
            {},
            [row, Coverage("w", True)],
            actor_aliases=link,
            shared_measures=["choose"],
        )
        b, eb = namespace_source(
            "s2", {}, [row], actor_aliases=link, shared_measures=["choose"]
        )
        return combine_sources((a, ea), (b, eb))

    p = panel(None)
    assert (
        p.paired_opportunities("s1:w", "s2:w", "s1:menu", "s2:menu", "choose")[
            "paired_logged_actors"
        ]
        == 0
    )
    assert (
        p.participation("s2:w", "source:s2:001", "s2:menu")["feed_coverage"]
        == "unknown"
    )
    linked = panel({"001": "validated"})
    assert (
        linked.paired_opportunities("s1:w", "s2:w", "s1:menu", "s2:menu", "choose")[
            "paired_logged_actors"
        ]
        == 1
    )


def test_merging_requires_the_same_coordinate_order_for_shared_channels():
    a = namespace_source("s1", {"axis": ["A", "B"]}, [], shared_channels=["axis"])
    b = namespace_source("s2", {"axis": ["B", "A"]}, [], shared_channels=["axis"])
    with pytest.raises(ValueError, match="incompatible coordinate order"):
        combine_sources(a, b)
