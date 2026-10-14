#!/usr/bin/env python3
"""Audit natural event addresses, source-role correction and cold parser replay."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.evidence_contracts import digest
from scripts.run_grain_multiview_address import BASE, load_material, write
from scripts.run_grain_source_event_graph import sample_comments


def run():
    """Independent address/count checks, without invoking the neural parser."""
    base = BASE / "all_plan_closure_v1"
    out = base / "source_event_r4"
    summary = json.loads((out / "summary.json").read_text())
    for name, sha in summary["files"].items():
        if digest(out / name) != sha:
            raise ValueError("Changed graph artifact: " + name)
    ids, _, raw, _, _ = load_material()
    parses = {
        r["source_row"]: r
        for r in (json.loads(line) for line in (out / "private_parses.jsonl").open())
    }
    graphs = [json.loads(line) for line in (out / "private_graphs.jsonl").open()]
    words_checked, events_checked, passive_unknown = 0, 0, 0
    for record in graphs:
        source = record["source_row"]
        text = raw[source]["body"]
        sentences = parses[source]["sentences"]
        for sentence in sentences:
            for w in sentence:
                assert text[w["start"] : w["end"]] == w["surface"]
                assert w["head"] == 0 or 1 <= w["head"] <= len(sentence)
                words_checked += 1
        for event in record["events"]:
            sentence = {w["id"]: w for w in sentences[event["sentence_index"]]}
            head = sentence[event["head_id"]]
            assert (
                head["start"] == event["predicate_start"]
                and head["end"] == event["predicate_end"]
            )
            assert head["lemma"] == event["predicate"]
            assert event["factuality"] == event["query_relevance"] == "not_inferred"
            cursor = head["head"]
            for item in event["embedding_path"]:
                assert item["predicate_id"] == cursor
                parent = sentence[cursor]
                cursor = parent["head"]
                if item["source_origin"] == "passive_source_unknown":
                    assert (
                        item["source_kind"] == "unresolved" and item["source_ids"] == []
                    )
            assert cursor == 0
            if event["source_path"]:
                first = event["source_path"][0]
                assert event["source_candidate"] == first["source_kind"]
                passive_unknown += int(
                    first["source_origin"] == "passive_source_unknown"
                )
            events_checked += 1
    assert (
        words_checked == summary["parser_words"] and events_checked == summary["events"]
    )
    assert len({raw[s]["author"] for s in parses}) == len(ids) == 624
    old = pd.read_csv(base / "source_event_r2/private_events.csv").set_index("event_id")
    new = pd.read_csv(out / "private_events.csv").set_index("event_id")
    assert old.index.equals(new.index)
    changed = old.source_candidate != new.source_candidate
    assert (new.loc[changed].source_candidate == "unresolved").all()
    correction = {
        "changed_source_assignments": int(changed.sum()),
        "old_first_person_assignments_corrected": int(
            ((old.source_candidate == "first_person_singular") & changed).sum()
        ),
        "passive_source_unknown_events": passive_unknown,
    }
    cold_equal = {}
    for name in (
        "private_parses.jsonl",
        "private_graphs.jsonl",
        "private_events.csv",
        "private_annotation_event_join.csv",
        "private_event_unit_join.csv",
    ):
        cold_equal[name] = digest(base / "source_event_r1" / name) == digest(
            base / "source_event_r2" / name
        )
    assert all(cold_equal.values())
    assert digest(out / "private_parses.jsonl") == digest(
        base / "source_event_r2/private_parses.jsonl"
    )
    links = pd.read_csv(out / "private_event_unit_join.csv")
    assert not links.duplicated(["event_id", "view"]).any()
    assert len(links) == events_checked * 5
    counts = links.groupby("view").head_retained.sum().to_dict()
    assert all(counts[r["view"]] == r["sum"] for r in summary["unit_head_coverage"])
    annotations = pd.read_csv(
        base / "semantic_link_r2/private_annotation_coordinate_join.csv"
    )
    panels = pd.read_csv(base / "multiview_address_r1/private_source_panels.csv")
    _, balanced, review = sample_comments(raw, panels, annotations)
    support = []
    for name, chosen in (
        ("author_balanced", balanced),
        ("previous_annotation", review),
    ):
        f = new[new.source_row.isin(chosen)]
        support.append(
            {
                "sample": name,
                "planned_comments": len(chosen),
                "comments_with_events": int(f.source_row.nunique()),
                "events": len(f),
                "writer_subject_candidates": int(
                    (f.subject_reference == "writer_first_person_candidate").sum()
                ),
                "unresolved_subjects": int((f.participant_kind == "unresolved").sum()),
                "source_embedded_events": int(
                    f.source_path.map(lambda x: bool(json.loads(x))).sum()
                ),
                "intention_embedded_events": int(
                    f.embedding_path.map(
                        lambda x: any(e["intention"] for e in json.loads(x))
                    ).sum()
                ),
            }
        )
    pd.DataFrame(support).to_csv(out / "audited_sample_support.csv", index=False)
    result = {
        "status": "PASS",
        "raw_word_addresses_checked": words_checked,
        "event_paths_checked": events_checked,
        "event_view_rows": len(links),
        "cold_replay_equal": cold_equal,
        "correction": correction,
        "sample_support": support,
        "sample_overlap_comments": len(balanced & review),
        "support_sha256": digest(out / "audited_sample_support.csv"),
        "summary_sha256": digest(out / "summary.json"),
        "audit_code_sha256": digest(Path(__file__)),
        "neural_calls": 0,
    }
    write(out / "audit.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    run()
