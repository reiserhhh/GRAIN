#!/usr/bin/env python3
"""Run a frozen learned UD front-end on author-balanced real-text evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.evidence_contracts import digest
from grain_research.source_event_graph import event_graph, quote_spans, stanza_words
from scripts.fetch_grain_syntax_reference import MODELS, PACKAGES, PROCESSORS
from scripts.run_grain_multiview_address import BASE, VIEWS, load_material, write

OUT = BASE / "all_plan_closure_v1/source_event_r4"
PRIOR = BASE / "all_plan_closure_v1/multiview_address_r1"
PLAN = ROOT / "docs/GRAIN_SOURCE_EVENT_GRAPH_PLAN_V1.md"


def verify_models():
    """Verify downloaded official hashes before loading pickle-based model files."""
    inventory = json.loads((MODELS / "grain_inventory.json").read_text())
    for name, item in inventory["files"].items():
        if digest(MODELS / name) != item["sha256"]:
            raise ValueError("Changed syntax model: " + name)
    return inventory


def sample_comments(raw, panels, annotation):
    """One equal-author raw comment plus previously annotated source comments."""
    full = panels[panels.panel == "full"]
    selected = set()
    for _, group in full.groupby("user_id"):
        choices = group.source_row.astype(int).tolist()
        selected.add(
            min(
                choices,
                key=lambda i: hashlib.sha256(
                    f"916:{raw[i]['comment_id']}".encode()
                ).hexdigest(),
            )
        )
    review = set((annotation.author_index * 36 + annotation.comment_index).astype(int))
    return sorted(selected | review), selected, review


def graph_record(source, raw, words):
    """Create a typed event graph from cached words without further model calls."""
    text = raw["body"]
    quotes = quote_spans([w for sentence in words for w in sentence])
    events = []
    for sentence in words:
        for e in event_graph(sentence, text, quote_context=quotes):
            events.append(
                {
                    **e,
                    "source_row": source,
                    "event_id": f"{source}:{e['sentence_index']}:{e['head_id']}",
                }
            )
    return {
        "source_row": source,
        "raw_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "quotes": quotes,
        "events": events,
    }


def join_outputs(records, raw, annotation, output):
    """Keep evidence support separate from event meaning, writer score and truth."""
    events = [e for r in records for e in r["events"]]
    frame = pd.DataFrame(events)
    table = frame.copy()
    for c in table:
        if any(isinstance(v, (list, dict)) for v in table[c]):
            table[c] = table[c].map(lambda v: json.dumps(v, ensure_ascii=False))
    table.to_csv(output / "private_events.csv", index=False)
    by_source = {r["source_row"]: r["events"] for r in records}
    joins = []
    for row in annotation.itertuples():
        source = int(row.author_index * 36 + row.comment_index)
        if raw[source]["body"][row.start : row.end] != row.sentence:
            raise ValueError("Old annotation source span changed")
        candidates = [
            e for e in by_source[source] if row.start <= e["predicate_start"] < row.end
        ]
        singular = [
            e
            for e in candidates
            if e["subject_reference"] == "writer_first_person_candidate"
        ]
        joins.append(
            {
                "audit_id": row.audit_id,
                "source_row": source,
                "sentence_index": row.sentence_index,
                "item_id": row.item_id,
                "old_self_attribution": row.self_attribution,
                "old_scope": row.scope,
                "old_simpa_category": row.simpa_category,
                "old_nli": row.nli,
                "event_ids": json.dumps([e["event_id"] for e in candidates]),
                "events": len(candidates),
                "singular_writer_subject_candidates": len(singular),
                "direct_writer_subject_candidates": sum(
                    e["subject_origin"] == "direct" and not e["source_path"]
                    for e in singular
                ),
                "reported_source_events": sum(
                    bool(e["source_path"]) for e in candidates
                ),
                "intention_embedded_events": sum(
                    any(x["intention"] for x in e["embedding_path"]) for e in candidates
                ),
                "condition_events": sum(
                    bool(e["adverbial_clauses"]) for e in candidates
                ),
                "unresolved_subject_events": sum(
                    e["participant_kind"] == "unresolved" for e in candidates
                ),
                "query_event_selection": "unresolved",
            }
        )
    pairs = pd.DataFrame(joins)
    pairs.to_csv(output / "private_annotation_event_join.csv", index=False)
    links = []
    for view in VIEWS:
        segments = pd.read_csv(PRIOR / f"private_segments_full_{view}.csv")
        byrow = {int(k): f for k, f in segments.groupby("source_row")}
        for e in events:
            seg = byrow.get(e["source_row"])
            unit_ids = (
                []
                if seg is None
                else seg.loc[
                    (seg.char_start <= e["predicate_start"])
                    & (seg.char_end >= e["predicate_end"]),
                    "unit_id",
                ].tolist()
            )
            links.append(
                {
                    "event_id": e["event_id"],
                    "source_row": e["source_row"],
                    "view": view,
                    "unit_ids": json.dumps(unit_ids),
                    "n_units_containing_head": len(unit_ids),
                    "head_retained": bool(unit_ids),
                }
            )
    pd.DataFrame(links).to_csv(output / "private_event_unit_join.csv", index=False)
    byannotation = (
        pairs.groupby("old_self_attribution")
        .agg(
            pairs=("audit_id", "size"),
            any_events=("events", lambda x: int((x > 0).sum())),
            any_writer_subject=(
                "singular_writer_subject_candidates",
                lambda x: int((x > 0).sum()),
            ),
            any_direct_writer_subject=(
                "direct_writer_subject_candidates",
                lambda x: int((x > 0).sum()),
            ),
        )
        .reset_index()
    )
    byannotation.to_csv(output / "old_annotation_support.csv", index=False)
    graph_summary = {
        "events": len(events),
        "subject_reference": frame.subject_reference.value_counts().to_dict(),
        "subject_origin": frame.subject_origin.value_counts().to_dict(),
        "passive_subject_events": int(frame.passive_subject.sum()),
        "reported_or_attitude_source_events": int(frame.source_path.map(bool).sum()),
        "intention_embedded_events": int(
            frame.embedding_path.map(lambda p: any(e["intention"] for e in p)).sum()
        ),
        "negated_events": int(frame.local_negation.map(bool).sum()),
        "modal_events": int(frame.modal_cues.map(bool).sum()),
        "conditional_or_adverbial_events": int(
            frame.adverbial_clauses.map(bool).sum()
        ),
        "frequency_cue_events": int(frame.frequency_cues.map(bool).sum()),
        "unbalanced_quote_comments": sum(
            any(not q["balanced"] for q in r["quotes"]) for r in records
        ),
        "annotation_support": byannotation.to_dict("records"),
        "unit_head_coverage": pd.DataFrame(links)
        .groupby("view")
        .head_retained.agg(["sum", "count"])
        .reset_index()
        .to_dict("records"),
    }
    return graph_summary


def run(args):
    """Journal per-comment learned parses and resume only matching contracts."""
    start = time.monotonic()
    ids, _folds, raw, _, _ = load_material()
    summary = json.loads((PRIOR / "summary.json").read_text())
    for name in (
        "private_source_panels.csv",
        *[f"private_segments_full_{v}.csv" for v in VIEWS],
    ):
        if digest(PRIOR / name) != summary["files"][name]:
            raise ValueError("Changed original-address source")
    annotation = pd.read_csv(
        BASE
        / "all_plan_closure_v1/semantic_link_r2/private_annotation_coordinate_join.csv"
    )
    if len(annotation) != 80 or (annotation.fold == 0).any():
        raise ValueError("Expected80 old source-only annotated pairs")
    panels = pd.read_csv(PRIOR / "private_source_panels.csv")
    selected, balanced, review = sample_comments(raw, panels, annotation)
    if len(balanced) != 624:
        raise ValueError("Author-balanced sample changed")
    models = verify_models()
    sys.path.insert(0, str(PACKAGES))
    import stanza
    import torch

    if stanza.__version__ != "1.11.0":
        raise ValueError("Wrong stanza version")
    contract = {
        "plan_sha256": digest(PLAN),
        "script_sha256": digest(Path(__file__)),
        "graph_code_sha256": digest(ROOT / "grain_research/source_event_graph.py"),
        "model_inventory_sha256": digest(MODELS / "grain_inventory.json"),
        "stanza": stanza.__version__,
        "torch": torch.__version__,
        "processors": PROCESSORS,
        "selected_source_rows": selected,
        "raw_sha256": {
            str(i): hashlib.sha256(raw[i]["body"].encode()).hexdigest()
            for i in selected
        },
        "questionnaire_labels_loaded": False,
        "train": False,
        "device": "cpu",
        "threads": 2,
        "reused_parse_summary": digest(args.reuse_parses / "summary.json")
        if args.reuse_parses
        else None,
    }
    output = args.output
    if output.exists():
        if (
            not args.resume
            or json.loads((output / "contract.json").read_text()) != contract
        ):
            raise ValueError("Fresh output or identical --resume contract required")
    else:
        output.mkdir(parents=True)
        write(output / "contract.json", contract)
    torch.set_num_threads(2)
    pipeline = stanza.Pipeline(
        "en",
        dir=str(MODELS),
        package=None,
        processors=PROCESSORS,
        use_gpu=False,
        download_method=None,
        verbose=False,
    )
    journal = output / "private_parses.jsonl"
    if args.reuse_parses and not journal.exists():
        previous = json.loads((args.reuse_parses / "contract.json").read_text())
        prior_summary = json.loads((args.reuse_parses / "summary.json").read_text())
        for key in (
            "raw_sha256",
            "selected_source_rows",
            "stanza",
            "torch",
            "model_inventory_sha256",
            "processors",
            "device",
        ):
            if previous[key] != contract[key]:
                raise ValueError("Incompatible parser cache: " + key)
        source_journal = args.reuse_parses / "private_parses.jsonl"
        if digest(source_journal) != prior_summary["files"]["private_parses.jsonl"]:
            raise ValueError("Changed parse cache")
        shutil.copyfile(source_journal, journal)
    cached = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            r = json.loads(line)
            if (
                r["source_row"] in cached
                or contract["raw_sha256"].get(str(r["source_row"])) != r["raw_sha256"]
            ):
                raise ValueError("Changed/duplicate cached parse")
            cached[r["source_row"]] = r
    new_calls = 0
    with journal.open("a") as handle:
        for j, source in enumerate(selected):
            if source in cached:
                continue
            if time.monotonic() - start >= 1200:
                write(
                    output / "summary.json",
                    {
                        "status": "PARTIAL_TIME_BOUND",
                        "completed": len(cached),
                        "planned": len(selected),
                    },
                )
                print("PARTIAL_TIME_BOUND", flush=True)
                return
            text = raw[source]["body"]
            words = stanza_words(pipeline(text), text)
            new_calls += 1
            record = {
                "source_row": source,
                "raw_sha256": contract["raw_sha256"][str(source)],
                "sentences": words,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            cached[source] = record
            if j % 50 == 0:
                print(
                    json.dumps(
                        {
                            "completed": len(cached),
                            "planned": len(selected),
                            "elapsed_seconds": round(time.monotonic() - start, 2),
                        }
                    ),
                    flush=True,
                )
    records = [graph_record(i, raw[i], cached[i]["sentences"]) for i in selected]
    with (output / "private_graphs.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    stats = join_outputs(records, raw, annotation, output)
    stats.update(
        {
            "status": "COMPLETE",
            "planned_comments": len(selected),
            "parsed_comments": len(cached),
            "new_parser_calls": new_calls,
            "balanced_comments": len(balanced),
            "review_comments": len(review),
            "authors": len(ids),
            "parser_sentences": sum(len(r["sentences"]) for r in cached.values()),
            "parser_words": sum(
                len(s) for r in cached.values() for s in r["sentences"]
            ),
            "model_bytes": models["total_bytes"],
            "elapsed_seconds": time.monotonic() - start,
            "files": {
                p.name: digest(p)
                for p in output.iterdir()
                if p.is_file() and p.name != "summary.json"
            },
        }
    )
    write(output / "summary.json", stats)
    print(json.dumps({k: v for k, v in stats.items() if k != "files"}), flush=True)


def main():
    """Command line for one frozen no-fit natural-text graph pass."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--reuse-parses", type=Path)
    run(p.parse_args())


if __name__ == "__main__":
    main()
