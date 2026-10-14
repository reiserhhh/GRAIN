#!/usr/bin/env python3
"""Bounded real query/event localization, scope comparison and frozen GRAIN join."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.evidence_contracts import digest
from grain_research.query_event import event_views, evidence_route, rank_events
from scripts.run_grain_entailment_reference import MODEL as NLI_MODEL
from scripts.run_grain_entailment_reference import REVISION as NLI_REVISION
from scripts.run_grain_evidence_calibration import infer_pairs
from scripts.run_grain_multiview_address import BASE, load_material, write
from scripts.run_grain_paraphrase_reference import MODEL, REVISION, encode_batches

BASE = BASE / "all_plan_closure_v1"
GRAPH = BASE / "source_event_r4"
ANNOTATION = BASE / "semantic_link_r2/private_annotation_coordinate_join.csv"
PLAN = ROOT / "docs/GRAIN_QUERY_EVENT_GROUNDING_PLAN_V1.md"


def read_jsonl(path):
    """Load a source-keyed local journal with unique identities."""
    rows = [json.loads(line) for line in path.open()]
    if len({r["source_row"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source rows")
    return {r["source_row"]: r for r in rows}


def candidates(raw, annotations, parses, graphs):
    """Locate original pairs and retain every syntactic event in their span."""
    result = []
    for row in annotations.itertuples():
        source = int(row.author_index * 36 + row.comment_index)
        text = raw[source]["body"]
        if text[row.start : row.end] != row.sentence:
            raise ValueError("Annotation source address changed")
        for event in graphs[source]["events"]:
            if not row.start <= event["predicate_start"] < row.end:
                continue
            views = event_views(
                text,
                parses[source]["sentences"][event["sentence_index"]],
                event,
                graphs[source]["quotes"],
            )
            result.append(
                {
                    "audit_id": row.audit_id,
                    "event_id": event["event_id"],
                    "source_row": source,
                    "predicate": event["predicate"],
                    "predicate_start": event["predicate_start"],
                    "predicate_end": event["predicate_end"],
                    "parser_sentence": event["sentence_index"],
                    "item_text": row.item_text,
                    **views,
                    **evidence_route(event),
                }
            )
    return result


def run(args):
    """Execute one fixed comparison; no labels, fitting, uploads or hidden truncation."""
    began = time.monotonic()
    _, folds, raw, _, material = load_material()
    graph_summary = json.loads((GRAPH / "summary.json").read_text())
    for name, sha in graph_summary["files"].items():
        if digest(GRAPH / name) != sha:
            raise ValueError("Changed graph artifact: " + name)
    annotation = pd.read_csv(ANNOTATION)
    if (
        len(annotation) != 80
        or annotation.audit_id.duplicated().any()
        or np.any(folds[annotation.author_index] == 0)
        or not (annotation.fold != 0).all()
    ):
        raise ValueError("Expected 80 source-only pairs")
    contract = {
        "plan": digest(PLAN),
        "material": material,
        "source_graph_summary": digest(GRAPH / "summary.json"),
        "annotation": digest(ANNOTATION),
        "code": {
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                Path(__file__),
                ROOT / "grain_research/query_event.py",
                ROOT / "scripts/run_grain_paraphrase_reference.py",
                ROOT / "scripts/run_grain_evidence_calibration.py",
            )
        },
        "encoder": [MODEL, REVISION],
        "nli": [NLI_MODEL, NLI_REVISION],
        "seed": None,
        "fitting": False,
        "max_seconds": 1200,
        "semantic_cap": 512,
        "pairs": 80,
    }
    if args.output.exists() and (
        not args.resume
        or json.loads((args.output / "contract.json").read_text()) != contract
    ):
        raise ValueError("Fresh output or identical explicit resume required")
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "contract.json", contract)
    records = candidates(
        raw,
        annotation,
        read_jsonl(GRAPH / "private_parses.jsonl"),
        read_jsonl(GRAPH / "private_graphs.jsonl"),
    )
    with (args.output / "private_candidate_views.jsonl").open("w") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    frame = pd.DataFrame(records)
    texts = sorted(
        set(annotation.item_text) | set(frame.local_text) | set(frame.scope_text)
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, trust_remote_code=False
    )
    lengths = tokenizer(texts, truncation=False)["input_ids"]
    eligible_texts = [
        t for t, tokens in zip(texts, lengths, strict=True) if len(tokens) <= 512
    ]
    vectors, encoder_contract = encode_batches(
        eligible_texts,
        args.output / "encoder_cache",
        MODEL,
        REVISION,
        cap=512,
        max_seconds=600,
    )
    if encoder_contract["truncated_texts"]:
        raise ValueError("Unexpected semantic truncation")
    lookup = dict(zip(eligible_texts, vectors, strict=True))
    for mode in ("local", "scope"):
        frame[mode + "_cosine"] = [
            float(lookup[q] @ lookup[t]) if q in lookup and t in lookup else np.nan
            for q, t in zip(frame.item_text, frame[mode + "_text"], strict=True)
        ]
    frame.to_csv(args.output / "private_candidates.csv", index=False)
    selections, requests = [], []
    for row in annotation.itertuples():
        f = frame[frame.audit_id == row.audit_id]
        selection = {
            "audit_id": row.audit_id,
            "n_events": len(f),
            "distinct_scopes": f.scope_text.nunique(),
            "old_self_attribution": row.self_attribution,
        }
        views = {"original": row.sentence}
        for mode in ("local", "scope"):
            selection.update({mode + "_event_id": None, mode + "_near_ties": None})
            for key in role_metadata_fields():
                selection[mode + "_" + key] = None
            valid = f.dropna(subset=[mode + "_cosine"])
            if valid.empty:
                selection[mode + "_status"] = "NO_ELIGIBLE_EVENT"
                continue
            ranking = rank_events(valid.event_id.tolist(), valid[mode + "_cosine"])
            chosen = valid.iloc[ranking.pop("index")]
            selection.update({mode + "_" + k: v for k, v in ranking.items()})
            selection.update(
                {mode + "_event_id": chosen.event_id, mode + "_status": "SELECTED"}
            )
            for key in role_metadata_fields():
                selection[mode + "_" + key] = chosen[key]
            if mode == "local":
                views["local_selected"] = chosen.local_text
                views["local_selected_scope"] = chosen.scope_text
            else:
                views["scope_selected"] = chosen.scope_text
        selections.append(selection)
        for view, sentence in views.items():
            requests.append(
                {
                    "audit_id": row.audit_id + ":" + view,
                    "pair_id": row.audit_id,
                    "view": view,
                    "sentence": sentence,
                    "item_text": row.item_text,
                }
            )
    selected = pd.DataFrame(selections)
    selected.to_csv(args.output / "private_selected.csv", index=False)
    requests = pd.DataFrame(requests)
    nli_tokenizer = AutoTokenizer.from_pretrained(
        NLI_MODEL, revision=NLI_REVISION, local_files_only=True, trust_remote_code=False
    )
    requests["pair_tokens"] = [
        len(t)
        for t in nli_tokenizer(
            requests.sentence.tolist(), requests.item_text.tolist(), truncation=False
        )["input_ids"]
    ]
    eligible = requests[requests.pair_tokens <= 512]
    unique = eligible.drop_duplicates(["sentence", "item_text"])
    cache = args.output / "private_nli_cache.csv"
    cache_meta = args.output / "nli_cache.json"
    if cache.exists() and cache_meta.exists():
        if json.loads(cache_meta.read_text())["sha256"] != digest(cache):
            raise ValueError("Changed NLI cache")
        result = pd.read_csv(cache)
    else:
        if time.monotonic() - began > 1100:
            raise TimeoutError("Budget reached before NLI; resume same contract")
        result = infer_pairs(unique)
        result.to_csv(cache, index=False)
        write(cache_meta, {"sha256": digest(cache), "unique_pairs": len(unique)})
    if set(result.audit_id) != set(unique.audit_id):
        raise ValueError("NLI request identities changed")
    scored = unique[["audit_id", "sentence", "item_text"]].merge(
        result, on="audit_id", validate="one_to_one"
    )
    all_scores = requests.merge(
        scored.drop(columns="audit_id"),
        on=["sentence", "item_text"],
        how="left",
        validate="many_to_one",
    )
    all_scores.to_csv(args.output / "private_nli_comparison.csv", index=False)
    links = pd.read_csv(GRAPH / "private_event_unit_join.csv")
    chosen_ids = selected[["audit_id", "local_event_id"]].rename(
        columns={"local_event_id": "event_id"}
    )
    joined = chosen_ids.merge(links, on="event_id", how="left", validate="one_to_many")
    joined.to_csv(args.output / "private_selected_grain_join.csv", index=False)
    pivot = all_scores.pivot(index="pair_id", columns="view", values="p_entailment")
    cats = all_scores.pivot(index="pair_id", columns="view", values="nli")
    common = pivot[["local_selected", "local_selected_scope"]].dropna()
    delta = common.local_selected_scope - common.local_selected
    both_selected = selected.local_status.eq("SELECTED") & selected.scope_status.eq(
        "SELECTED"
    )
    unique_scope = both_selected & selected.scope_near_ties.eq(1)
    changed = selected.local_event_id.ne(selected.scope_event_id)
    summary = {
        "status": "COMPLETE",
        "pairs": len(annotation),
        "authors": annotation.author_index.nunique(),
        "events": len(frame),
        "pairs_with_events": frame.audit_id.nunique(),
        "encoder_texts": len(eligible_texts),
        "overlength_encoder_texts": len(texts) - len(eligible_texts),
        "nli_requests": len(requests),
        "unique_nli_requests": len(unique),
        "overlength_nli_requests": int((requests.pair_tokens > 512).sum()),
        "selection_changed_including_tied_representatives": int(
            (changed & both_selected).sum()
        ),
        "selection_changed_among_unique_scope": int((changed & unique_scope).sum()),
        "unique_scope_selection_pairs": int(unique_scope.sum()),
        "pairs_one_distinct_scope": int((selected.distinct_scopes == 1).sum()),
        "near_ties": {
            m: int((selected[m + "_near_ties"] > 1).sum()) for m in ("local", "scope")
        },
        "local_fragments": int(frame.local_is_fragment.sum()),
        "same_local_scope": int(frame.scope_equals_local.sum()),
        "selected_routes": {
            m: selected[m + "_route"].value_counts().to_dict()
            for m in ("local", "scope")
        },
        "scope_routes_are_tied_representatives_not_identified_events": True,
        "old_other_local_routes": selected.loc[
            selected.old_self_attribution == "other", "local_route"
        ]
        .value_counts()
        .to_dict(),
        "nli_counts": all_scores.groupby("view")
        .nli.value_counts()
        .unstack(fill_value=0)
        .to_dict("index"),
        "scope_delta": {
            "n": len(common),
            "mean": float(delta.mean()) if len(delta) else None,
            "quantiles": np.quantile(delta, [0.05, 0.5, 0.95]).tolist()
            if len(delta)
            else [],
            "absolute_over_0_1": int((delta.abs() > 0.1).sum()),
            "class_changes": int(
                (
                    cats.loc[common.index, "local_selected"]
                    != cats.loc[common.index, "local_selected_scope"]
                ).sum()
            ),
        },
        "grain_head_coverage": joined.groupby("view")
        .head_retained.agg(["sum", "count"])
        .reset_index()
        .to_dict("records"),
        "new_parser_calls": 0,
        "nli_class_transition": pd.crosstab(
            cats.loc[common.index, "local_selected"],
            cats.loc[common.index, "local_selected_scope"],
        ).to_dict("index"),
        "labels_loaded": False,
        "accuracy": None,
        "scope": "Source-only model/query localization and scope diagnostic; no semantic gold or personality outcome",
        "wall_seconds": time.monotonic() - began,
    }
    summary["files"] = {
        p.name: digest(p)
        for p in args.output.iterdir()
        if p.is_file()
        and not p.name.startswith(".")
        and p.name not in {"summary.json", "audit.json"}
    }
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def role_metadata_fields():
    """Names of role metadata copied without mixing them into cosine selection."""
    return (
        "route",
        "subject_reference",
        "subject_origin",
        "source_candidate",
        "passive_subject",
        "intention_embedded",
        "ancestor_negation",
        "local_negation",
        "modal",
        "adverbial_linked",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "query_event_r1")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
