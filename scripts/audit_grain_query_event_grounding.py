#!/usr/bin/env python3
"""Recalculate query-event ranking and scope changes from exact cached addresses."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.evidence_contracts import digest
from scripts.run_grain_multiview_address import load_material, write
from scripts.run_grain_query_event_grounding import ANNOTATION, BASE, GRAPH, read_jsonl


def run(args):
    """Independent arithmetic/address checks; no encoder/NLI/parser inference."""
    out = args.output
    summary = json.loads((out / "summary.json").read_text())
    for name, sha in summary["files"].items():
        assert digest(out / name) == sha, name
    _, folds, raw, _, _ = load_material()
    annotation = pd.read_csv(ANNOTATION).set_index("audit_id")
    assert (folds[annotation.author_index] != 0).all()
    records = [
        json.loads(line) for line in (out / "private_candidate_views.jsonl").open()
    ]
    parses = read_jsonl(GRAPH / "private_parses.jsonl")
    graphs = read_jsonl(GRAPH / "private_graphs.jsonl")
    event_index = {e["event_id"]: e for r in graphs.values() for e in r["events"]}
    spans_checked = 0
    for r in records:
        body = raw[r["source_row"]]["body"]
        a = annotation.loc[r["audit_id"]]
        event = event_index[r["event_id"]]
        assert a.start <= r["predicate_start"] < a.end
        assert r["predicate_start"] == event["predicate_start"]
        assert r["scope_text"] == body[r["scope_start"] : r["scope_end"]]
        assert r["local_text"] == " ... ".join(
            body[x:y] for x, y in r["local_segments"]
        )
        words = {
            w["id"]: w
            for w in parses[r["source_row"]]["sentences"][r["parser_sentence"]]
        }
        assert set(r["local_ids"]) == set(event["local_word_ids"]) | set(
            event["subject_ids"]
        )
        for i in r["local_ids"]:
            w = words[i]
            assert body[w["start"] : w["end"]] == w["surface"]
            assert any(
                x <= w["start"] and y >= w["end"] for x, y in r["local_segments"]
            )
        assert set(r["scope_ids"]) >= set(r["local_ids"])
        for p in event["embedding_path"]:
            assert p["predicate_id"] in r["scope_ids"]
        for x, y in r["local_segments"]:
            assert r["scope_start"] <= x < y <= r["scope_end"]
            spans_checked += 1
    candidates = pd.read_csv(out / "private_candidates.csv")
    texts = sorted(
        set(annotation.item_text)
        | set(candidates.local_text)
        | set(candidates.scope_text)
    )
    encoder = json.loads((out / "encoder_cache/contract.json").read_text())
    assert encoder["truncated_texts"] == 0 and summary["overlength_encoder_texts"] == 0
    assert (
        hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
        == encoder["texts_sha256"]
    )
    blocks = []
    for p in sorted((out / "encoder_cache").glob("batch_*.npz")):
        meta = json.loads(p.with_suffix(".json").read_text())
        assert digest(p) == meta["sha256"]
        with np.load(p, allow_pickle=False) as z:
            blocks.append(z["vectors"])
    vectors = np.concatenate(blocks)
    assert len(vectors) == len(texts)
    index = dict(zip(texts, vectors, strict=True))
    selected = pd.read_csv(out / "private_selected.csv").set_index("audit_id")
    max_error = 0.0
    for mode in ("local", "scope"):
        recalculated = np.array(
            [
                float(index[r.item_text] @ index[getattr(r, mode + "_text")])
                for r in candidates.itertuples()
            ]
        )
        np.testing.assert_allclose(
            recalculated, candidates[mode + "_cosine"], rtol=0, atol=1e-7
        )
        max_error = max(
            max_error,
            float(np.max(np.abs(recalculated - candidates[mode + "_cosine"]))),
        )
        for key, f in candidates.groupby("audit_id"):
            # Do not call the ranking function under audit.
            ranked = f.sort_values(
                [mode + "_cosine", "event_id"], ascending=[False, True]
            )
            assert ranked.event_id.iloc[0] == selected.loc[key, mode + "_event_id"]
            ties = (
                np.abs(ranked[mode + "_cosine"] - ranked[mode + "_cosine"].iloc[0])
                <= 1e-7
            ).sum()
            assert ties == selected.loc[key, mode + "_near_ties"]
    nli = pd.read_csv(out / "private_nli_comparison.csv")
    probs = nli[["p_contradiction", "p_entailment", "p_neutral"]].to_numpy()
    assert np.isfinite(probs).all() and ((0 <= probs) & (probs <= 1)).all()
    np.testing.assert_allclose(probs.sum(1), 1, atol=2e-7)
    assert not nli.duplicated(["pair_id", "view"]).any()
    for _, f in nli.groupby(["sentence", "item_text"]):
        assert f.nli.nunique() == 1 and f.p_entailment.nunique() == 1
    cats = nli.pivot(index="pair_id", columns="view", values="nli")
    original = nli[nli.view == "original"].set_index("pair_id").loc[annotation.index]
    original_equal = int((original.nli == annotation.nli).sum())
    original_error = float(
        (original.p_entailment - annotation.p_entailment).abs().max()
    )
    assert original_equal == len(annotation) and original_error < 1e-6
    transitions = pd.crosstab(cats.local_selected, cats.local_selected_scope).to_dict(
        "index"
    )
    assert transitions == summary["nli_class_transition"]
    changes = int((cats.local_selected != cats.local_selected_scope).sum())
    assert changes == summary["scope_delta"]["class_changes"]
    source_links = pd.read_csv(GRAPH / "private_event_unit_join.csv").set_index(
        ["event_id", "view"]
    )
    links = pd.read_csv(out / "private_selected_grain_join.csv")
    assert (
        len(links) == len(annotation) * 5
        and not links.duplicated(["audit_id", "view"]).any()
    )
    for r in links.itertuples():
        assert r.event_id == selected.loc[r.audit_id, "local_event_id"]
        assert r.unit_ids == source_links.loc[(r.event_id, r.view), "unit_ids"]
    result = {
        "status": "PASS",
        "pairs": len(annotation),
        "candidate_events": len(records),
        "raw_segments": spans_checked,
        "cosine_error": max_error,
        "nli_rows": len(nli),
        "grain_join_rows": len(links),
        "scope_class_changes": changes,
        "original_class_replay_equal": original_equal,
        "original_max_entailment_error": original_error,
        "transition": transitions,
        "source_only": True,
        "new_model_calls": 0,
        "summary_sha256": digest(out / "summary.json"),
        "audit_code_sha256": digest(Path(__file__)),
    }
    if args.compare:
        equality = {
            name: digest(out / name) == digest(args.compare / name)
            for name in (
                "private_candidate_views.jsonl",
                "private_candidates.csv",
                "private_nli_cache.csv",
                "private_nli_comparison.csv",
                "private_selected_grain_join.csv",
            )
        }
        assert all(equality.values())
        result["cold_repeat_equal"] = equality
    write(out / "audit.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "query_event_r2")
    parser.add_argument("--compare", type=Path)
    run(parser.parse_args())
