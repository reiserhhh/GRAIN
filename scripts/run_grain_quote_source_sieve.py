#!/usr/bin/env python3
"""Explicit reporting-source repair, preserving the frozen NLI-only baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.quote_source_sieve import explicit_quote_source
from grain_research.source_event_graph import stanza_words
from grain_research.source_ownership import select_observations
from grain_research.vector_mechanisms.navigation import (
    digest,
    load_functions,
    read_json,
    write_json,
)
from scripts.fetch_grain_syntax_reference import MODELS, PACKAGES, PROCESSORS
from scripts.run_grain_navigation_baseline import checked_arrays
from scripts.run_grain_ownership_scoring import DEFAULT, GRAPH, OLD, REFERENCE, verify
from scripts.run_grain_source_event_graph import verify_models
from scripts.run_grain_upstream_routes import collector_guard

PLAN = ROOT / "docs/GRAIN_QUOTE_SOURCE_SIEVE_ADDENDUM.md"


def run(out, seconds=1200):
    """Parse source cues, replay selection and retain every failure and abstention."""
    import torch

    parent = out.parent
    verify(parent)
    baseline = read_json(parent / "summary.json")
    for name, expected in baseline["files"].items():
        if digest(parent / name) != expected:
            raise ValueError("NLI-only output changed")
    verify_models()
    contract = {
        "parent_manifest": digest(parent / "manifest.json"),
        "parent_summary": digest(parent / "summary.json"),
        "plan": digest(PLAN),
        "code": {
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                Path(__file__),
                ROOT / "grain_research/quote_source_sieve.py",
                ROOT / "grain_research/source_event_graph.py",
                ROOT / "grain_research/source_ownership.py",
            )
        },
        "parser_inventory": digest(MODELS / "grain_inventory.json"),
        "natural_parses": digest(GRAPH / "private_parses.jsonl"),
        "developmental_after_nli_failure": True,
        "training": False,
        "additional_native_calls": 0,
    }
    if out.exists():
        if read_json(out / "manifest.json") != contract:
            raise ValueError("Sieve packet changed")
    else:
        out.mkdir(parents=True)
        (out / "private_parses").mkdir()
        write_json(out / "manifest.json", contract)
    sys.path.insert(0, str(PACKAGES))
    import stanza

    if stanza.__version__ != "1.11.0":
        raise ValueError("Wrong parser version")
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
    natural = {
        r["source_row"]: r
        for r in (json.loads(l) for l in (GRAPH / "private_parses.jsonl").open())
    }
    records = read_json(parent / "private_records.json")
    prior = pd.read_csv(parent / "source_decisions.csv").set_index("record_id")
    start, calls = time.monotonic(), 0
    rows = []
    for i, r in enumerate(records):
        path = out / "private_parses" / f"record_{i:04d}.json"
        if path.exists():
            saved = read_json(path)
            if (
                saved["record_id"] != r["record_id"]
                or saved["text_sha256"]
                != hashlib.sha256(r["text"].encode()).hexdigest()
            ):
                raise ValueError("Parser resume source mismatch")
            words = saved["sentences"]
        else:
            if time.monotonic() - start >= seconds:
                write_json(
                    out / "run_status.json",
                    {
                        "status": "PAUSED_RESOURCE_BOUNDARY",
                        "parsed": i,
                        "new_parser_calls": calls,
                    },
                )
                return
            if r["domain"] == "natural":
                words = natural[r["source_row"]]["sentences"]
            else:
                words = stanza_words(pipeline(r["text"]), r["text"])
                calls += 1
            write_json(
                path,
                {
                    "record_id": r["record_id"],
                    "text_sha256": hashlib.sha256(r["text"].encode()).hexdigest(),
                    "sentences": words,
                },
            )
        a = r["start"] if r["domain"] == "natural" else r["text"].index(r["quote"])
        b = r["end"] if r["domain"] == "natural" else a + len(r["quote"])
        if r["text"][a:b] != r["quote"]:
            raise ValueError("Quote address mismatch")
        fallback = {
            "source_owner": prior.loc[r["record_id"], "source_owner"],
            "reason": prior.loc[r["record_id"], "reason"],
        }
        decision = explicit_quote_source(r["text"], words, a, b, fallback)
        rows.append(
            {
                "record_id": r["record_id"],
                "domain": r["domain"],
                "wrapper": r.get("wrapper"),
                "expected_owner": r["expected_owner"],
                "nli_owner": fallback["source_owner"],
                **decision,
            }
        )
        if (i + 1) % 100 == 0:
            print(
                {
                    "status": "PARSING",
                    "records": i + 1,
                    "new_parser_calls": calls,
                    "seconds": time.monotonic() - start,
                },
                flush=True,
            )
    decisions = {r["record_id"]: r for r in rows}
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "source_decisions.csv", index=False)
    functions = load_functions(
        REFERENCE / "reference.npz", queries=list(read_json(REFERENCE / "queries.json"))
    )
    scores = {}
    for r in records:
        if r["domain"] == "synthetic" and r["old_index"] is not None:
            meta, z = checked_arrays(
                OLD / "private_packets" / f"unit_{r['old_index']:04d}.json"
            )
            if meta["world_id"] != r["old_world_id"] or meta["query"] != r["query"]:
                raise ValueError("Old score identity mismatch")
            scores[r["record_id"]] = {
                "native": meta["native"]["conditional_score"],
                "affine": functions[r["query"]].parts(z["P"], z["Q"])["score"],
            }
    for d in read_json(parent / "private_design.json"):
        meta, z = checked_arrays(
            parent / "private_packets" / f"unit_{d['index']:04d}.json"
        )
        if meta["case_id"] != d["case_id"] or meta["query"] != d["query"]:
            raise ValueError("New score identity mismatch")
        scores[d["case_id"]] = {
            "native": meta["native"]["conditional_score"],
            "affine": functions[d["query"]].parts(z["P"], z["Q"])["score"],
        }
    rec = {r["record_id"]: r for r in records}
    mixed = []
    for case in read_json(parent / "private_mixtures.json"):
        ids = case["record_ids"]
        selection = select_observations(
            [rec[i]["text"] for i in ids], [decisions[i] for i in ids]
        )
        picked = [ids[i] for i in selection["selected_indices"]]
        values = (
            scores[picked[0]]
            if len(picked) == 1
            else scores[case["case_id"]]
            if len(picked) == 2
            else None
        )
        for readout in ("native", "affine"):
            mixed.append(
                {
                    k: case[k]
                    for k in ("case_id", "query", "wrapper", "polarity", "order")
                }
                | {
                    "readout": readout,
                    "unfiltered": scores[case["case_id"]][readout],
                    "selected": values[readout] if values else np.nan,
                    "oracle_self": scores[case["oracle_self_id"]][readout],
                    "selected_ids": picked,
                    "context_ids": [ids[i] for i in selection["context_indices"]],
                    "correct_self_only": picked == [case["oracle_self_id"]],
                    "status": selection["status"],
                }
            )
    mixed = pd.DataFrame(mixed)
    mixed.to_csv(out / "mixed_scores.csv", index=False)
    metrics = []
    for keys, g in mixed.groupby(["wrapper", "readout"], sort=True):
        valid = g.selected.notna()
        metrics.append(
            {
                "wrapper": keys[0],
                "readout": keys[1],
                "cases": len(g),
                "observed": int(valid.sum()),
                "correct_self_only": int(g.correct_self_only.sum()),
                "raw_mae_all": float(np.abs(g.unfiltered - g.oracle_self).mean()),
                "selected_mae_observed": float(
                    np.abs(
                        g.loc[valid, "selected"] - g.loc[valid, "oracle_self"]
                    ).mean()
                )
                if valid.any()
                else None,
            }
        )
    pd.DataFrame(metrics).to_csv(out / "scoring_summary.csv", index=False)
    classifications = []
    for wrapper, g in frame[frame.domain == "synthetic"].groupby("wrapper", sort=True):
        classifications.append(
            {
                "wrapper": wrapper,
                "records": len(g),
                "correct": int((g.source_owner == g.expected_owner).sum()),
                "unresolved": int((g.source_owner == "unresolved").sum()),
                "wrong": int(
                    (
                        (g.source_owner != g.expected_owner)
                        & (g.source_owner != "unresolved")
                    ).sum()
                ),
            }
        )
    natural_frame = frame[frame.domain == "natural"]
    files = ("source_decisions.csv", "mixed_scores.csv", "scoring_summary.csv")
    write_json(
        out / "summary.json",
        {
            "status": "EXPLICIT_SOURCE_ADAPTER_TESTED",
            "classification": classifications,
            "scoring": metrics,
            "natural_routes": natural_frame.source_owner.value_counts().to_dict(),
            "mechanisms": frame.mechanism.value_counts().to_dict(),
            "natural_mechanisms": natural_frame.mechanism.value_counts().to_dict(),
            "files": {n: digest(out / n) for n in files},
            "independent_replication": False,
            "scope": "same exposed synthetic packet, source-cue repair; natural sources lack independent annotation",
        },
    )
    execution = {
        "status": "COMPLETE",
        "records": len(records),
        "new_parser_calls": calls,
        "seconds": time.monotonic() - start,
    }
    if not (out / "first_completion.json").exists():
        write_json(out / "first_completion.json", execution)
    write_json(out / "run_status.json", execution)
    print(read_json(out / "summary.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT / "sieve_v1")
    parser.add_argument("--seconds", type=int, default=1200)
    args = parser.parse_args()
    if not 0 < args.seconds <= 1200 or args.out.parent.resolve() != DEFAULT.resolve():
        parser.error("Use bounded nested source-adapter output")
    with collector_guard(args.out):
        run(args.out, args.seconds)
