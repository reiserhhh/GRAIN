#!/usr/bin/env python3
"""Replay source selection and score arithmetic without extra model calls."""

from __future__ import annotations

import argparse
import ast
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.quote_source_sieve import explicit_quote_source
from grain_research.source_event_graph import validate_words
from grain_research.source_ownership import classify_source, select_observations
from grain_research.vector_mechanisms.navigation import digest, read_json, write_json
from scripts.run_grain_navigation_baseline import checked_arrays
from scripts.run_grain_ownership_scoring import DEFAULT, OLD, REFERENCE, verify


def run(out):
    """Check 700 native/affine endpoints and both 300-case selection routes."""
    verify(out)
    child = out / "sieve_v1"
    contract = read_json(child / "manifest.json")
    for name, expected in contract["code"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Changed frozen sieve code")
    if digest(out / "summary.json") != contract["parent_summary"]:
        raise ValueError("Changed parent source result")
    for folder in (out, child):
        for name, expected in read_json(folder / "summary.json")["files"].items():
            if digest(folder / name) != expected:
                raise ValueError("Changed result before audit")
    records = read_json(out / "private_records.json")
    record_map = {r["record_id"]: r for r in records}
    decisions = {"nli": {}, "sieve": {}}
    saved_sieve = pd.read_csv(child / "source_decisions.csv").set_index("record_id")
    requests = read_json(out / "private_nli_requests.json")
    address_count = 0
    for i, r in enumerate(records):
        request = requests[i]
        if request["record_id"] != r["record_id"] or not request["eligible"]:
            raise ValueError("This packet requires complete, matched NLI requests")
        saved = read_json(out / "private_nli" / f"record_{i:04d}.json")
        original = classify_source(saved["probabilities"])
        if saved["record_id"] != r["record_id"] or any(
            saved[k] != v for k, v in original.items()
        ):
            raise ValueError("NLI decision differs from saved probability rule")
        parsed = read_json(child / "private_parses" / f"record_{i:04d}.json")
        if (
            parsed["record_id"] != r["record_id"]
            or parsed["text_sha256"] != hashlib.sha256(r["text"].encode()).hexdigest()
        ):
            raise ValueError("Parser text changed")
        for sentence in parsed["sentences"]:
            validate_words(sentence, r["text"])
            address_count += len(sentence)
        a = r["start"] if r["domain"] == "natural" else r["text"].index(r["quote"])
        b = r["end"] if r["domain"] == "natural" else a + len(r["quote"])
        if r["text"][a:b] != r["quote"]:
            raise ValueError("Quote address changed")
        repaired = explicit_quote_source(r["text"], parsed["sentences"], a, b, original)
        actual = saved_sieve.loc[r["record_id"]]
        for key in ("source_owner", "mechanism", "reason"):
            if repaired[key] != actual[key]:
                raise ValueError("Sieve rule replay mismatch")
        if repaired["cue_span"] is not None:
            if a <= repaired["cue_span"][0] < b:
                raise ValueError(
                    "Quote-internal source incorrectly treated as outer cue"
                )
            if ast.literal_eval(actual.source_spans) != repaired["source_spans"]:
                raise ValueError("Source address mismatch")
        decisions["nli"][r["record_id"]] = original
        decisions["sieve"][r["record_id"]] = repaired

    with np.load(REFERENCE / "reference.npz", allow_pickle=False) as archive:
        coefficients = dict(archive)
    old_design = {d["index"]: d for d in read_json(OLD / "private_design.json")}
    endpoints = []
    for r in records:
        if r["domain"] == "synthetic" and r["old_index"] is not None:
            endpoints.append(
                (r["record_id"], OLD, old_design[r["old_index"]], r["old_world_id"])
            )
    endpoints.extend(
        (d["case_id"], out, d, None) for d in read_json(out / "private_design.json")
    )
    question_tokens = read_json(OLD / "manifest.json")["question_tokens"]
    points, error = {}, 0.0
    for key, folder, d, world_id in endpoints:
        meta, z = checked_arrays(
            folder / "private_packets" / f"unit_{d['index']:04d}.json"
        )
        if meta["query"] != d["query"] or (
            meta["world_id"] != world_id if world_id else meta["case_id"] != key
        ):
            raise ValueError("Endpoint identity mismatch")
        q = d["query"]
        if (
            np.asarray(d["input_ids"])[np.asarray(d["mask"], bool)].tolist()
            != question_tokens[q]
        ):
            raise ValueError("Query token coordinate mismatch")
        native = meta["native"]
        p = softmax(np.asarray(native["candidate_logits"], float))
        point = float(p @ np.asarray(d["alphabet"]["values"], float))
        mass = float(
            np.exp(
                logsumexp(native["candidate_logits"])
                - native["full_vocab_log_normalizer"]
            )
        )
        error = max(
            error,
            abs(point - native["conditional_score"]),
            abs(mass - native["candidate_mass"]),
            float(np.max(np.abs(p - native["conditional_probabilities"]))),
        )
        affine = float(
            coefficients[q + "__b"]
            + np.sum(
                coefficients[q + "__G"]
                * (z["P"].astype(float) - coefficients[q + "__Z"])
            )
            + coefficients[q + "__g"] @ (z["Q"].astype(float) - coefficients[q + "__h"])
        )
        points[key] = {"native": point, "affine": affine}

    rows_checked, context_checked = 0, 0
    comparison, contrasts = [], []
    for route, folder in (("nli", out), ("sieve", child)):
        frame = pd.read_csv(folder / "mixed_scores.csv")
        table = frame.set_index(["case_id", "readout"])
        for case in read_json(out / "private_mixtures.json"):
            ids = case["record_ids"]
            selected = select_observations(
                [record_map[i]["text"] for i in ids], [decisions[route][i] for i in ids]
            )
            picked = [ids[i] for i in selected["selected_indices"]]
            contexts = [ids[i] for i in selected["context_indices"]]
            if selected["all_texts_retained"] != [
                record_map[i]["text"] for i in ids
            ] or set(picked + contexts) != set(ids):
                raise ValueError("Discarded source or context")
            context_checked += 1
            for reader in ("native", "affine"):
                row = table.loc[(case["case_id"], reader)]
                expected = (
                    points[picked[0]][reader]
                    if len(picked) == 1
                    else points[case["case_id"]][reader]
                    if len(picked) == 2
                    else None
                )
                if (
                    ast.literal_eval(row.selected_ids) != picked
                    or ast.literal_eval(row.context_ids) != contexts
                ):
                    raise ValueError("Selected/context identities changed")
                if (
                    bool(row.correct_self_only) != (picked == [case["oracle_self_id"]])
                    or row.status != selected["status"]
                ):
                    raise ValueError("Selection status mismatch")
                if expected is None:
                    if not pd.isna(row.selected):
                        raise ValueError("Unobserved score was imputed")
                elif pd.isna(row.selected):
                    raise ValueError("Observed value lost")
                else:
                    error = max(error, abs(expected - row.selected))
                error = max(
                    error,
                    abs(points[case["case_id"]][reader] - row.unfiltered),
                    abs(points[case["oracle_self_id"]][reader] - row.oracle_self),
                )
                rows_checked += 1
        for (wrapper, reader), group in frame.groupby(
            ["wrapper", "readout"], sort=True
        ):
            observed = group.selected.notna()
            comparison.append(
                {
                    "route": route,
                    "wrapper": wrapper,
                    "readout": reader,
                    "cases": len(group),
                    "observed": int(observed.sum()),
                    "correct_self_only": int(group.correct_self_only.sum()),
                    "raw_mae_matched": float(
                        abs(
                            group.loc[observed, "unfiltered"]
                            - group.loc[observed, "oracle_self"]
                        ).mean()
                    )
                    if observed.any()
                    else None,
                    "selected_mae_observed": float(
                        abs(
                            group.loc[observed, "selected"]
                            - group.loc[observed, "oracle_self"]
                        ).mean()
                    )
                    if observed.any()
                    else None,
                }
            )
        for keys, group in frame.groupby(
            ["query", "wrapper", "readout", "polarity"], sort=True
        ):
            g = group.set_index("order")
            contrasts.append(
                dict(
                    zip(("query", "wrapper", "readout", "polarity"), keys, strict=True)
                )
                | {
                    "route": route,
                    "raw_order_delta": float(
                        g.loc["self_first", "unfiltered"]
                        - g.loc["other_first", "unfiltered"]
                    ),
                    "selected_order_delta": float(
                        g.loc["self_first", "selected"]
                        - g.loc["other_first", "selected"]
                    ),
                    "oracle_order_delta": float(
                        g.loc["self_first", "oracle_self"]
                        - g.loc["other_first", "oracle_self"]
                    ),
                }
            )
    pd.DataFrame(comparison).to_csv(out / "audited_route_comparison.csv", index=False)
    pd.DataFrame(contrasts).to_csv(out / "audited_order_comparison.csv", index=False)
    result = {
        "status": "PASS" if error < 1e-10 else "FAIL",
        "max_numeric_error": error,
        "endpoint_logit_and_point_checks": len(endpoints),
        "nli_records": len(records),
        "sieve_records": len(records),
        "word_addresses": address_count,
        "mixed_score_rows": rows_checked,
        "complete_source_ledgers": context_checked,
        "additional_model_calls": 0,
        "independent_executor": False,
        "auditor_sha256": digest(Path(__file__)),
        "files": {
            n: digest(out / n)
            for n in ("audited_route_comparison.csv", "audited_order_comparison.csv")
        },
    }
    write_json(out / "cold_replay_audit.json", result)
    print(result)
    if result["status"] != "PASS":
        raise ValueError("Numeric replay failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    run(parser.parse_args().out)
