#!/usr/bin/env python3
"""Freeze one source-span program on exposed external inputs before joining criteria."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import (
    BASE,
    digest,
    quality,
    read_json,
    write_json,
)
from grain_research.vector_mechanisms.source_span import predict_linear
from grain_research.vector_mechanisms.state_transport import pack_state, unpack_state
from scripts.analyze_grain_state_transport import external_quality
from scripts.run_grain_source_span_extension import DEFAULT, OLD, load_extension
from scripts.run_grain_state_transport import input_units, load_operators, load_runtime
from scripts.run_grain_upstream_routes import collector_guard


def collect(program, seconds):
    """Use no human criteria in raw-input execution; exclude construction authors."""
    models = load_extension(program)
    gate = program / "natural/analysis/decision.json"
    if not read_json(gate)["external_probe_all_queries"]:
        raise ValueError("Model-only response packet did not trigger external work")
    p, _old, functions = load_operators(OLD)
    units, sources = input_units("external")
    members = read_json(program / "private_membership.json")
    refs = set(members["reference_authors"])
    excluded = [
        u for u in units if u["dataset"] == "PANDORA_source" and u["author_id"] in refs
    ]
    eligible = [
        u
        for u in units
        if not (u["dataset"] == "PANDORA_source" and u["author_id"] in refs)
    ]
    if {u["author_id"] for u in excluded} != set(members["external_source_overlap"]):
        raise ValueError("Source-overlap exclusion changed")
    out = program / "external"
    manifest = {
        "protocol": "NAV02_SOURCE_SPAN_EXTERNAL_V1",
        "construction_sha256": digest(program / "construction_freeze.json"),
        "code_sha256": digest(Path(__file__)),
        "sources": sources,
        "model_only_gate_sha256": digest(gate),
        "eligible_units": len(eligible),
        "excluded_units": len(excluded),
        "human_labels_before_freeze": False,
        "MEPS_use": "external-only; no fitting or route selection",
    }
    if out.exists():
        if read_json(out / "manifest.json") != manifest:
            raise ValueError("External packet changed")
        if read_json(out / "run_status.json")["status"] == "COMPLETE":
            print(read_json(out / "run_status.json"), flush=True)
            return
    else:
        out.mkdir()
        (out / "private_units").mkdir()
        write_json(out / "manifest.json", manifest)
        write_json(
            out / "private_membership.json",
            {
                "excluded_source_authors": sorted({u["author_id"] for u in excluded}),
                "eligible_identity": [
                    {k: u[k] for k in ("index", "author_id", "dataset", "query")}
                    for u in eligible
                ],
            },
        )
        write_json(
            out / "run_status.json", {"status": "PREPARED", "calls": 0, "seconds": 0}
        )
    prior = read_json(out / "run_status.json")
    _, runtime = load_runtime(OLD)
    start, rows = time.monotonic(), []
    with threadpool_limits(limits=2):
        for u in eligible:
            path = out / "private_units" / f"unit_{u['index']:04d}.json"
            identity = {k: u[k] for k in ("index", "author_id", "dataset", "query")}
            if path.exists():
                saved = read_json(path)
                if saved["identity"] != identity:
                    raise ValueError("External unit identity changed")
                rows.append(saved["row"])
                continue
            if time.monotonic() - start > seconds:
                write_json(
                    out / "run_status.json",
                    {
                        "status": "PAUSED_RESOURCE_BOUNDARY",
                        "calls": prior["calls"] + runtime.calls,
                        "seconds": prior["seconds"] + time.monotonic() - start,
                        "units": len(rows),
                    },
                )
                return
            q, values = u["query"], []
            for view in u["views"]:
                ids, mask = view["input_ids"], np.asarray(view["question_mask"], bool)
                if np.asarray(ids)[mask].tolist() != p["question_tokens"][q]:
                    raise ValueError("External question coordinates changed")
                h = runtime.capture(ids)[3]
                y = predict_linear(models[q], pack_state(h[mask], h[-1]))
                values.append(
                    functions[q].parts(*unpack_state(y, p["state_shapes"][q]))
                )
            row = {
                **identity,
                "readout": "source_span",
                **{k: float(np.mean([v[k] for v in values])) for k in values[0]},
            }
            write_json(path, {"identity": identity, "row": row})
            rows.append(row)
            if len(rows) % 60 == 0:
                print(
                    {
                        "units": len(rows),
                        "total": len(eligible),
                        "seconds": time.monotonic() - start,
                    },
                    flush=True,
                )
    pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)
    write_json(
        out / "prediction_freeze.json",
        {
            "sha256": digest(out / "predictions.csv"),
            "construction_sha256": manifest["construction_sha256"],
            "human_labels_read": False,
        },
    )
    write_json(
        out / "run_status.json",
        {
            "status": "COMPLETE",
            "calls": prior["calls"] + runtime.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "units": len(rows),
        },
    )
    print(read_json(out / "run_status.json"), flush=True)


def analyze(program):
    """Compare all frozen candidates on identical exposed human-endpoint membership."""
    load_extension(program)
    out = program / "external"
    if read_json(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Finish predictions before criteria")
    if (
        digest(out / "predictions.csv")
        != read_json(out / "prediction_freeze.json")["sha256"]
    ):
        raise ValueError("Predictions changed")
    analysis = out / "analysis"
    if analysis.exists():
        raise FileExistsError("Choose a new analysis")
    new = pd.read_csv(out / "predictions.csv", dtype={"author_id": str})
    keys = ["dataset", "author_id", "query"]
    expected = set(map(tuple, new[keys].to_numpy()))
    sources = {
        "synthetic": BASE / "navigation_v1/NAV-10/state_external_r1/predictions.csv",
        "token": BASE
        / "navigation_v1/NAV-10/projection_transfer_r1/private_predictions.csv",
        "context4": BASE
        / "navigation_v1/NAV-10/context_external_r2/private_predictions.csv",
    }
    frames, hashes = [new], {}
    for name, path in sources.items():
        if digest(path) != read_json(path.parent / "prediction_freeze.json")["sha256"]:
            raise ValueError("Comparator prediction freeze changed")
        frame = pd.read_csv(path, dtype={"author_id": str})
        if name == "synthetic":
            frame = frame.query("readout == 'transport1'")
        else:
            frame = frame.query(
                "method == 'semantic_order'"
                if name == "token"
                else "method == 'context4_mean'"
            )
        frame = frame[[tuple(row) in expected for row in frame[keys].to_numpy()]]
        if (
            len(frame) != len(new)
            or set(map(tuple, frame[keys].to_numpy())) != expected
        ):
            raise ValueError("Comparator membership differs")
        frames.append(frame.assign(readout=name)[keys + ["readout", "score"]])
        hashes[name] = digest(path)
    frame = pd.concat(frames, ignore_index=True)
    ref = BASE / "navigation_v1/NAV-10/reference_transfer_r1"
    source = pd.read_csv(
        ref / "private_source_external.csv", dtype={"author_id": str}
    ).query("reference=='R_synth'")
    mood = pd.read_csv(
        ref / "private_counseling_predictions.csv", dtype={"participant_code": str}
    ).query("reference=='R_synth'")
    joined = frame[frame.dataset == "PANDORA_source"].merge(
        source[["author_id", "query", "score", "y_true", "prediction"]].rename(
            columns={"score": "full_function", "prediction": "native"}
        ),
        on=["author_id", "query"],
        validate="many_to_one",
    )
    chat = frame[frame.dataset != "PANDORA_source"].merge(
        mood[["participant_code", "view", "query", "score", "native_score"]].rename(
            columns={
                "participant_code": "author_id",
                "view": "dataset",
                "score": "full_function",
                "native_score": "native",
            }
        ),
        on=keys,
        validate="many_to_one",
    )
    if len(joined) + len(chat) != len(frame):
        raise ValueError("External endpoint join dropped observations")
    criterion = BASE / "counseling_mood_m30c_v1/private_mood_scores.csv"
    if (
        digest(criterion)
        != read_json(criterion.parent / "manifest.json")["criterion_cache_sha256"]
    ):
        raise ValueError("Criterion cache changed")
    labels = pd.read_csv(criterion, dtype={"participant_code": str}).pivot(
        index="participant_code", columns="scale", values="score"
    )
    chat["y_true"] = [
        (
            float(
                labels.loc[
                    r.author_id,
                    f"{'post' if r.dataset == 'free_chat' else 'mid'}_adapted_dejected5",
                ]
            )
            - 1
        )
        / 5
        for r in chat.itertuples()
    ]
    all_rows = pd.concat([joined, chat], ignore_index=True)
    if (
        all_rows.duplicated(keys + ["readout"]).any()
        or all_rows[["score", "native", "y_true"]].isna().any().any()
    ):
        raise ValueError("Duplicated or missing external scores")
    metrics, comparisons = [], []
    for (dataset, q, readout), g in all_rows.groupby(["dataset", "query", "readout"]):
        metrics.append(
            {
                "dataset": dataset,
                "query": q,
                "readout": readout,
                "full_function_r": quality(g.full_function, g.score)["r"],
                **external_quality(g.score, g.native, g.y_true, binary=q == "TF_cont"),
            }
        )
    for (dataset, q), g in all_rows.groupby(["dataset", "query"]):
        pivot = g.pivot(index="author_id", columns="readout", values="score")
        target = (
            g.drop_duplicates("author_id")
            .set_index("author_id")
            .reindex(pivot.index)
            .y_true
        )
        for baseline in ("synthetic", "token", "context4"):
            comparisons.append(
                {
                    "dataset": dataset,
                    "query": q,
                    "baseline": baseline,
                    **external_quality(
                        pivot.source_span,
                        pivot[baseline],
                        target,
                        binary=q == "TF_cont",
                    ),
                }
            )
    analysis.mkdir()
    all_rows.to_csv(analysis / "private_joined.csv", index=False)
    pd.DataFrame(metrics).to_csv(analysis / "external_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(
        analysis / "paired_candidate_comparisons.csv", index=False
    )
    write_json(
        analysis / "summary.json",
        {
            "status": "COMPLETE_EXTERNAL_SOURCE_SPAN_PACKET",
            "prediction_sha256": digest(out / "predictions.csv"),
            "criterion_sha256": digest(criterion),
            "comparators": hashes,
            "analysis_code_sha256": digest(Path(__file__)),
            "external_membership": read_json(out / "manifest.json")["eligible_units"],
            "construction_overlap_excluded": read_json(out / "manifest.json")[
                "excluded_units"
            ],
            "fit_or_sign_selection_from_human_labels": False,
        },
    )
    print(
        pd.DataFrame(metrics)[
            [
                "dataset",
                "query",
                "readout",
                "n",
                "full_function_r",
                "function_external_r",
                "function_auc",
            ]
        ]
        .round(4)
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "analyze"))
    parser.add_argument("--program", type=Path, default=DEFAULT)
    parser.add_argument("--seconds", type=int, default=1200)
    args = parser.parse_args()
    with collector_guard(args.program / "external"):
        with threadpool_limits(limits=2):
            collect(
                args.program, args.seconds
            ) if args.action == "collect" else analyze(args.program)
