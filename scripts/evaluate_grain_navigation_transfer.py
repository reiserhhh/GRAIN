#!/usr/bin/env python3
"""NAV-10: transport synthetic-reference functions to existing permitted real states.

This evaluates the frozen reader, not a raw-text-independent upstream program.
MEPS remains external-only. Predictions are saved before any criterion is read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.factor_catalog import association
from grain_research.vector_mechanisms.navigation import (
    BASE,
    OUTPUT,
    digest,
    load_functions,
    read_json,
    write_json,
)
from scripts.analyze_grain_all_factor_source import paired_quality
from scripts.run_grain_navigation_baseline import checked_arrays


def score_source(source, functions, tokens):
    """Average the same three saved source-author views with no criterion access."""
    m = read_json(source / "manifest.json")
    if digest(source / "private_design.json") != m["input_sha256"]:
        raise ValueError("Source design changed")
    rows = []
    for d in read_json(source / "private_design.json"):
        q = d["query"]
        if q not in ("extraversion", "TF_cont"):
            continue
        for view in d["views"]:
            if (
                np.asarray(view["input_ids"])[
                    np.asarray(view["question_mask"], bool)
                ].tolist()
                != tokens[q]
            ):
                raise ValueError("Source query coordinate mismatch")
        meta, z = checked_arrays(
            source / "private_packets" / f"unit_{d['index']:04d}.json"
        )
        if any(
            meta[k] != d[k] for k in ("index", "query", "author_id", "source_group")
        ):
            raise ValueError("Source identity changed")
        if len(z["P"]) != 3 or len(z["Q"]) != 3:
            raise ValueError("Three matched source views required")
        for arm, funcs in functions.items():
            parts = [funcs[q].parts(P, Q) for P, Q in zip(z["P"], z["Q"], strict=True)]
            rows.append(
                {
                    "author_id": d["author_id"],
                    "query": q,
                    "reference": arm,
                    "source_group": d["source_group"],
                    **{k: float(np.mean([r[k] for r in parts])) for k in parts[0]},
                }
            )
    frame = pd.DataFrame(rows)
    if frame.duplicated(["author_id", "query", "reference"]).any() or len(frame) != 554:
        raise ValueError("Expected277 source authors x2 readers")
    return frame


def score_counseling(source, functions, tokens):
    """Read only saved de-identified states and native responses, not questionnaires."""
    m = read_json(source / "manifest.json")
    q = "adapted_dejected5"
    if digest(source / "private_design.json") != m["input_sha256"]:
        raise ValueError("Counseling design changed")
    if m["question_tokens"][q] != tokens[q]:
        raise ValueError("Counseling query coordinate mismatch")
    rows = []
    for d in read_json(source / "private_design.json"):
        if d["query"] != q:
            continue
        meta, z = checked_arrays(
            source / "private_packets" / f"cell_{d['index']:04d}.json"
        )
        if any(meta[k] != d[k] for k in ("index", "query", "participant_code", "view")):
            raise ValueError("Counseling identity changed")
        for arm, funcs in functions.items():
            rows.append(
                {k: d[k] for k in ("participant_code", "query", "view")}
                | {
                    "reference": arm,
                    "native_score": meta["native_score"],
                    **funcs[q].parts(z["P"], z["Q"]),
                }
            )
    frame = pd.DataFrame(rows)
    if (
        frame.duplicated(["participant_code", "query", "view", "reference"]).any()
        or len(frame) != 184
    ):
        raise ValueError("Expected46 counseling participants x2 views x2 readers")
    return frame


def evaluate(reference, out):
    """Freeze new predictions, then join existing permitted author-linked criteria."""
    if out.exists():
        raise FileExistsError("Preserve prior transfer run")
    if read_json(reference / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Reference construction must finish first")
    m, refs = (
        read_json(reference / "manifest.json"),
        read_json(reference / "reference_source_manifest.json"),
    )
    functions = {}
    for arm, name in (
        ("R_real", "real_reference.npz"),
        ("R_synth", "synthetic_reference.npz"),
    ):
        if digest(reference / name) != refs[arm]:
            raise ValueError("Frozen reference changed")
        functions[arm] = load_functions(reference / name)
    source, counseling = (
        BASE / "all_factor_source_m29_v1",
        BASE / "counseling_mood_m30c_v1",
    )
    for folder in (source, counseling):
        if read_json(folder / "run_status.json")["status"] != "COMPLETE":
            raise ValueError("Only completed cached real data permitted")
    source_scores = score_source(source, functions, m["question_tokens"])
    counseling_scores = score_counseling(counseling, functions, m["question_tokens"])
    out.mkdir(parents=True)
    source_scores.to_csv(out / "private_source_predictions.csv", index=False)
    counseling_scores.to_csv(out / "private_counseling_predictions.csv", index=False)
    write_json(
        out / "prediction_freeze.json",
        {
            "source_sha256": digest(out / "private_source_predictions.csv"),
            "counseling_sha256": digest(out / "private_counseling_predictions.csv"),
            "references": refs,
            "human_criteria_read_before_freeze": False,
        },
    )
    # Criteria access is deliberately below prediction persistence.
    source_labels = pd.read_csv(
        source / "private_external_links.csv", dtype={"author_id": str}
    )
    source_labels = source_labels[
        source_labels["query"].isin(["extraversion", "TF_cont"])
    ]
    joined = source_scores.merge(
        source_labels[["author_id", "query", "y_true", "prediction"]],
        on=["author_id", "query"],
        validate="many_to_one",
    )
    if len(joined) != len(source_scores):
        raise ValueError("Missing source labels")
    mood_path = counseling / "private_mood_scores.csv"
    if (
        digest(mood_path)
        != read_json(counseling / "manifest.json")["criterion_cache_sha256"]
    ):
        raise ValueError("Audited mood criteria changed")
    mood = pd.read_csv(mood_path).pivot(
        index="participant_code", columns="scale", values="score"
    )
    metrics, timing = [], []
    for (q, arm), group in joined.groupby(["query", "reference"]):
        metrics.append(
            {
                "dataset": "PANDORA_source",
                "query": q,
                "reference": arm,
                "criterion": q,
                "native_source": "historical_cached_native_not_same_runtime",
                **paired_quality(
                    group.score, group.prediction, group.y_true, binary=q == "TF_cont"
                ),
            }
        )
    for (view, arm), group in counseling_scores.groupby(["view", "reference"]):
        group = group.set_index("participant_code").sort_index()
        if set(group.index) != set(mood.index):
            raise ValueError("Mood participant mismatch")
        criterion = f"{'post' if view == 'free_chat' else 'mid'}_adapted_dejected5"
        y = mood.loc[group.index, criterion]
        metrics.append(
            {
                "dataset": view,
                "query": "adapted_dejected5",
                "reference": arm,
                "criterion": criterion,
                "native_source": "same_cached_state_forward",
                **paired_quality(group.score, group.native_score, (y - 1) / 5),
            }
        )
        for c in mood.columns:
            for readout in ("score", "native_score"):
                timing.append(
                    {
                        "view": view,
                        "reference": arm,
                        "readout": readout,
                        "criterion": c,
                        **association(group[readout], mood.loc[group.index, c]),
                    }
                )
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(out / "external_metrics.csv", index=False)
    pd.DataFrame(timing).to_csv(out / "state_occasion_links.csv", index=False)
    joined.to_csv(out / "private_source_external.csv", index=False)
    write_json(
        out / "summary.json",
        {
            "status": "COMPLETE_REFERENCE_TRANSPORT",
            "source_authors": 277,
            "counseling_participants": 46,
            "queries": 3,
            "new_neural_calls": 0,
            "reference_sources": 2,
            "meps_used_for_fit": False,
            "runtime": "cached_frozen_prefix_states; no independent upstream program yet",
            "external_data_previously_exposed": True,
            "source_criterion_sha256": digest(source / "private_external_links.csv"),
            "mood_criterion_sha256": digest(mood_path),
        },
    )
    columns = [
        "dataset",
        "query",
        "reference",
        "n",
        "native_fidelity_r",
        "function_external_r",
        "function_r_lower",
        "function_r_upper",
        "function_auc",
        "delta_r",
    ]
    table = metric_frame[columns].round(5).to_string(index=False)
    (out / "REPORT.md").write_text(
        "# NAV-10 first synthetic-reference transfer\n\n```text\n"
        + table
        + "\n```\n\nReference functions were frozen before external joins. Existing exposed cohorts are exploratory transport, not a new independent confirmation. PANDORA native uses the historical cached observer; backend effects are not mechanism effects. New upstream independence is not tested by cached-state replay.\n",
        encoding="utf-8",
    )
    print(table, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=OUTPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "navigation_v1/NAV-10/reference_transfer_r1",
    )
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to((BASE / "navigation_v1").resolve()):
        parser.error("Use ignored local navigation outputs")
    evaluate(args.reference_dir.resolve(), args.output_dir.resolve())
