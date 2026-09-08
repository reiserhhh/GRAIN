#!/usr/bin/env python3
"""Run NAV-03's explicit program on existing real input tokens, without neural states."""

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
from grain_research.vector_mechanisms.projection_program import METHODS, score_token_ids
from scripts.analyze_grain_all_factor_source import paired_quality
from scripts.run_grain_projection_program import load_program


def run(program_dir, out, seconds=1200):
    """Freeze all token-only predictions before joining human or full-prefix scores."""
    program, table, charts = load_program(program_dir)
    source = BASE / "all_factor_source_m29_v1"
    mood = BASE / "counseling_mood_m30c_v1"
    inputs = {
        "source": source / "private_design.json",
        "mood": mood / "private_design.json",
    }
    units = []
    for d in read_json(inputs["source"]):
        if d["query"] in ("extraversion", "TF_cont"):
            units.append(
                {
                    "index": len(units),
                    "dataset": "PANDORA_source",
                    "author_id": d["author_id"],
                    "query": d["query"],
                    "views": d["views"],
                }
            )
    for d in read_json(inputs["mood"]):
        if d["query"] == "adapted_dejected5":
            units.append(
                {
                    "index": len(units),
                    "dataset": d["view"],
                    "author_id": d["participant_code"],
                    "query": d["query"],
                    "views": [
                        {
                            "input_ids": d["input_ids"],
                            "question_mask": d["question_mask"],
                        }
                    ],
                }
            )
    if len(units) != 369:
        raise ValueError("Expected277 source units and92 counseling views")
    if not out.exists():
        out.mkdir(parents=True)
        (out / "private_units").mkdir()
        write_json(
            out / "manifest.json",
            {
                "protocol": "NAV03_TOKEN_ONLY_REAL_TRANSFER_V1",
                "program_sha256": digest(program_dir / "program.json"),
                "source_sha256": {k: digest(p) for k, p in inputs.items()},
                "code_sha256": {
                    p: digest(ROOT / p)
                    for p in (
                        "scripts/evaluate_grain_projection_external.py",
                        "grain_research/vector_mechanisms/projection_program.py",
                    )
                },
                "methods": list(METHODS),
                "units": len(units),
                "neural_states_used": False,
                "human_label_fit": False,
            },
        )
    m = read_json(out / "manifest.json")
    if m["program_sha256"] != digest(program_dir / "program.json") or any(
        digest(inputs[k]) != sha for k, sha in m["source_sha256"].items()
    ):
        raise ValueError("Fixed program/input changed")
    for p, sha in m["code_sha256"].items():
        if digest(ROOT / p) != sha:
            raise ValueError("Frozen implementation changed")
    start = time.monotonic()
    rows = []
    with threadpool_limits(limits=2):
        for u in units:
            path = out / "private_units" / f"unit_{u['index']:04d}.json"
            if path.exists():
                saved = read_json(path)
                if any(
                    saved[k] != u[k] for k in ("index", "author_id", "query", "dataset")
                ):
                    raise ValueError("Unit identity changed")
                rows.extend(saved["rows"])
                continue
            if time.monotonic() - start >= seconds:
                write_json(
                    out / "run_status.json",
                    {
                        "status": "PAUSED_RESOURCE_BOUNDARY",
                        "units_completed": len(rows) // len(METHODS),
                        "neural_calls": 0,
                    },
                )
                print(read_json(out / "run_status.json"), flush=True)
                return
            q = u["query"]
            values = []
            for method in METHODS:
                predictions = []
                for view in u["views"]:
                    ids, mask = view["input_ids"], view["question_mask"]
                    if np.asarray(ids)[mask].tolist() != program["question_tokens"][q]:
                        raise ValueError("Token-only query differs from constructor")
                    predictions.append(
                        score_token_ids(
                            ids, mask, table, charts[q], program["anchors"][q], method
                        )
                    )
                values.append(
                    {k: u[k] for k in ("dataset", "author_id", "query")}
                    | {
                        "method": method,
                        **{
                            k: float(np.mean([p[k] for p in predictions]))
                            for k in ("projection1", "projection2", "score")
                        },
                    }
                )
            write_json(
                path,
                {k: u[k] for k in ("index", "author_id", "query", "dataset")}
                | {"rows": values},
            )
            rows.extend(values)
            if (u["index"] + 1) % 20 == 0:
                print(
                    {
                        "status": "TOKEN_ONLY_SCORING",
                        "complete": u["index"] + 1,
                        "units": len(units),
                        "seconds": time.monotonic() - start,
                    },
                    flush=True,
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "private_predictions.csv", index=False)
    write_json(
        out / "prediction_freeze.json",
        {
            "sha256": digest(out / "private_predictions.csv"),
            "neural_states_used": False,
            "human_criteria_read_before_freeze": False,
            "torch_imported": "torch" in sys.modules,
        },
    )
    # External labels and neural reference scores are loaded only after freezing.
    ref = BASE / "navigation_v1/NAV-10/reference_transfer_r1"
    full_source = pd.read_csv(
        ref / "private_source_external.csv", dtype={"author_id": str}
    ).query("reference=='R_synth'")
    joined = frame[frame.dataset == "PANDORA_source"].merge(
        full_source[["author_id", "query", "score", "y_true", "prediction"]].rename(
            columns={"score": "full_function"}
        ),
        on=["author_id", "query"],
        validate="many_to_one",
    )
    if len(joined) != 831:
        raise ValueError("Missing source criterion rows")
    full_mood = (
        pd.read_csv(ref / "private_counseling_predictions.csv")
        .query("reference=='R_synth'")
        .rename(
            columns={
                "participant_code": "author_id",
                "view": "dataset",
                "score": "full_function",
            }
        )
    )
    mood_joined = frame[frame.dataset != "PANDORA_source"].merge(
        full_mood[["author_id", "dataset", "query", "full_function", "native_score"]],
        on=["author_id", "dataset", "query"],
        validate="many_to_one",
    )
    if len(mood_joined) != 276:
        raise ValueError("Missing counseling reference rows")
    criterion_path = mood / "private_mood_scores.csv"
    if (
        digest(criterion_path)
        != read_json(mood / "manifest.json")["criterion_cache_sha256"]
    ):
        raise ValueError("Mood criterion changed")
    criteria = pd.read_csv(criterion_path).pivot(
        index="participant_code", columns="scale", values="score"
    )
    metrics = []
    for (q, method), g in joined.groupby(["query", "method"]):
        metrics.append(
            {
                "dataset": "PANDORA_source",
                "query": q,
                "method": method,
                "full_function_r": quality(g.full_function, g.score)["r"],
                **paired_quality(
                    g.score, g.prediction, g.y_true, binary=q == "TF_cont"
                ),
            }
        )
    for (view, method), g in mood_joined.groupby(["dataset", "method"]):
        criterion = f"{'post' if view == 'free_chat' else 'mid'}_adapted_dejected5"
        y = criteria.loc[g.author_id, criterion].to_numpy()
        metrics.append(
            {
                "dataset": view,
                "query": "adapted_dejected5",
                "method": method,
                "full_function_r": quality(g.full_function, g.score)["r"],
                **paired_quality(g.score, g.native_score, (y - 1) / 5),
            }
        )
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(out / "metrics.csv", index=False)
    write_json(
        out / "run_status.json",
        {
            "status": "COMPLETE_TOKEN_ONLY_EXTERNAL",
            "units_completed": len(units),
            "neural_calls": 0,
            "torch_imported": "torch" in sys.modules,
            "seconds_this_invocation": time.monotonic() - start,
        },
    )
    columns = [
        "dataset",
        "query",
        "method",
        "n",
        "full_function_r",
        "native_fidelity_r",
        "function_external_r",
        "function_auc",
    ]
    print(metrics[columns].round(5).to_string(index=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--program-dir",
        type=Path,
        default=BASE / "navigation_v1/NAV-03/input_pullback_r1",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "navigation_v1/NAV-10/projection_transfer_r1",
    )
    p.add_argument("--seconds", type=float, default=1200)
    a = p.parse_args()
    if not 0 < a.seconds <= 1200 or not a.output_dir.resolve().is_relative_to(
        (BASE / "navigation_v1").resolve()
    ):
        p.error("Use bounded local navigation output")
    run(a.program_dir.resolve(), a.output_dir.resolve(), a.seconds)
