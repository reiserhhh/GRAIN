#!/usr/bin/env python3
"""Evaluate generated P/Q against the actual teacher only after prediction freeze."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import (
    BASE,
    OUTPUT,
    digest,
    quality,
    read_json,
    write_json,
)
from grain_research.vector_mechanisms.state_transport import pack_state, state_errors
from scripts.analyze_grain_all_factor_source import paired_quality
from scripts.analyze_grain_evidence_packets import compare_responses, response_rows
from scripts.run_grain_navigation_baseline import checked_arrays
from scripts.run_grain_state_transport import DEFAULT, load_operators


def external_quality(score, native, target, binary=False):
    """Keep constant-state controls with undefined correlation, never a fake zero."""
    f, n, y = (np.asarray(v, float) for v in (score, native, target))
    if f.ndim != 1 or f.shape != n.shape or f.shape != y.shape:
        raise ValueError("Aligned endpoint vectors required")
    if not np.isfinite([f, n, y]).all():
        raise ValueError("Nonfinite endpoint")
    if np.ptp(f) > 1e-12:
        return paired_quality(f, n, y, binary=binary)
    result = {
        "n": len(f),
        "native_fidelity_r": None,
        "function_external_r": None,
        "native_external_r": quality(y, n)["r"],
        "delta_r": None,
        "function_mae": quality(y, f)["mae"],
        "function_rmse": quality(y, f)["rmse"],
        "bootstrap_valid": 0,
        "correlation_status": "undefined_constant_control",
    }
    if binary:
        from sklearn.metrics import roc_auc_score

        result["function_auc"] = 0.5
        result["native_auc"] = float(roc_auc_score(y, n))
        result["delta_auc"] = 0.5 - result["native_auc"]
    return result


def external_links(frame, out):
    """Reuse exact, exposed source/occasion endpoints after the program is frozen."""
    ref = BASE / "navigation_v1/NAV-10/reference_transfer_r1"
    source = pd.read_csv(
        ref / "private_source_external.csv", dtype={"author_id": str}
    ).query("reference=='R_synth'")
    full_mood = pd.read_csv(
        ref / "private_counseling_predictions.csv", dtype={"participant_code": str}
    ).query("reference=='R_synth'")
    a = frame[frame.dataset == "PANDORA_source"].merge(
        source[["author_id", "query", "score", "y_true", "prediction"]].rename(
            columns={"score": "full_function"}
        ),
        on=["author_id", "query"],
        validate="many_to_one",
    )
    b = frame[frame.dataset != "PANDORA_source"].merge(
        full_mood[
            ["participant_code", "view", "query", "score", "native_score"]
        ].rename(
            columns={
                "participant_code": "author_id",
                "view": "dataset",
                "score": "full_function",
            }
        ),
        on=["author_id", "dataset", "query"],
        validate="many_to_one",
    )
    if len(a) != 277 * 4 or len(b) != 92 * 4:
        raise ValueError("External linkage changed")
    criterion = BASE / "counseling_mood_m30c_v1/private_mood_scores.csv"
    if (
        digest(criterion)
        != read_json(criterion.parent / "manifest.json")["criterion_cache_sha256"]
    ):
        raise ValueError("External criterion changed")
    labels = pd.read_csv(criterion).pivot(
        index="participant_code", columns="scale", values="score"
    )
    rows = []
    for (q, method), g in a.groupby(["query", "readout"]):
        rows.append(
            {
                "dataset": "PANDORA_source",
                "query": q,
                "readout": method,
                "full_function_r": quality(g.full_function, g.score)["r"],
                **external_quality(
                    g.score, g.prediction, g.y_true, binary=q == "TF_cont"
                ),
            }
        )
    for (dataset, method), g in b.groupby(["dataset", "readout"]):
        key = f"{'post' if dataset == 'free_chat' else 'mid'}_adapted_dejected5"
        y = (labels.loc[g.author_id, key].to_numpy() - 1) / 5
        rows.append(
            {
                "dataset": dataset,
                "query": "adapted_dejected5",
                "readout": method,
                "full_function_r": quality(g.full_function, g.score)["r"],
                **external_quality(g.score, g.native_score, y),
            }
        )
    pd.DataFrame(rows).to_csv(out / "external_metrics.csv", index=False)
    a.to_csv(out / "private_source_joined.csv", index=False)
    b.to_csv(out / "private_mood_joined.csv", index=False)
    print(
        pd.DataFrame(rows)[
            [
                "dataset",
                "query",
                "readout",
                "full_function_r",
                "function_external_r",
                "function_auc",
            ]
        ]
        .round(4)
        .to_string(index=False),
        flush=True,
    )


def analyze(program_dir, packet, out):
    """Separate full-state error, effective-reader error and behavioral response."""
    p, operators, funcs = load_operators(program_dir)
    m = read_json(packet / "manifest.json")
    if read_json(packet / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Finish independent inference")
    if (
        digest(packet / "predictions.csv")
        != read_json(packet / "prediction_freeze.json")["sha256"]
    ):
        raise ValueError("Predictions changed")
    if m["program_sha256"] != digest(program_dir / "program.json"):
        raise ValueError("State program changed")
    if out.exists():
        raise FileExistsError("Choose a new analysis ID")
    out.mkdir(parents=True)
    frame = pd.read_csv(packet / "predictions.csv", dtype={"author_id": str})
    mode = m["mode"]
    if mode == "external":
        external_links(frame, out)
    else:
        source = (
            OUTPUT
            if mode == "worlds"
            else BASE / "navigation_v1/NAV-04/natural_evidence_r1"
        )
        design_name = "design.json" if mode == "worlds" else "private_design.json"
        world_name = "worlds.json" if mode == "worlds" else "private_worlds.json"
        packet_name = "packets" if mode == "worlds" else "private_packets"
        worlds = pd.DataFrame(read_json(source / world_name)).drop(columns="texts")
        frame = frame.merge(worlds, on="world_id", validate="many_to_one")
        actual, errors = [], []
        for d in read_json(source / design_name):
            path = packet / "private_units" / f"unit_{d['index']:04d}.json"
            meta, states = checked_arrays(path)
            if any(meta["identity"][k] != d[k] for k in ("index", "query", "world_id")):
                raise ValueError("Generated state identity changed")
            tm, teacher = checked_arrays(
                source / packet_name / f"unit_{d['index']:04d}.json"
            )
            if any(tm[k] != d[k] for k in ("index", "query", "world_id")):
                raise ValueError("Teacher state identity changed")
            q = d["query"]
            target = pack_state(teacher["P"], teacher["Q"])
            native = tm["native"]["conditional_score"]
            parts = funcs[q].parts(teacher["P"], teacher["Q"])
            actual.append(
                {k: d[k] for k in ("world_id", "query")}
                | {"native": native, "full_function": parts["score"]}
            )
            for name, estimated in states.items():
                errors.append(
                    {k: d[k] for k in ("world_id", "query")}
                    | {
                        "readout": name,
                        **state_errors(
                            estimated, target, operators[q], p["state_shapes"][q]
                        ),
                    }
                )
        actual = pd.DataFrame(actual)
        joined = frame.merge(actual, on=["world_id", "query"], validate="many_to_one")
        diagnostics = pd.DataFrame(errors).merge(
            worlds[["world_id", "split"]], on="world_id", validate="many_to_one"
        )
        diagnostics.to_csv(out / "state_errors.csv", index=False)
        aggregate = (
            diagnostics.groupby(["query", "readout", "split"])
            .mean(numeric_only=True)
            .reset_index()
        )
        aggregate.to_csv(out / "state_error_summary.csv", index=False)
        metrics = []
        for (q, method, split), g in joined.groupby(["query", "readout", "split"]):
            metrics.append(
                {
                    "query": q,
                    "readout": method,
                    "split": split,
                    **quality(g.full_function, g.score),
                    "native_r": quality(g.native, g.score)["r"],
                }
            )
        pd.DataFrame(metrics).to_csv(out / "level_metrics.csv", index=False)
        joined.to_csv(out / "private_joined.csv", index=False)
        # Responses are paired within the same scene/author; constant controls stay.
        target_rows = (
            actual.rename(columns={"native": "score"})
            .drop(columns="full_function")
            .assign(readout="native")
            .merge(worlds, on="world_id", validate="many_to_one")
        )
        responses = response_rows(
            pd.concat([frame, target_rows], ignore_index=True),
            "identity" if mode == "worlds" else "evidence",
        )
        _, effects = compare_responses(responses)
        responses.to_csv(out / "paired_responses.csv", index=False)
        effects.to_csv(out / "response_metrics.csv", index=False)
        print(
            pd.DataFrame(metrics)
            .query("split != 'construction'")
            .round(4)
            .to_string(index=False),
            flush=True,
        )
    write_json(
        out / "summary.json",
        {
            "status": "COMPLETE_STATE_TRANSPORT_ANALYSIS",
            "mode": mode,
            "prediction_sha256": digest(packet / "predictions.csv"),
            "program_sha256": digest(program_dir / "program.json"),
            "code_sha256": digest(Path(__file__)),
            "human_label_fitting": False,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--program", type=Path, default=DEFAULT)
    p.add_argument(
        "--mode", choices=("worlds", "natural", "external"), default="worlds"
    )
    p.add_argument("--packet", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    packet = a.packet or BASE / f"navigation_v1/NAV-10/state_{a.mode}_r1"
    analyze(a.program, packet, a.out or packet / "analysis")
