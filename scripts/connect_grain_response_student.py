#!/usr/bin/env python3
"""Connect the already frozen S2 student to previously exposed source-only outcomes."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.external_material import unpack_features
from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.fit_grain_response_student import (
    predict_kernel_student,
    transform_features,
)
from scripts.run_grain_real_author_path import BASE, read, sha, write


def paired_source(frame, query):
    """Require unique, identically labeled cue-mask source rows for both controls."""
    task = "TF_cont" if query == "TF_cont" else "big5"
    rows = frame[
        (frame.task == task)
        & (frame.trait == query)
        & (frame.arm == "cue_mask")
        & frame.observer.isin(["native_query", "e5_large"])
    ]
    if rows.duplicated(["author_id", "observer"]).any():
        raise ValueError("Duplicate author/observer in source-only outcomes")
    native = rows[rows.observer == "native_query"].set_index("author_id")
    encoder = rows[rows.observer == "e5_large"].set_index("author_id")
    if set(native.index) != set(encoder.index) or len(native) < 3:
        raise ValueError("Unpaired or empty source controls")
    native, encoder = native.sort_index(), encoder.sort_index()
    if not np.array_equal(native.y_true, encoder.y_true):
        raise ValueError("Source controls disagree about labels")
    return pd.DataFrame(
        {
            "y_true": native.y_true,
            "native_query": native.prediction,
            "e5_large": encoder.prediction,
        }
    )


def main():
    """Freeze provenance, score cached vectors, and report without any refitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Private connections belong in ignored results")
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    summary = read(args.student / "summary.json")
    artifact = args.student / "frozen_student.joblib"
    if summary["labels_read"] or sha(artifact) != summary["student_sha256"]:
        raise ValueError("Require a verified student frozen before outcomes")
    # Only load the locally generated, hash-verified joblib; never an external pickle.
    student = joblib.load(artifact)
    source = BASE / "d1_author_discovery_r1"
    source_summary = read(source / "summary.json")
    predictions = source / "private_predictions.csv"
    if sha(predictions) != source_summary["outputs_sha256"][predictions.name]:
        raise ValueError("D1 outcomes changed")
    panel = BASE / "multiquery_sensitivity_s1_r1"
    members = {row["author_id"] for row in read(panel / "membership.json")}
    if sha(panel / "manifest.json") != summary["panel_manifest_sha256"]:
        raise ValueError("Student training membership changed")
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_S2_FROZEN_SOURCE_CONNECTION_V1",
            "student_sha256": sha(artifact),
            "source_sha256": sha(predictions),
            "script_sha256": sha(__file__),
            "queries": ["TF_cont", "neuroticism"],
            "arm": "cue_mask",
            "selection": "one frozen all-query student; no labels used for selection or calibration",
            "scope": "D1 was exposed discovery data, not a new untouched confirmation sample",
            "bridge_or_meps_or_lockbox_read": False,
        },
    )
    frame = pd.read_csv(predictions, dtype={"author_id": str})
    sources = {q: paired_source(frame, q) for q in ("TF_cont", "neuroticism")}
    wanted = set().union(*(set(part.index) for part in sources.values()))
    if wanted & members:
        raise ValueError("Student training authors overlap external connection")
    paths = {}
    for shard in sorted((BASE / "vm05_full_vectors_r1/e5_large").glob("shard_*")):
        for uid in read(shard / "manifest.json")["authors"]:
            if uid in wanted:
                if uid in paths:
                    raise ValueError("Duplicate encoder cache")
                paths[uid] = shard / "private_cache" / (uid + ".npz")
    if set(paths) != wanted:
        raise ValueError("Missing source encoder caches")
    scored, inputs = {}, {}
    with threadpool_limits(limits=2):
        for j, uid in enumerate(sorted(wanted)):
            path = paths[uid]
            metadata = read(path.with_suffix(".json"))
            digest = sha(path)
            if digest != metadata["feature_sha256"]:
                raise ValueError("Encoder cache changed")
            inputs[str(path.relative_to(ROOT))] = digest
            with np.load(path, allow_pickle=False) as z:
                vectors = unpack_features(dict(z), metadata["feature_aliases"])[
                    "cue_mask_comment_vectors"
                ]
            if vectors.shape != (3, 12, student["feature"]["poles"].shape[-1]):
                raise ValueError("Require the same three views of twelve comments")
            queries = [q for q in sources if uid in sources[q].index]
            scored[uid] = {}
            for query in queries:
                q = np.full(3, student["queries"].index(query), dtype=int)
                x = transform_features(list(vectors), q, student["feature"])
                scored[uid][query] = float(
                    predict_kernel_student(x, q, student["model"]).mean()
                )
            if (j + 1) % 500 == 0:
                print(
                    f"Frozen S2 connection: {j + 1}/{len(wanted)} authors", flush=True
                )
    rows, output = [], []
    for query, part in sources.items():
        part = part.copy()
        part["student"] = [scored[uid][query] for uid in part.index]
        for name in ("native_query", "e5_large", "student"):
            metrics = scalar_metrics(part.y_true, part[name], query == "TF_cont")
            fidelity = scalar_metrics(part.native_query, part[name])
            rows.append(
                {
                    "query": query,
                    "observer": name,
                    **metrics,
                    "teacher_fidelity_r": fidelity["pearson_r"],
                }
            )
        output.append(part.reset_index().assign(query=query))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.concat(output, ignore_index=True).to_csv(
        out / "private_predictions.csv", index=False
    )
    write(out / "private_cache_hashes.json", inputs)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE_EXPLORATORY_CONNECTION",
            "authors": len(wanted),
            "seconds": time.monotonic() - start,
            "new_fits": 0,
            "neural_forwards": 0,
            "source_labels_read": True,
            "bridge_or_meps_or_lockbox_read": False,
            "frozen_spec": summary["frozen_spec"],
            "manifest_sha256": sha(out / "manifest.json"),
            "metrics": metrics.astype(object)
            .where(pd.notna(metrics), None)
            .to_dict("records"),
        },
    )
    report = "# Frozen S2 student: source-only psychological connection\n\n"
    report += "Cue-mask only; identical three-view authors. No refit, sign flip or calibration.\n"
    report += "The statistical E5 control used source-label training; the student used teacher scores only.\n"
    report += (
        "D1 outcomes were previously exposed; this is an exploratory connection.\n\n"
    )
    report += metrics.to_markdown(index=False, floatfmt=".4f") + "\n"
    (out / "report.md").write_text(report, encoding="utf-8")
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
