#!/usr/bin/env python3
"""S23: source connection of a fixed, score-preserving functional decomposition."""

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

from grain_research.vector_mechanisms.cross_query_functional import QUERIES
from grain_research.vector_mechanisms.functional_metric import functional_metric
from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.analyze_grain_teacher_student_bridge import paired_intervals
from scripts.run_grain_cross_query_functional import (
    freeze_readers,
    reference_arrays,
)
from scripts.run_grain_cross_query_functional import (
    verify as verify_s21,
)
from scripts.run_grain_functional_transfer import verify as verify_s20
from scripts.run_grain_real_author_path import BASE, read, sha, write
from scripts.run_grain_source_remainder import CONNECTION, covariance


def main():
    """Freeze complete scores before joining the already exposed N questionnaire."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=BASE / "cross_query_functional_s21_r1"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "functional_source_split_s23_r1"
    )
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError("Private results must remain ignored")
    _m, s20, _s19, s12, material = verify_s21(source, args.device)
    if read(source / "summary.json")["status"] != "COMPLETE":
        raise ValueError("Complete S21 source required")
    _m, _s14, inherited_s12, full_material, old_scores = verify_s20(s20, args.device)
    if s12 != inherited_s12 or material != full_material:
        raise ValueError("Same complete source population required")
    references = freeze_readers(source, material)
    _packets, _g, d_reference = reference_arrays(source, material)
    with np.load(s20 / "private_reference.npz", allow_pickle=False) as z:
        original = z["reference"]
    control = references[QUERIES.index("quantitative_content")]
    out.mkdir(parents=True, exist_ok=False)
    files = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/functional_metric.py",
        ROOT / "docs/GRAIN_FUNCTIONAL_SOURCE_SPLIT_S23_PLAN.md",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_S23_FUNCTIONAL_SOURCE_SPLIT_V1",
            "source": str(source),
            "source_manifest_sha256": sha(source / "manifest.json"),
            "source_summary_sha256": sha(source / "summary.json"),
            "s20_manifest_sha256": sha(s20 / "manifest.json"),
            "s20_scores_sha256": sha(s20 / "private_scores_before_outcomes.csv"),
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "primary_authors": 189,
            "gradient_reference_authors": 24,
            "metric_reference_authors": 12,
            "anchor": "quantitative_content",
            "neural_calls": 0,
            "human_fit": False,
            "max_seconds": 60,
        },
    )
    start = time.monotonic()
    with threadpool_limits(limits=2):
        split = functional_metric(np.stack([original, control]), d_reference, 1)
    rows = []
    for i, author in enumerate(material):
        with np.load(
            s12 / "private_packets" / f"author_{i:03d}.npz", allow_pickle=False
        ) as z:
            d = z["kv_pairing"]
        total = float(np.mean(d @ original))
        aligned = float(np.mean(d @ split["shared"][0]))
        orthogonal = float(np.mean(d @ split["specific"][0]))
        rows.append(
            {
                "author_id": author["author_id"],
                "reference_author": i < 24,
                "total": total,
                "content_aligned": aligned,
                "content_orthogonal": orthogonal,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.author_id.tolist() != old_scores.author_id.tolist():
        raise ValueError("Complete ordered source population required")
    replay = float(np.max(np.abs(frame.total - old_scores.mean_response)))
    addition = float(
        np.max(np.abs(frame.total - frame.content_aligned - frame.content_orthogonal))
    )
    if replay > 1e-12 or addition > 1e-12:
        raise ValueError("S20 total replay or retained-channel identity failed")
    frozen = out / "private_scores_before_outcomes.csv"
    frame.to_csv(frozen, index=False)
    np.savez_compressed(
        out / "private_components.npz", original=original, control=control, **split
    )
    write(
        out / "scores_frozen.json",
        {
            "sha256": sha(frozen),
            "components_sha256": sha(out / "private_components.npz"),
            "human_labels_used": False,
            "total_replay_error": replay,
            "addition_error": addition,
        },
    )
    labels = pd.read_csv(
        CONNECTION / "private_predictions.csv",
        dtype={"author_id": str},
        usecols=["author_id", "query", "y_true"],
    )
    labels = labels[labels["query"] == "neuroticism"][["author_id", "y_true"]]
    frame = frame.merge(labels, on="author_id", validate="one_to_one")
    if len(frame) != 213 or not np.isfinite(frame.y_true).all():
        raise ValueError("Complete authorized source outcomes required")
    metrics, identities = [], []
    for name, group in (
        ("primary189", frame[~frame.reference_author]),
        ("discovery24", frame[frame.reference_author]),
        ("all213", frame),
    ):
        covariances = {}
        for component in ("total", "content_aligned", "content_orthogonal"):
            covariances[component] = covariance(group.y_true, group[component])
            metrics.append(
                {
                    "cohort": name,
                    "component": component,
                    "n": len(group),
                    **scalar_metrics(group.y_true, group[component]),
                    **paired_intervals(
                        group.y_true, group[component], group.total, False
                    ),
                    "source_covariance": covariances[component],
                }
            )
        covariance_error = (
            covariances["total"]
            - covariances["content_aligned"]
            - covariances["content_orthogonal"]
        )
        if abs(covariance_error) > 1e-12:
            raise ValueError("Source covariance accounting failed")
        identities.append(
            {
                "cohort": name,
                "covariances": covariances,
                "addition_error": float(covariance_error),
                "component_covariance": covariance(
                    group.content_aligned, group.content_orthogonal
                ),
            }
        )
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    frame.to_csv(out / "private_source_connections.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "neural_calls": 0,
        "human_fit": False,
        "seconds": time.monotonic() - start,
        "source_authors": 213,
        "anchor_coefficient": float(split["coefficients"][0]),
        "total_replay_error": replay,
        "score_addition_error": addition,
        "reference_orthogonal_anchor_covariance": float(
            split["reference_anchor_covariance"][0]
        ),
        "metrics": metrics,
        "covariance_accounting": identities,
        "scores_sha256": sha(frozen),
    }
    summary["resource_budget_exceeded"] = summary["seconds"] > 60
    write(out / "summary.json", summary)
    print(
        pd.DataFrame(metrics)[
            [
                "cohort",
                "component",
                "pearson_r",
                "pearson_p",
                "student_lower",
                "student_upper",
                "delta",
                "delta_lower",
                "delta_upper",
                "source_covariance",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(
        {
            k: summary[k]
            for k in (
                "anchor_coefficient",
                "total_replay_error",
                "score_addition_error",
                "seconds",
            )
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
