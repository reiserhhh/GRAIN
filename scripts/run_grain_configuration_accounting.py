#!/usr/bin/env python3
"""S26: source covariance of retained actual-configuration and pairing paths."""

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

from grain_research.vector_mechanisms.configuration_accounting import (
    covariance_accounting,
    retained_paths,
    vector_geometry,
)
from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.analyze_grain_teacher_student_bridge import paired_intervals
from scripts.run_grain_configuration_source import collect, verify
from scripts.run_grain_functional_transfer import verify as verify_s20
from scripts.run_grain_real_author_path import BASE, read, sha, write
from scripts.run_grain_source_remainder import CONNECTION


def main():
    """Freeze all retained vector scores before exposed source outcome accounting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=BASE / "configuration_source_s25_r1"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "configuration_accounting_s26_r1"
    )
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError("Private outputs must remain ignored")
    m, material = verify(source, args.device)
    if read(source / "summary.json")["status"] != "COMPLETE":
        raise ValueError("Complete S25 source collection and connection required")
    frozen = source / "private_scores_before_outcomes.csv"
    if sha(frozen) != read(source / "scores_frozen.json")["scores_sha256"]:
        raise ValueError("Original S25 score snapshot changed")
    configuration_frame, _packets = collect(source, m, material)
    m24 = read(Path(m["source"]) / "manifest.json")
    _m20, _s14, s12, full_material, pair_frame = verify_s20(
        Path(m24["s20"]), args.device
    )
    if (
        material != full_material
        or configuration_frame.author_id.tolist() != pair_frame.author_id.tolist()
    ):
        raise ValueError("Same complete source order required")
    with np.load(source / "private_reference.npz", allow_pickle=False) as z:
        gradient = z["reference"]
    with np.load(Path(m24["s20"]) / "private_reference.npz", allow_pickle=False) as z:
        if not np.array_equal(gradient, z["reference"]):
            raise ValueError("S20 and S25 must use identical covectors")
    configuration, pairing = [], []
    for i, author in enumerate(material):
        name = f"author_{i:03d}"
        if (
            read(s12 / "private_packets" / (name + ".json"))["author_id"]
            != author["author_id"]
        ):
            raise ValueError("S12 pairing identity differs")
        with np.load(
            s12 / "private_packets" / (name + ".npz"), allow_pickle=False
        ) as z:
            pairing.append(z["kv_pairing"])
        with np.load(
            source / "private_packets" / (name + ".npz"), allow_pickle=False
        ) as z:
            configuration.append(z["delta_heads"])
    configuration, pairing = np.asarray(configuration), np.asarray(pairing)
    out.mkdir(parents=True, exist_ok=False)
    files = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/configuration_accounting.py",
        ROOT / "docs/GRAIN_CONFIGURATION_ACCOUNTING_S26_PLAN.md",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_S26_CONFIGURATION_ACCOUNTING_V1",
            "source": str(source),
            "source_manifest_sha256": sha(source / "manifest.json"),
            "source_summary_sha256": sha(source / "summary.json"),
            "source_scores_sha256": sha(frozen),
            "reference_sha256": sha(source / "private_reference.npz"),
            "s12_manifest_sha256": sha(s12 / "manifest.json"),
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "neural_calls": 0,
            "human_outcome_fit": False,
            "max_seconds": 60,
        },
    )
    start = time.monotonic()
    with threadpool_limits(limits=2):
        remaining, views = retained_paths(configuration, pairing, gradient)
    frame = pd.DataFrame(
        {
            "author_id": [a["author_id"] for a in material],
            "reference_author": np.arange(213) < 24,
            **{name: scores.mean(axis=1) for name, scores in views.items()},
        }
    )
    c_error = float(
        np.max(np.abs(frame.configuration - configuration_frame.configuration_score))
    )
    p_error = float(np.max(np.abs(frame.pairing - pair_frame.mean_response)))
    addition = float(np.max(np.abs(remaining + pairing - configuration)))
    if max(c_error, p_error, addition) > 1e-12:
        raise ValueError("Original path or vector sum replay failed")
    score_path = out / "private_scores_before_outcomes.csv"
    frame.to_csv(score_path, index=False)
    np.savez_compressed(
        out / "private_retained_paths.npz",
        remaining=remaining,
        configuration=configuration,
        pairing=pairing,
        reference=gradient,
        **{name + "_views": values for name, values in views.items()},
    )
    write(
        out / "scores_frozen.json",
        {
            "scores_sha256": sha(score_path),
            "arrays_sha256": sha(out / "private_retained_paths.npz"),
            "human_outcome_fit": False,
        },
    )
    if sha(CONNECTION / "private_predictions.csv") != m["source_y_sha256"]:
        raise ValueError("Authorized source outcomes changed")
    labels = pd.read_csv(
        CONNECTION / "private_predictions.csv",
        dtype={"author_id": str},
        usecols=["author_id", "query", "y_true"],
    )
    labels = labels[labels["query"] == "neuroticism"][["author_id", "y_true"]]
    frame = frame.merge(labels, on="author_id", validate="one_to_one", sort=False)
    if (
        len(frame) != 213
        or frame.author_id.tolist() != configuration_frame.author_id.tolist()
        or not np.isfinite(frame.y_true).all()
    ):
        raise ValueError("Complete ordered source label join required")
    metrics, identities, consistency, geometries = [], [], [], []
    for cohort, mask in (
        ("primary189", np.arange(213) >= 24),
        ("discovery24", np.arange(213) < 24),
        ("all213", np.ones(213, dtype=bool)),
    ):
        group = frame.loc[mask]
        account = covariance_accounting(
            group.y_true, group.configuration, group.pairing, group.remaining
        )
        identities.append({"cohort": cohort, **account})
        geometries.append(
            {
                "cohort": cohort,
                **vector_geometry(configuration[mask], pairing[mask], remaining[mask]),
            }
        )
        for method in ("configuration", "pairing", "remaining"):
            metrics.append(
                {
                    "cohort": cohort,
                    "method": method,
                    "n": len(group),
                    **scalar_metrics(group.y_true, group[method]),
                    **paired_intervals(
                        group.y_true, group[method], group.pairing, False
                    ),
                    "covariance_y": account["source_covariances"][method],
                    "mean": float(group[method].mean()),
                    "std": float(group[method].std(ddof=0)),
                }
            )
            for a, b in ((0, 1), (0, 2), (1, 2)):
                consistency.append(
                    {
                        "cohort": cohort,
                        "method": method,
                        "view_a": a,
                        "view_b": b,
                        **scalar_metrics(
                            views[method][mask, a], views[method][mask, b]
                        ),
                    }
                )
    frame.to_csv(out / "private_source_connections.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(consistency).to_csv(out / "view_consistency.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "source_authors": 213,
        "neural_calls": 0,
        "human_outcome_fit": False,
        "metrics": metrics,
        "accounting": identities,
        "vector_geometry": geometries,
        "view_consistency": consistency,
        "replay": {
            "configuration": c_error,
            "pairing": p_error,
            "vector_addition": addition,
        },
        "seconds": time.monotonic() - start,
        "scores_sha256": sha(score_path),
    }
    summary["resource_budget_exceeded"] = summary["seconds"] > 60
    write(out / "summary.json", summary)
    print(
        pd.DataFrame(metrics)[
            [
                "cohort",
                "method",
                "pearson_r",
                "pearson_p",
                "student_lower",
                "student_upper",
                "covariance_y",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print({"accounting": identities, "seconds": summary["seconds"]}, flush=True)


if __name__ == "__main__":
    main()
