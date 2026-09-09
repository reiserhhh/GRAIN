#!/usr/bin/env python3
"""NEXT14: connect measured M11 anchors to current frozen target-map gaps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.functional_profiles import FunctionalQuery
from grain_research.vector_mechanisms.projected_components import (
    constant_fiber_loss,
    projected_component_draws,
    projected_components,
)
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from grain_research.vector_mechanisms.scalar_component import ARMS
from scripts.analyze_grain_teacher_targets import interval, verify_digest

KEYS = ["author_id", "query"]
NAMES = (
    "aligned_total",
    "remaining_conditional_anchor",
    "remaining_fixed_function",
    "remaining_transport",
)


def anchor_fibers(anchors, out):
    """Replay the actual FP32 anchor input and expose its finite constant risk."""
    old_path = BASE / "complete_level_response_v1/reference.npz"
    new_path = BASE / "all_factor_function_analysis_m28_v2/nine_query_functions.npz"
    verify_digest(
        old_path,
        read(BASE / "source_anchor_diagnosis_v1/manifest.json")["reference_sha256"],
    )
    verify_digest(
        new_path,
        read(BASE / "all_factor_function_analysis_m28_v2/summary.json")[
            "functions_sha256"
        ],
    )
    rows = []
    with (
        np.load(old_path, allow_pickle=False) as old,
        np.load(new_path, allow_pickle=False) as new,
    ):
        for q, group in anchors.groupby("query", sort=True):
            for k in ("Z", "h", "b", "G", "g"):
                if not np.array_equal(old[q + "__" + k], new[q + "__" + k]):
                    raise ValueError(
                        "Old intervention and current function reference differ"
                    )
            function = FunctionalQuery(
                *(new[q + "__" + k] for k in ("Z", "h", "G", "g", "b"))
            )
            # The old collector casts common reference states to the FP32 runtime.
            p0, q0 = (
                old[q + "__" + k].astype(np.float32).astype(float) for k in ("Z", "h")
            )
            values = np.array(
                [
                    [
                        function.parts(p0, q0)[k]
                        for k in ("score", "mean", "configuration", "final")
                    ]
                    for _ in range(len(group))
                ]
            )
            if np.ptp(values, axis=0).max() != 0:
                raise ValueError("The declared input fiber is not constant")
            loss = constant_fiber_loss(group.anchor.to_numpy(), values[0, 0])
            if loss["anchor_range"] < 1e-6:
                raise ValueError("No nonconstant native anchor witness")
            rows.append(
                {
                    "query": q,
                    "scope": "measured_anchored_intervention_author_means",
                    "fixed_input_spread": float(np.ptp(values, axis=0).max()),
                    "fp32_reference_cast_score_difference": float(
                        abs(values[0, 0] - function.anchor)
                    ),
                    **loss,
                }
            )
    pd.DataFrame(rows).to_csv(out / "conditional_input_fibers.csv", index=False)
    return {str(p.relative_to(ROOT)): sha(p) for p in (old_path, new_path)}


def run(predictions_dir, out):
    """Keep all existing arms and both queries; fit no model or new endpoint."""
    freeze = read(predictions_dir / "prediction_freeze.json")
    prediction_file = predictions_dir / "private_author_predictions.csv"
    verify_digest(prediction_file, freeze["files"][prediction_file.name])
    anchor_dir = BASE / "source_anchor_diagnosis_v1"
    anchor_path = anchor_dir / "private_anchors_before_outcomes.csv"
    verify_digest(anchor_path, read(anchor_dir / "anchors_frozen.json")["sha256"])
    anchors = pd.read_csv(anchor_path, dtype={"author_id": str})
    old_path = anchor_dir / "private_anchor_source_connection.csv"
    old = pd.read_csv(
        old_path,
        usecols=KEYS + ["anchor", "native", "PQ", "base_ref", "transport_remainder"],
        dtype={"author_id": str},
    )
    if anchors.duplicated(KEYS).any() or old.duplicated(KEYS).any():
        raise ValueError("Duplicate measured anchors")
    old = old.merge(
        anchors, on=KEYS, how="left", validate="one_to_one", suffixes=("", "_verified")
    )
    if len(old) != 48 or not np.allclose(
        old.anchor, old.anchor_verified, rtol=0, atol=1e-12
    ):
        raise ValueError("Measured anchor replay changed")
    current = pd.read_csv(prediction_file, dtype={"author_id": str})
    current = current.merge(old, on=KEYS, how="inner", validate="many_to_one")
    if (
        len(current) != 48 * len(ARMS)
        or not np.allclose(current.teacher, current.native, rtol=0, atol=1e-12)
        or not np.allclose(
            current[current.arm == "raw_f"].prediction,
            current[current.arm == "raw_f"].PQ,
            rtol=0,
            atol=1e-12,
        )
    ):
        raise ValueError("Old/current cohort or native/function replay failed")
    if not np.allclose(
        current.native,
        current.anchor + current.PQ - current.base_ref + current.transport_remainder,
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("Measured conditional terms do not reconstruct native")
    out.mkdir(parents=True, exist_ok=False)
    current.to_csv(out / "private_model_only_crosswalk.csv", index=False)
    write(
        out / "crosswalk_freeze.json",
        {
            "sha256": sha(out / "private_model_only_crosswalk.csv"),
            "human_labels_read": False,
        },
    )
    fiber_hashes = anchor_fibers(anchors, out)
    criterion_path = BASE / "all_factor_source_m29_v1/private_external_links.csv"
    verify_digest(
        criterion_path,
        read(BASE / "logic_reading_levels_v1/manifest.json")["criterion_sha256"],
    )
    criterion = pd.read_csv(
        criterion_path, usecols=KEYS + ["y_true"], dtype={"author_id": str}
    )
    data = current.merge(criterion, on=KEYS, how="left", validate="many_to_one")
    if not np.isfinite(data.y_true).all():
        raise ValueError("Missing source outcome")
    data.to_csv(out / "private_criterion_crosswalk.csv", index=False)
    results, max_error = [], 0.0
    for qi, (q, group) in enumerate(data.groupby("query", sort=True)):
        if group.author_id.nunique() != 24 or set(group.arm) != set(ARMS):
            raise ValueError("The old complete24-author/query panel must be retained")
        ix = np.random.default_rng(20260910 + qi).integers(0, 24, (1999, 24))
        for arm, part in group.groupby("arm", sort=True):
            part = part.sort_values("author_id")
            t, p, y = (
                part.native.to_numpy(),
                part.prediction.to_numpy(),
                part.y_true.to_numpy(),
            )
            a = np.column_stack(
                [part.anchor, part.PQ - part.base_ref, part.transport_remainder]
            )
            account = projected_components(t, a, p, y)
            boot = projected_component_draws(t, a, p, y, ix)
            max_error = max(
                max_error, account["closure_error"], account["max_reconstruction_error"]
            )
            for i, name in enumerate(NAMES):
                results.append(
                    {
                        "query": q,
                        "arm": arm,
                        "component": name,
                        "n": len(t),
                        "native_criterion_r": account["total_r"],
                        "r_contribution": account["contributions"][i],
                        **interval(boot[:, i]),
                    }
                )
    if max_error > 1e-12:
        raise ValueError("Projected accounting failed")
    pd.DataFrame(results).to_csv(out / "conditional_gap_accounts.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "authors": 48,
            "arms": len(ARMS),
            "max_identity_error": max_error,
            "new_fits": 0,
            "new_neural_calls": 0,
            "source_prediction_freeze_sha256": sha(
                predictions_dir / "prediction_freeze.json"
            ),
            "script_sha256": sha(__file__),
            "core_sha256": sha(
                ROOT / "grain_research/vector_mechanisms/projected_components.py"
            ),
            "plan_sha256": sha(ROOT / "docs/GRAIN_LOGIC_SCALAR_COMPONENT_PLAN_V1.md"),
            "input_hashes": {
                str(v.relative_to(ROOT)): sha(v)
                for v in (prediction_file, anchor_path, old_path, criterion_path)
            },
            "fiber_reference_hashes": fiber_hashes,
        },
    )
    print(pd.DataFrame(results).query("arm=='raw_f'").to_string(index=False))


def main():
    """Require frozen inputs and a fresh private destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.predictions_dir.resolve(), args.output_dir.resolve()
    if (
        not source.is_relative_to(BASE.resolve())
        or not out.is_relative_to(BASE.resolve())
        or out.exists()
    ):
        raise ValueError("Private source and new ignored output required")
    run(source, out)


if __name__ == "__main__":
    main()
