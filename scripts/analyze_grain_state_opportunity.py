#!/usr/bin/env python3
"""NAV-07: paired material/occasion associations with fixed transported readers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.factor_catalog import association
from grain_research.vector_mechanisms.navigation import (
    BASE,
    digest,
    read_json,
    write_json,
)
from scripts.analyze_grain_counseling_views import paired_delta
from scripts.analyze_grain_mood_timing import partial_correlation


def cross_occasion_links(predictions, criteria):
    """Compare both text views on each identical criterion and participant set."""
    rows, paired, timing, changes = [], [], [], []
    for (reference, query), group in predictions.groupby(["reference", "query"]):
        for readout in ("score", "native_score"):
            views = group.pivot(
                index="participant_code", columns="view", values=readout
            ).sort_index()
            if (
                set(views.index) != set(criteria.index)
                or len(views) != 46
                or views.isna().any().any()
            ):
                raise ValueError("Complete same-participant views required")
            y = criteria.loc[views.index]
            free, direct = views.free_chat, views.direct_answers
            for criterion in y.columns:
                target = y[criterion]
                for view in views:
                    rows.append(
                        {
                            "reference": reference,
                            "query": query,
                            "readout": readout,
                            "view": view,
                            "criterion": criterion,
                            **association(views[view], target),
                        }
                    )
                paired.append(
                    {
                        "reference": reference,
                        "query": query,
                        "readout": readout,
                        "criterion": criterion,
                        "free_r": association(free, target)["pearson_r"],
                        "direct_r": association(direct, target)["pearson_r"],
                        **paired_delta(free, direct, target),
                    }
                )
            for view in views:
                before, after = (
                    ("mid", "post") if view == "free_chat" else ("pre", "mid")
                )
                x, a, b = views[view], y[f"{before}_{query}"], y[f"{after}_{query}"]
                delta = paired_delta(b, a, x)
                timing.append(
                    {
                        "reference": reference,
                        "query": query,
                        "readout": readout,
                        "view": view,
                        "before": before,
                        "after": after,
                        "before_r": association(x, a)["pearson_r"],
                        "after_r": association(x, b)["pearson_r"],
                        "after_minus_before_r": delta["delta_r_free_minus_direct"],
                        "delta_lower": delta["delta_lower"],
                        "delta_upper": delta["delta_upper"],
                        **partial_correlation(x, b, a),
                        "score_vs_criterion_change_r": association(x, b - a)[
                            "pearson_r"
                        ],
                    }
                )
            changes.append(
                {
                    "reference": reference,
                    "query": query,
                    "readout": readout,
                    "contrast": "free_minus_direct vs post_minus_mid",
                    **association(
                        free - direct, y[f"post_{query}"] - y[f"mid_{query}"]
                    ),
                    "interpretation": "material and occasion jointly change; not identified personal dynamics",
                }
            )
    return [pd.DataFrame(v) for v in (rows, paired, timing, changes)]


def run(source, out):
    """Read authorized local questionnaire links without predictor or path fitting."""
    if out.exists():
        raise FileExistsError("Choose a new state analysis ID")
    prediction = source / "private_counseling_predictions.csv"
    criteria_path = BASE / "counseling_mood_m30c_v1/private_mood_scores.csv"
    if (
        digest(criteria_path)
        != read_json(criteria_path.parent / "manifest.json")["criterion_cache_sha256"]
    ):
        raise ValueError("Criterion bytes changed")
    freeze = read_json(source / "prediction_freeze.json")
    if digest(prediction) not in freeze.values():
        raise ValueError("Prediction freeze does not cover mood predictions")
    frame = pd.read_csv(prediction)
    criteria = pd.read_csv(criteria_path).pivot(
        index="participant_code", columns="scale", values="score"
    )
    result = cross_occasion_links(frame, criteria)
    out.mkdir(parents=True)
    names = [
        "cross_occasion_links.csv",
        "paired_material_comparison.csv",
        "occasion_specificity.csv",
        "joint_view_time_change.csv",
    ]
    for name, values in zip(names, result, strict=True):
        values.to_csv(out / name, index=False)
    write_json(
        out / "summary.json",
        {
            "status": "COMPLETE_REAL_OBSERVATIONAL_PACKET",
            "participants": 46,
            "prediction_sha256": digest(prediction),
            "criterion_sha256": digest(criteria_path),
            "code_sha256": {
                p: digest(ROOT / p)
                for p in (
                    "scripts/analyze_grain_state_opportunity.py",
                    "scripts/analyze_grain_counseling_views.py",
                    "scripts/analyze_grain_mood_timing.py",
                )
            },
            "query": "adapted_dejected5",
            "views": 2,
            "occasion_criteria": list(criteria.columns),
            "paired_material_comparisons": len(result[1]),
            "bootstrap_draws": 2000,
            "new_predictor_fit": False,
            "synthetic_time_order_packet_complete": False,
            "scope": "same criterion resolves endpoint confounding, not material assignment causality",
            "native_duplication": "native_score identical across references; not independent replication",
        },
    )
    print(
        result[1]
        .query("reference=='R_synth' and readout=='score'")[
            [
                "criterion",
                "free_r",
                "direct_r",
                "delta_r_free_minus_direct",
                "delta_lower",
                "delta_upper",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=BASE / "navigation_v1/NAV-10/reference_transfer_r1",
    )
    parser.add_argument(
        "--out", type=Path, default=BASE / "navigation_v1/NAV-07/state_opportunity_r1"
    )
    a = parser.parse_args()
    run(a.source, a.out)
