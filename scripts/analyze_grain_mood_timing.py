#!/usr/bin/env python3
"""M30C: quantify occasion-specific links without changing any native predictor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.factor_catalog import association, bh_adjust
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from scripts.analyze_grain_counseling_views import paired_delta


def partial_correlation(x, y, z):
    """One-covariate descriptive partial r with its correct residual degrees of freedom."""
    values = np.column_stack([x, y, z]).astype(float)
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < 5 or (np.ptp(values, axis=0) < 1e-14).any():
        raise ValueError("Nonconstant complete vectors required")
    r = np.corrcoef(values.T)
    den = np.sqrt((1 - r[0, 2] ** 2) * (1 - r[1, 2] ** 2))
    if den < 1e-12:
        raise ValueError("Degenerate conditioning variable")
    partial = float(np.clip((r[0, 1] - r[0, 2] * r[1, 2]) / den, -1, 1))
    df = len(values) - 3
    statistic = partial * np.sqrt(df / max(1 - partial**2, 1e-15))
    return {
        "partial_r_after_given_before": partial,
        "partial_p": float(2 * student_t.sf(abs(statistic), df)),
        "partial_df": df,
    }


def run(folder, out):
    """Compare the same fixed text readout with preceding and following state scores."""
    if (
        read(folder / "external_summary.json")["status"]
        != "COMPLETE_M30C_EXPLORATORY_STATE_LINK"
    ):
        raise ValueError("Complete native state associations first")
    if (
        sha(folder / "private_predictions.csv")
        != read(folder / "prediction_freeze.json")["sha256"]
    ):
        raise ValueError("Prediction bytes changed")
    frame = pd.read_csv(folder / "private_predictions.csv")
    ys = pd.read_csv(folder / "private_mood_scores.csv").pivot(
        index="participant_code", columns="scale", values="score"
    )
    rows = []
    for (view, q), g in frame.groupby(["view", "query"]):
        g = g.set_index("participant_code").sort_index()
        start, end = ("mid", "post") if view == "free_chat" else ("pre", "mid")
        y = ys.reindex(g.index)
        before, after, x = y[f"{start}_{q}"], y[f"{end}_{q}"], g.native_score
        delta = paired_delta(after, before, x)
        rows.append(
            {
                "view": view,
                "query": q,
                "before": start,
                "after": end,
                "before_r": association(x, before)["pearson_r"],
                "after_r": association(x, after)["pearson_r"],
                "after_minus_before_r": delta["delta_r_free_minus_direct"],
                "paired_delta_lower": delta["delta_lower"],
                "paired_delta_upper": delta["delta_upper"],
                **partial_correlation(x, after, before),
            }
        )
    result = pd.DataFrame(rows)
    result["partial_q_bh_view"] = result.groupby("view").partial_p.transform(bh_adjust)
    out.mkdir(parents=True, exist_ok=False)
    result.to_csv(out / "occasion_specificity.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE_OBSERVATIONAL_TIMING_DESCRIPTION",
            "new_predictor_fit": False,
            "causal_effect_claim": False,
            "participants": 46,
            "comparisons": len(result),
            "prediction_sha256": sha(folder / "private_predictions.csv"),
        },
    )
    print(result.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    a = parser.parse_args()
    if not a.output_dir.resolve().is_relative_to(BASE.resolve()):
        parser.error("Use local ignored mechanism output")
    run(a.source_dir.resolve(), a.output_dir.resolve())
