#!/usr/bin/env python3
"""Separate real-source recentering from newly covered directions, without new fitting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.affine_program import compile_affine, score_affine
from grain_research.vector_mechanisms.navigation import digest, read_json, write_json
from grain_research.vector_mechanisms.state_geometry import (
    linear_reader,
    residual_basis,
)
from grain_research.vector_mechanisms.state_transport import pack_state
from scripts.run_grain_navigation_baseline import checked_arrays
from scripts.run_grain_source_span_extension import (
    BASE,
    DEFAULT,
    NATURAL,
    OLD,
    load_extension,
)
from scripts.run_grain_state_transport import load_operators


def analyze():
    """Use teacher only for three nested/shifted affine-space oracle diagnostics."""
    out = DEFAULT / "center_span_analysis"
    if out.exists():
        raise FileExistsError("Keep the original diagnostic")
    models = load_extension(DEFAULT)
    _p, old, functions = load_operators(OLD)
    bases = {q: (residual_basis(old[q]), residual_basis(models[q])) for q in models}
    affine = {
        q: (
            compile_affine(old[q], functions[q]),
            compile_affine(models[q], functions[q]),
        )
        for q in models
    }
    rows = []
    with threadpool_limits(limits=2):
        for d in read_json(NATURAL / "private_design.json"):
            q, i = d["query"], d["index"]
            tm, tz = checked_arrays(NATURAL / "private_packets" / f"unit_{i:04d}.json")
            om, oz = checked_arrays(
                BASE
                / f"navigation_v1/NAV-10/state_natural_r1/private_units/unit_{i:04d}.json"
            )
            if any(
                tm[k] != d[k] or om["identity"][k] != d[k]
                for k in ("index", "world_id", "query")
            ):
                raise ValueError("Diagnostic identity changed")
            actual = pack_state(tz["P"], tz["Q"])
            x = oz["residual_mean"] - old[q]["mean_r"]
            r = actual - x
            vo, vn = bases[q]
            missing = []
            for center, v in (
                (old[q]["mean_r"], vo),
                (models[q]["mean_r"], vo),
                (models[q]["mean_r"], vn),
            ):
                z = r - center
                missing.append(z - (z @ v.T) @ v)
            energies = [float(e @ e) for e in missing]
            g = linear_reader(functions[q])[0].sum(1)
            ao, an = affine[q]
            x0 = old[q]["mean_x"]
            shift = score_affine(an, x0)["score"] - score_affine(ao, x0)["score"]
            slope = float((x - x0) @ (an["coefficients"] - ao["coefficients"]).sum(1))
            total = score_affine(an, x)["score"] - score_affine(ao, x)["score"]
            rows.append(
                {
                    "query": q,
                    "world_id": d["world_id"],
                    "old_outside": energies[0],
                    "recentered_outside": energies[1],
                    "expanded_outside": energies[2],
                    "recenter_gain": energies[0] - energies[1],
                    "span_gain": energies[1] - energies[2],
                    "old_outside_action": float(missing[0] @ g),
                    "recentered_outside_action": float(missing[1] @ g),
                    "expanded_outside_action": float(missing[2] @ g),
                    "function_change": total,
                    "fixed_origin_shift": shift,
                    "slope_action": slope,
                    "affine_accounting_error": abs(total - shift - slope),
                }
            )
    frame = pd.DataFrame(rows)
    if (frame.span_gain < -1e-5).any() or frame.affine_accounting_error.max() > 1e-10:
        raise ValueError("Nested-space or affine accounting failed")
    summaries = []
    for q, g in frame.groupby("query"):
        total_gain = g.old_outside.sum() - g.expanded_outside.sum()
        summaries.append(
            {
                "query": q,
                "units": len(g),
                "outside_energy_reduction": 1
                - g.expanded_outside.sum() / g.old_outside.sum(),
                "fraction_gain_due_to_recenter": g.recenter_gain.sum() / total_gain,
                "fraction_gain_due_to_added_span": g.span_gain.sum() / total_gain,
                "fixed_origin_score_shift": float(g.fixed_origin_shift.iloc[0]),
                "slope_action_mean": float(g.slope_action.mean()),
                "slope_action_std": float(g.slope_action.std()),
                **{
                    s + "_rmse": float(np.sqrt(np.mean(g[s] ** 2)))
                    for s in (
                        "old_outside_action",
                        "recentered_outside_action",
                        "expanded_outside_action",
                    )
                },
            }
        )
    out.mkdir()
    frame.to_csv(out / "private_decomposition.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    write_json(
        out / "manifest.json",
        {
            "status": "COMPLETE",
            "oracle_diagnostic_only": True,
            "new_model_calls": 0,
            "script_sha256": digest(Path(__file__)),
            "construction_sha256": digest(DEFAULT / "construction_freeze.json"),
            "max_affine_accounting_error": float(frame.affine_accounting_error.max()),
            "anchoring": "old synthetic construction mean early state, not chosen from test",
        },
    )
    print(pd.DataFrame(summaries).round(5).to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        analyze()
