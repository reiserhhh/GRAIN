#!/usr/bin/env python3
"""NEXT15: localize measured conditional gaps without fitting a new predictor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.conditional_factorial import (
    COMPONENTS,
    ORDERS,
    factorial_parts,
    ordered_parts,
)
from grain_research.vector_mechanisms.projected_components import (
    projected_component_draws,
    projected_components,
)
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from grain_research.vector_mechanisms.scalar_component import ARMS
from scripts.analyze_grain_teacher_targets import interval, verify_digest

KEYS = ["author_id", "query"]
CELL_NAMES = ["f11", "f01", "f10", "f00"]
PREVIOUS = BASE / "logic_conditional_target_gap_v1/r3_replay"
M15 = BASE / "configuration_query_factorial_v1"
PLAN = ROOT / "docs/GRAIN_LOGIC_CONDITIONAL_FACTORIAL_PLAN_V1.md"


def construct(out):
    """Verify old/new endpoints and freeze model-only contrasts before reading Y."""
    source_path = PREVIOUS / "private_model_only_crosswalk.csv"
    prior = read(PREVIOUS / "summary.json")
    for name, digest in prior["input_hashes"].items():
        verify_digest(ROOT / name, digest)
    verify_digest(source_path, read(PREVIOUS / "crosswalk_freeze.json")["sha256"])
    cells_path = M15 / "private_cells_before_outcomes.csv"
    frozen = read(M15 / "cells_frozen.json")
    if frozen["outcomes_used"] is not False:
        raise ValueError("Expected model-only cell freeze")
    verify_digest(cells_path, frozen["sha256"])
    cols = KEYS + [
        "arm",
        "prediction",
        "teacher",
        "anchor",
        "PQ",
        "base_ref",
        "transport_remainder",
    ]
    source = pd.read_csv(source_path, usecols=cols, dtype={"author_id": str})
    cells = pd.read_csv(cells_path, dtype={"author_id": str})
    if (
        "y_true" in cells
        or cells.duplicated(KEYS).any()
        or source.duplicated(KEYS + ["arm"]).any()
    ):
        raise ValueError("Duplicate or outcome-linked construction input")
    parts = factorial_parts(cells[CELL_NAMES])
    if not np.allclose(
        parts,
        cells[["baseline", "E_main", "X_main", "interaction"]],
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("Historical four-cell parts changed")
    cells = cells[KEYS + CELL_NAMES].copy()
    for i, name in enumerate(COMPONENTS):
        cells[name] = parts[:, i]
    data = source.merge(
        cells, on=KEYS, how="outer", validate="many_to_one", indicator=True
    )
    if len(data) != 48 * len(ARMS) or not data._merge.eq("both").all():
        raise ValueError("All48 author-query rows and seven arms required")
    data = data.drop(columns="_merge").sort_values(["query", "author_id", "arm"])
    error = max(
        float(np.max(np.abs(data.teacher - data.f11))),
        float(np.max(np.abs(data.anchor - data.f10))),
    )
    if error > 1e-12:
        raise ValueError("Current native or conditional anchor changed")
    for _, g in data.groupby("query"):
        if g.author_id.nunique() != 24 or set(g.arm) != set(ARMS):
            raise ValueError("Preserve complete query/arm cohort")
    old_account = PREVIOUS / "conditional_gap_accounts.csv"
    public_audit = read(
        ROOT
        / "reports/grain_vector_mechanisms/scalar_conditional_tables_v1/reproduction.json"
    )
    verify_digest(
        old_account, public_audit["public_tables"][old_account.name]["sha256"]
    )
    out.mkdir(parents=True, exist_ok=False)
    data.to_csv(out / "private_model_contrasts.csv", index=False)
    write(
        out / "contrast_freeze.json",
        {
            "sha256": sha(out / "private_model_contrasts.csv"),
            "human_labels_read": False,
        },
    )
    write(
        out / "manifest.json",
        {
            "status": "CONTRASTS_FROZEN",
            "author_queries": 48,
            "proxies": len(ARMS),
            "input_hashes": {
                str(p.relative_to(ROOT)): sha(p)
                for p in (source_path, cells_path, old_account)
            },
            "max_endpoint_replay_error": error,
            "new_neural_calls": 0,
            "new_predictor_fits": 0,
            "plan_sha256": sha(PLAN),
            "script_sha256": sha(__file__),
            "core_sha256": sha(
                ROOT / "grain_research/vector_mechanisms/conditional_factorial.py"
            ),
        },
    )
    print({"status": "CONTRASTS_FROZEN", "rows": len(data), "endpoint_error": error})


def analyze(out):
    """Apply one projector per proxy; retain all signed conditional components."""
    if (out / "summary.json").exists():
        raise ValueError("Preserve completed analysis")
    freeze = read(out / "contrast_freeze.json")
    if freeze["human_labels_read"] is not False:
        raise ValueError("Expected model-only contrast freeze")
    private = out / "private_model_contrasts.csv"
    verify_digest(private, freeze["sha256"])
    manifest = read(out / "manifest.json")
    for name, digest in manifest["input_hashes"].items():
        verify_digest(ROOT / name, digest)
    criterion_path = BASE / "all_factor_source_m29_v1/private_external_links.csv"
    verify_digest(
        criterion_path,
        read(BASE / "logic_reading_levels_v1/manifest.json")["criterion_sha256"],
    )
    source = pd.read_csv(private, dtype={"author_id": str})
    y = pd.read_csv(criterion_path, usecols=KEYS + ["y_true"], dtype={"author_id": str})
    data = source.merge(y, on=KEYS, how="left", validate="many_to_one")
    if not np.isfinite(data.y_true).all():
        raise ValueError("Source criterion incomplete")
    data.to_csv(out / "private_criterion_crosswalk.csv", index=False)
    previous = pd.read_csv(PREVIOUS / "conditional_gap_accounts.csv")
    rows, order_rows, geometry, bridge = [], [], [], []
    max_error, max_orthogonal = 0.0, 0.0
    for qi, (query, group) in enumerate(data.groupby("query", sort=True)):
        ix = np.random.default_rng(20260910 + qi).integers(0, 24, (1999, 24))
        one = group[group.arm == "raw_f"].sort_values("author_id")
        raw_parts = one[list(COMPONENTS)].to_numpy()
        for i, component in enumerate(COMPONENTS):
            v = raw_parts[:, i]
            geometry.append(
                {
                    "query": query,
                    "component": component,
                    "n": 24,
                    "rms": float(np.sqrt(np.mean(v**2))),
                    "sd": float(v.std()),
                    "mean": float(v.mean()),
                }
            )
        for arm, part in group.groupby("arm", sort=True):
            part = part.sort_values("author_id")
            t, p, target = (part[c].to_numpy() for c in ("f11", "prediction", "y_true"))
            a = part[list(COMPONENTS)].to_numpy()
            result = projected_components(t, a, p, target)
            boot = projected_component_draws(t, a, p, target, ix)
            v = np.asarray(result["contributions"])
            max_error = max(
                max_error, result["closure_error"], result["max_reconstruction_error"]
            )
            max_orthogonal = max(max_orthogonal, result["orthogonality_error"])
            base = {
                "query": query,
                "arm": arm,
                "n": len(t),
                "native_criterion_r": result["total_r"],
            }
            for i, component in enumerate(("aligned_total", *COMPONENTS)):
                rows.append(
                    {
                        **base,
                        "component": component,
                        "r_contribution": float(v[i]),
                        **interval(boot[:, i]),
                    }
                )
            old = (
                previous[(previous["query"] == query) & (previous.arm == arm)]
                .set_index("component")
                .r_contribution
            )
            checks = {
                "conditional_anchor": (v[1] + v[2], old.remaining_conditional_anchor),
                "transport": (
                    v[3] + v[4] - old.remaining_fixed_function,
                    old.remaining_transport,
                ),
                "aligned_total": (v[0], old.aligned_total),
            }
            for name, (actual, expected) in checks.items():
                error = abs(actual - expected)
                max_error = max(max_error, error)
                bridge.append(
                    {
                        **base,
                        "identity": name,
                        "current_value": actual,
                        "previous_value": expected,
                        "error": error,
                    }
                )
            for order in ORDERS:
                allocated = ordered_parts(a, order)
                account = projected_components(t, allocated, p, target)
                resampled = projected_component_draws(t, allocated, p, target, ix)
                for i, component in (
                    (2, "evidence_allocation"),
                    (3, "query_allocation"),
                ):
                    order_rows.append(
                        {
                            **base,
                            "order": order,
                            "component": component,
                            "r_contribution": account["contributions"][i],
                            **interval(resampled[:, i]),
                        }
                    )
                max_error = max(
                    max_error,
                    account["closure_error"],
                    account["max_reconstruction_error"],
                )
    if max_error > 1e-12 or max_orthogonal > 1e-12:
        raise ValueError("Common projection or previous-account bridge failed")
    for name, values in (
        ("component_accounts", rows),
        ("order_accounts", order_rows),
        ("component_geometry", geometry),
        ("previous_account_bridge", bridge),
    ):
        pd.DataFrame(values).to_csv(out / (name + ".csv"), index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "author_queries": 48,
            "proxies": len(ARMS),
            "max_identity_error": max_error,
            "max_orthogonality_error": max_orthogonal,
            "new_neural_calls": 0,
            "new_predictor_fits": 0,
            "pointwise_draws": 1999,
            "criterion_sha256": sha(criterion_path),
            "analysis_script_sha256": sha(__file__),
        },
    )
    print(pd.DataFrame(rows).query("arm=='raw_f'").to_string(index=False))


def main():
    """Explicit no-label construction followed by analysis into a private directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("construct", "analyze"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()) or (
        args.phase == "construct" and out.exists()
    ):
        raise ValueError("Use a fresh ignored results path")
    {"construct": construct, "analyze": analyze}[args.phase](out)


if __name__ == "__main__":
    main()
