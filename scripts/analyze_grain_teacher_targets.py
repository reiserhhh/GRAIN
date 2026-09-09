#!/usr/bin/env python3
"""NEXT11: reconcile frozen native, fixed, GRAIN and OOF target information."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.direct_traits import AXES, BIG5
from grain_research.vector_mechanisms.factor_catalog import association
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from grain_research.vector_mechanisms.target_accounting import (
    native_chain,
    projection_account,
    projection_draws,
)

PLAN = ROOT / "docs/GRAIN_LOGIC_TEACHER_TARGET_PLAN_V1.md"
M29 = BASE / "all_factor_source_m29_v1"
READING = BASE / "logic_reading_levels_v1"
S9 = BASE / "source_remainder_s9_r1"
KEYS = ["author_id", "query"]
DRAWS = 1999


def verify_digest(path, expected):
    """Refuse changed frozen prediction bytes before opening outcomes."""
    if sha(path) != expected:
        raise ValueError("Frozen prediction bytes changed")


def join_panel(oof, fixed, criteria, observer):
    """Join exact source membership and preserve all three views per author."""
    oof = oof[oof.arm == "pooled"].copy()
    if oof.empty or oof.duplicated(KEYS + ["view"]).any():
        raise ValueError("Unique pooled author/query/view rows required")
    groups = oof.groupby(KEYS)
    if (
        groups.size().ne(3).any()
        or groups.view.apply(lambda v: set(v) != {0, 1, 2}).any()
        or groups.fold.nunique().ne(1).any()
    ):
        raise ValueError("Complete views and one held-author fold required")
    for frame in (fixed, criteria, observer):
        if frame.duplicated(KEYS).any():
            raise ValueError("Duplicate author/query join")
    columns = ["actual_c", "actual_r", "predicted_c", "predicted_r"]
    values = groups[columns].mean().reset_index()
    folds = groups.fold.first().reset_index()
    values = values.merge(folds, on=KEYS, validate="one_to_one")
    values = values.merge(fixed, on=KEYS, how="left", validate="one_to_one")
    numerical = columns + ["score", "configuration", "mean", "final"]
    if not np.isfinite(values[numerical]).all().all():
        raise ValueError("Missing or nonfinite frozen value")
    if not np.allclose(values.actual_c, values.configuration, rtol=0, atol=1e-12):
        raise ValueError("Configuration replay differs")
    values = values.merge(
        criteria[KEYS + ["y_true", "prediction"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    if (
        not np.isfinite(values[["y_true", "prediction"]]).all().all()
        or not values.y_true.between(0, 1).all()
    ):
        raise ValueError("Missing or invalid source criterion/native score")
    values = values.merge(
        observer[KEYS + ["old_teacher", "new_teacher"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    matched = values["query"].isin(["TF_cont", "neuroticism"])
    if (
        not np.isfinite(values.loc[matched, ["old_teacher", "new_teacher"]]).all().all()
        or values.loc[~matched, "new_teacher"].notna().any()
        or not np.allclose(
            values.loc[matched, "old_teacher"],
            values.loc[matched, "prediction"],
            rtol=0,
            atol=1e-12,
        )
    ):
        raise ValueError("Observer-native replay or coverage differs")
    values["grain_cv"] = values.actual_c - values.actual_r
    return values.sort_values(KEYS).reset_index(drop=True)


def interval(values):
    """Pointwise paired-bootstrap interval, counting degenerate omissions."""
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 100:
        raise ValueError("Too few nondegenerate author resamples")
    lo, hi = np.quantile(finite, [0.025, 0.975])
    return {"lower": float(lo), "upper": float(hi), "valid_draws": len(finite)}


def chain_intervals(chain, native, y, indices):
    """Resample each chain intact without mixing covariance denominators."""
    target = np.asarray(native)[indices]
    outcome = np.asarray(y)[indices]
    target -= target.mean(1, keepdims=True)
    outcome -= outcome.mean(1, keepdims=True)
    denominator = np.sqrt((target**2).mean(1) * (outcome**2).mean(1))
    result = {}
    for i, name in enumerate(chain["names"]):
        part = chain["values"][:, i][indices]
        part -= part.mean(1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            draws = (part * outcome).mean(1) / denominator
        result[name] = interval(draws)
    return result


def run(out):
    """Compute the fixed finite account; never fit a predictor or use held data."""
    paths = [
        M29 / "private_predictions.csv",
        M29 / "private_external_links.csv",
        READING / "private_source_oof.csv",
        S9 / "private_components_before_outcomes.csv",
    ]
    verify_digest(paths[0], read(M29 / "prediction_freeze.json")["sha256"])
    verify_digest(
        paths[2], read(READING / "prediction_freeze.json")["files"][paths[2].name]
    )
    verify_digest(paths[3], read(S9 / "components_frozen.json")["sha256"])
    verify_digest(paths[1], read(READING / "manifest.json")["criterion_sha256"])
    fixed, criteria, oof, observer = (
        pd.read_csv(p, dtype={"author_id": str}) for p in paths
    )
    merged = join_panel(oof, fixed, criteria, observer)
    if (
        len(merged) != 1317
        or merged.author_id.nunique() != 276
        or set(merged["query"]) != {*AXES, *BIG5}
        or any(len(g) != (63 if q in AXES else 213) for q, g in merged.groupby("query"))
        or oof.groupby(KEYS).fold.nunique().ne(1).any()
    ):
        raise ValueError("Current reference-disjoint nine-query panel changed")
    # The four extra older rows are one author, not a label-driven exclusion.
    excluded = criteria.merge(merged[KEYS], on=KEYS, how="left", indicator=True)
    excluded = excluded[excluded._merge == "left_only"]
    if (
        len(excluded) != 4
        or excluded.author_id.nunique() != 1
        or set(excluded["query"]) != set(AXES)
    ):
        raise ValueError("Unexpected source membership difference")
    out.mkdir(parents=True, exist_ok=False)
    merged.to_csv(out / "private_author_accounts.csv", index=False)
    manifest = {
        "hashes": {
            str(p.relative_to(ROOT)): sha(p)
            for p in paths
            + [
                PLAN,
                Path(__file__),
                ROOT / "grain_research/vector_mechanisms/target_accounting.py",
                ROOT / "grain_research/vector_mechanisms/reading_levels.py",
                ROOT / "grain_research/vector_mechanisms/criterion_accounting.py",
            ]
        },
        "authors": 276,
        "author_queries": len(merged),
        "views_per_author": 3,
        "older_excluded_author_queries": 4,
        "older_excluded_authors": 1,
        "new_predictor_fits": 0,
        "new_neural_calls": 0,
        "draws": DRAWS,
        "alignment": "descriptive target/proxy only, recomputed per resample; not a deployed calibration",
        "native_scope": "D1 BF16 all nine; S9 FP32 same observer only for TF/N",
    }
    write(out / "manifest.json", manifest)
    links, projections, chains, summary = [], [], [], []
    max_error = 0.0
    for qi, (query, group) in enumerate(merged.groupby("query", sort=True)):
        y = group.y_true.to_numpy()
        n = len(group)
        indices = np.random.default_rng(20260910 + qi).integers(0, n, (DRAWS, n))
        native, fixed, c, residual, extracted = (
            group[k].to_numpy()
            for k in ("prediction", "score", "actual_c", "actual_r", "predicted_r")
        )
        channels = {
            "native_D1": native,
            "fixed_f": fixed,
            "configuration_c": c,
            "residual_cv": residual,
            "grain_cv": group.grain_cv.to_numpy(),
            "extracted_c": group.predicted_c.to_numpy(),
            "extracted_R": extracted,
        }
        edges = [
            ("native_D1_to_fixed", native, fixed, "mixed_observer"),
            ("fixed_to_configuration", fixed, c, "same_observer"),
            ("configuration_to_extracted", c, channels["extracted_c"], "frozen_oof"),
            ("residual_to_extracted", residual, extracted, "frozen_oof"),
        ]
        if query in ("TF_cont", "neuroticism"):
            channels["native_FP32"] = group.new_teacher.to_numpy()
            edges.extend(
                [
                    (
                        "native_D1_to_FP32",
                        native,
                        channels["native_FP32"],
                        "runtime_conditions",
                    ),
                    (
                        "native_FP32_to_fixed",
                        channels["native_FP32"],
                        fixed,
                        "same_observer",
                    ),
                ]
            )
        minority = int(min(np.sum(y == 0), np.sum(y == 1))) if query in AXES else None
        for channel, values in channels.items():
            draw = projection_draws(values, values, y, indices)["target_r"]
            links.append(
                {
                    "query": query,
                    "channel": channel,
                    "minority_count": minority,
                    **association(values, y, binary=query in AXES),
                    **interval(draw),
                }
            )
        records = {}
        for edge, target, proxy, scope in edges:
            account = projection_account(target, proxy, y)
            draws = projection_draws(target, proxy, y, indices)
            row = {
                "query": query,
                "edge": edge,
                "scope": scope,
                "minority_count": minority,
                **account,
            }
            for metric, values in draws.items():
                row.update({metric + "_" + k: v for k, v in interval(values).items()})
            projections.append(row)
            records[edge] = account
            max_error = max(
                max_error,
                account["closure_error"],
                account["projection_identity_error"],
            )
        chain = native_chain(native, fixed, c, residual, extracted, y)
        all_chains = [("mixed_observer", chain)]
        if "native_FP32" in channels:
            all_chains.append(
                (
                    "observer_separated",
                    native_chain(
                        native,
                        fixed,
                        c,
                        residual,
                        extracted,
                        y,
                        channels["native_FP32"],
                    ),
                )
            )
        for scope, chain in all_chains:
            cis = chain_intervals(chain, native, y, indices)
            max_error = max(max_error, chain["closure_error"])
            for name, value in zip(chain["names"], chain["contributions"], strict=True):
                chains.append(
                    {
                        "query": query,
                        "scope": scope,
                        "component": name,
                        "n": n,
                        "native_r": chain["total_r"],
                        "native_denominator": chain["denominator"],
                        "contribution": value,
                        **cis[name],
                    }
                )
        summary.append(
            {
                "query": query,
                "n": n,
                "minority_count": minority,
                **{
                    k + "_r": float(np.corrcoef(v, y)[0, 1])
                    for k, v in channels.items()
                },
                **{
                    k + "_non_affine": a["non_affine_contribution"]
                    for k, a in records.items()
                },
            }
        )
    if max_error > 1e-12:
        raise ValueError("Numerical account did not close")
    for name, rows in (
        ("channel_links", links),
        ("projection_accounts", projections),
        ("signed_chain_accounts", chains),
        ("query_summary", summary),
    ):
        pd.DataFrame(rows).to_csv(out / (name + ".csv"), index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "max_identity_error": max_error,
            "channels": len(links),
            "edges": len(projections),
            "chain_rows": len(chains),
            "queries": 9,
        },
    )
    print(
        pd.DataFrame(summary)[
            [
                "query",
                "n",
                "native_D1_r",
                "fixed_f_r",
                "configuration_c_r",
                "residual_cv_r",
                "extracted_R_r",
            ]
        ].to_string(index=False)
    )


def main():
    """Accept only a new ignored destination and retain prior evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError("Individual outputs must remain in ignored results")
    if out.exists():
        raise ValueError("Preserve previous evidence; choose a new directory")
    run(out)


if __name__ == "__main__":
    main()
