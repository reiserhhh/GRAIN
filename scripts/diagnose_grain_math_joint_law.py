#!/usr/bin/env python3
"""Result-after joint-law source uncertainty and reference diagnostics; no calls."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from run_grain_math_response import ROOT, sha, write_csv, write_json

DEFAULT = ROOT / "results/grain_math_response_v1/joint_law_r1"
DEST = ROOT / "results/grain_math_response_v1/joint_law_diagnostic_r1"


def matched_arrays(rows, role):
    """Create complete family-by-design arrays, rejecting missing joint sources."""
    selected = [r for r in rows if r["role"] == role]
    families = sorted({int(r["family"]) for r in selected})
    keys = sorted(
        {(int(r["frame"]), r["actor"], int(r["incidence"])) for r in selected}
    )
    lookup = {
        (int(r["family"]), (int(r["frame"]), r["actor"], int(r["incidence"]))): float(
            r["interaction"]
        )
        for r in selected
    }
    if len(lookup) != len(selected):
        raise ValueError("duplicated source observations")
    if any((f, k) not in lookup for f in families for k in keys):
        raise ValueError("missing joint source cell; no bootstrap imputation")
    return np.array([[lookup[f, k] for k in keys] for f in families]), families, keys


def independent_family_bootstrap(source, application, keys, draws=2000):
    """Resample source and application families independently with all arms paired."""
    rng_source, rng_application = (
        np.random.default_rng(10101),
        np.random.default_rng(10102),
    )
    si = rng_source.integers(0, len(source), (draws, len(source)))
    ai = rng_application.integers(0, len(application), (draws, len(application)))
    source_mean = source[si].mean(axis=1)
    actual = application[ai]
    shifted = [
        keys.index(((f + 1) % 4, actor, 1 - incidence)) for f, actor, incidence in keys
    ]
    corrections = {
        "source_conditional": source_mean,
        "source_global": source_mean.mean(axis=1)[:, None],
        "source_shifted_key_control": source_mean[:, shifted],
    }
    results = []
    for arm, correction in corrections.items():
        errors = (actual - correction[:, None, :]) ** 2
        improvement = (actual**2 - errors).mean(axis=(1, 2))
        conditional = source.mean(0)
        fixed = (
            conditional
            if arm == "source_conditional"
            else np.array([conditional.mean()])
            if arm == "source_global"
            else conditional[shifted]
        )
        point_delta = float(np.mean(application**2 - (application - fixed) ** 2))
        results.append(
            {
                "arm": arm,
                "mse_improvement": point_delta,
                "source_and_application_family_ci": np.quantile(
                    improvement, [0.025, 0.975]
                ).tolist(),
                "source_and_application_rmse_ci": np.quantile(
                    np.sqrt(errors.mean((1, 2))), [0.025, 0.975]
                ).tolist(),
                "draws": draws,
                "source_families": len(source),
                "application_families": len(application),
            }
        )
    return results


def main():
    """Diagnose archived actual contrasts without altering any primary result."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet", type=Path, default=DEFAULT)
    p.add_argument("--output", type=Path, default=DEST)
    args = p.parse_args()
    if (args.output / "summary.json").exists():
        raise ValueError("result-after diagnostic frozen")
    manifest = json.loads((args.packet / "04/artifact_hashes.json").read_text())
    for name, digest in manifest.items():
        if sha(args.packet / name) != digest:
            raise ValueError("changed primary artifact: " + name)
    with (args.packet / "03/four_corner_tables.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["reader"], row["metric"]].append(row)
    uncertainties, source_parts = [], []
    for (reader, metric), rr in grouped.items():
        src, _, keys = matched_arrays(rr, "source")
        app, _, other_keys = matched_arrays(rr, "application")
        if keys != other_keys:
            raise ValueError("source/application design units differ")
        for r in independent_family_bootstrap(src, app, keys):
            uncertainties.append({"reader": reader, "metric": metric, **r})
        grand = float(src.mean())
        family = src.mean(1, keepdims=True) - grand
        design = src.mean(0, keepdims=True) - grand
        remaining = src - grand - family - design
        source_parts.append(
            {
                "reader": reader,
                "metric": metric,
                "grand_interaction": grand,
                "grand_squared": grand**2,
                "family_variance": float(np.mean(family**2)),
                "conditional_key_variance": float(np.mean(design**2)),
                "family_by_key_variance": float(np.mean(remaining**2)),
                "total_second_moment": float(np.mean(src**2)),
                "closure_error": abs(
                    float(np.mean(src**2))
                    - grand**2
                    - float(np.mean(family**2))
                    - float(np.mean(design**2))
                    - float(np.mean(remaining**2))
                ),
                "units": "source lexical families and crossed text designs, not human variance components",
            }
        )
    with (args.packet / "03/fourier_energies.csv").open(newline="") as stream:
        energies = list(csv.DictReader(stream))
    energy_group = defaultdict(list)
    for r in energies:
        energy_group[r["reader"], r["metric"], r["reference"]].append(r)
    energy_summary = []
    for (reader, metric, reference), rr in energy_group.items():
        denominator = sum(float(r["energy"]) for r in rr if int(r["mask"]) > 0)
        if denominator <= 0:
            continue
        energy_summary.append(
            {
                "reader": reader,
                "metric": metric,
                "reference": reference,
                "summed_within_cube_variance": denominator,
                "pair_and_triple_fraction": sum(
                    float(r["energy"]) for r in rr if int(r["mask"]).bit_count() >= 2
                )
                / denominator,
                "triple_fraction": sum(
                    float(r["energy"]) for r in rr if int(r["mask"]) == 7
                )
                / denominator,
                "summary_unit": "ratio of summed energies, not average cube fraction or population variance",
            }
        )
    write_csv(args.output / "source_application_uncertainty.csv", uncertainties)
    write_csv(args.output / "source_interaction_decomposition.csv", source_parts)
    write_csv(args.output / "reference_energy_summary.csv", energy_summary)
    result = {
        "packet": "HM-10A-result-after",
        "primary_manifest": sha(args.packet / "04/artifact_hashes.json"),
        "source_application_bootstrap": "independent lexical-family resampling, all fixed arms paired; not observer-refit or human CI",
        "draws": 2000,
        "uncertainty_rows": len(uncertainties),
        "decomposition_rows": len(source_parts),
        "max_source_decomposition_closure": max(
            r["closure_error"] for r in source_parts
        ),
        "encoder_parser_neural_api_calls": 0,
        "selection_changed": False,
    }
    write_json(args.output / "summary.json", result)
    write_json(
        args.output / "artifact_hashes.json",
        {
            str(f.relative_to(args.output)): sha(f)
            for f in sorted(args.output.rglob("*"))
            if f.is_file() and f.name != "artifact_hashes.json"
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
