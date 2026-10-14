#!/usr/bin/env python3
"""Result-after analytic transport accounting, without fitting or new model calls."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts import run_grain_conditional_chart_hm114 as R


def run(out):
    """Decompose fixed chart transport into offset, level mixing and response gain."""
    manifest = R.check(out)
    if (
        H.load(out / "verification.json")["status"]
        != "VERIFIED_FINITE_HM114_NOT_GLOBAL"
    ):
        raise ValueError("This diagnostic requires complete primary finite charts")
    for seal in (
        "02/instrument_seal.json",
        "02/observation_seal.json",
        "03/output_seal.json",
    ):
        H.verify_hashes(out, H.load(out / seal))
    folder = out / "04_transport_diagnostic"
    folder.mkdir()
    charts = H.load(out / "02/calibration.json")
    with (out / "03/author_charts.csv").open() as handle:
        scored = list(csv.DictReader(handle))
    variance_a = np.var(manifest["spec"]["levels"])
    variance_b = np.var(manifest["spec"]["responses"])
    maps, rows, max_error = {}, [], 0.0
    for family in range(6):
        for unit in range(2):
            source, target = charts[f"0:{family}:{unit}"], charts[f"1:{family}:{unit}"]
            gs, gt = np.asarray(source["gain"]), np.asarray(target["gain"])
            delta = np.asarray(target["offset"]) - source["offset"]
            norms2 = np.sum(gs**2, axis=1)
            h = np.sum(gs * gt, axis=1) / norms2
            d = np.sum(gs * delta, axis=1) / norms2
            transport = np.array(
                [[h.mean(), (h[1] - h[0]) / 2], [(h[1] - h[0]) / 2, h.mean()]]
            )
            offset = np.array([d.mean(), (d[1] - d[0]) / 2])
            subset = [
                r
                for r in scored
                if r["mode"] == "source"
                and r["bank"] == "1"
                and int(r["family"]) == family
                and int(r["unit"]) == unit
            ]
            errors = []
            for r in subset:
                prediction = offset + transport @ [
                    float(r["level"]),
                    float(r["response"]),
                ]
                actual = [float(r["observed_level"]), float(r["observed_response"])]
                max_error = max(max_error, float(np.max(abs(prediction - actual))))
                errors.append(float(r["response_error"]) ** 2)
            components = np.array(
                [
                    offset[1] ** 2,
                    transport[1, 0] ** 2 * variance_a,
                    (transport[1, 1] - 1) ** 2 * variance_b,
                ]
            )
            max_error = max(max_error, abs(float(components.sum()) - np.mean(errors)))
            key = f"{family}:{unit}"
            maps[key] = {
                "offset": offset.tolist(),
                "matrix": transport.tolist(),
                "condition_gain_ratios": h.tolist(),
                "response_MSE_components": components.tolist(),
            }
            rows.append(
                {
                    "family": manifest["spec"]["targets"][family],
                    "unit": unit,
                    "B_offset": float(offset[1]),
                    "level_to_B": float(transport[1, 0]),
                    "B_gain": float(transport[1, 1]),
                    "offset_MSE": float(components[0]),
                    "level_mixing_MSE": float(components[1]),
                    "response_gain_MSE": float(components[2]),
                    "observed_response_MSE": float(np.mean(errors)),
                }
            )
    if max_error > manifest["spec"]["identity_atol"]:
        raise ValueError("Analytic transport does not replay actual cold application")
    summaries = []
    for unit in range(2):
        subset = [r for r in rows if r["unit"] == unit]
        components = np.array(
            [
                [r["offset_MSE"], r["level_mixing_MSE"], r["response_gain_MSE"]]
                for r in subset
            ]
        ).mean(0)
        summaries.append(
            {
                "unit": unit,
                "mean_response_MSE_components": components.tolist(),
                "response_MSE_fractions": (components / components.sum()).tolist(),
                "not_personality_information_fractions": True,
            }
        )
    coupling = H.load(out / "03/target_coupling.json")
    diagnostics = {}
    for key, record in coupling.items():
        singular = np.asarray(record["singular_values"])
        cosines = np.asarray(record["family_cosines_by_condition"])
        off_diagonal = cosines[:, ~np.eye(6, dtype=bool)]
        diagnostics[key] = {
            "rank": record["rank"],
            "condition_number": float(singular[0] / singular[-1]),
            "off_diagonal_cosine_min_max": [
                float(off_diagonal.min()),
                float(off_diagonal.max()),
            ],
            "mean_abs_off_diagonal_cosine": float(np.mean(abs(off_diagonal))),
            "not_author_correlation": True,
        }
    H.table(folder / "transport_components.csv", rows)
    H.write(folder / "affine_maps.json", maps)
    H.write(
        folder / "summary.json",
        {
            "status": "RESULT_AFTER_ANALYTIC_DIAGNOSTIC_NOT_NEW_EXPERIMENT",
            "affine_replay_max_error": max_error,
            "units": summaries,
            "target_coupling": diagnostics,
            "new_model_calls": 0,
            "fits": 0,
            "condition_semantics_not_automatically_noise": True,
        },
    )
    H.write(
        folder / "seal.json",
        {
            "primary_verification_sha256": H.sha(out / "verification.json"),
            "diagnostic_code_sha256": H.sha(Path(__file__)),
            "outputs": H.seals(out, folder),
        },
    )
    print("TRANSPORT_DIAGNOSTIC", summaries, "alternate error", max_error)


def main():
    """Run only once in a new companion directory; preserve all primary results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()
