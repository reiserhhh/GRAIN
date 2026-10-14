#!/usr/bin/env python3
"""Result-after HM-07 unit/availability diagnostics; no new neural operations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from run_grain_math_binding import read_bank
from run_grain_math_response import ROOT, sha, write_csv, write_json
from verify_grain_math_purpose_accounts import table

from grain_research.construction.math_conditional_variance import product_components
from grain_research.construction.math_purpose_accounts import ACCOUNTS

DEFAULT = ROOT / "results/grain_math_response_v1/observation_dag_units_diagnostic_r2"
FACTORS = ("action_family", "mode", "object_style", "polarity", "actor")


def availability(rows):
    """Separate parser coverage, supported endpoints and numeric author scores."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["reader"], row["target"], row["account"]].append(row)
    result = []
    for (reader, target, account), values in groups.items():
        scored = [r for r in values if r["level"] != "" and r["response"] != ""]
        result.append(
            {
                "reader": reader,
                "target": target,
                "account": account,
                "programs": len(values),
                "parser_complete": sum(r["complete"] == "True" for r in values),
                "level_and_response_available": len(scored),
                "missing_condition_endpoint_programs": sum(
                    any(v is None for v in json.loads(r["endpoints"]).values())
                    for r in values
                ),
                "level_rmse_on_available": float(
                    np.sqrt(
                        np.mean(
                            [
                                (float(r["level"]) - float(r["expected_level"])) ** 2
                                for r in scored
                            ]
                        )
                    )
                )
                if scored
                else None,
                "response_rmse_on_available": float(
                    np.sqrt(
                        np.mean(
                            [
                                (float(r["response"]) - float(r["expected_response"]))
                                ** 2
                                for r in scored
                            ]
                        )
                    )
                )
                if scored
                else None,
                "scope": "controlled author programs; no human psychological labels",
            }
        )
    return result


def main():
    """Freeze a companion without changing primary r1/r2 or their thresholds."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--packet",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/observation_dag_r2",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT)
    args = ap.parse_args()
    hashes = json.loads((args.packet / "04/artifact_hashes.json").read_text())
    for name, digest in hashes.items():
        if sha(args.packet / name) != digest:
            raise ValueError("changed primary artifact " + name)
    signature = {
        "primary_manifest": sha(args.packet / "04/artifact_hashes.json"),
        "runner": sha(Path(__file__)),
        "decomposition": sha(
            ROOT / "grain_research/construction/math_conditional_variance.py"
        ),
    }
    contract = args.out / "00/diagnostic_contract.json"
    if contract.exists() and json.loads(contract.read_text())["signature"] != signature:
        raise ValueError("result-after companion already frozen with different code")
    write_json(
        contract,
        {
            "signature": signature,
            "result_after": True,
            "new_fit_or_encoding_or_parser_calls": 0,
            "primary": str(args.packet),
            "estimand": "unit errors under uniform controlled product; not population decomposition",
            "factors": FACTORS,
            "no_threshold_or_source_model_changes": True,
        },
    )
    bank = read_bank(args.packet / "00/documents.jsonl")
    metadata = {r["material_id"]: r for r in bank}
    event_rows = table(args.packet / "02/event_readouts.csv")
    profiles = availability(table(args.packet / "03/author_profiles.csv"))
    write_csv(args.out / "01/author_score_availability.csv", profiles)
    groups, failures = defaultdict(list), []
    for row in event_rows:
        material = metadata[row["material_id"]]
        if (
            row["arm"] not in {"dag_typed", "declared_typed"}
            or material["kind"] != "single"
        ):
            continue
        if row["account"] == "actual":
            groups[row["reader"], row["target"], row["arm"]].append(row)
        expected = material["events"][0]
        if (
            row["account"] == expected["mode"]
            and expected["bits"][0] == 1
            and expected["actor"] == 1
            and row["eligible"] != "True"
        ):
            failures.append(
                {
                    **row,
                    "text": material["text"],
                    "primary_missing": False,
                    "failed_layer": "target_relevance"
                    if float(row["k"]) <= 0
                    else "role_or_mode_qualification",
                    "threshold_changed": False,
                }
            )
    write_csv(args.out / "01/qualified_target_failures.csv", failures)
    spectral, summaries = [], []
    for (reader, target, arm), rows in groups.items():
        values, occupied = np.full((8, 5, 2, 2, 2), np.nan), set()
        for row in rows:
            material = metadata[row["material_id"]]
            e = material["events"][0]
            key = (
                material["family"],
                ACCOUNTS.index(e["mode"]),
                material["style"],
                int(e["bits"][1] == 1),
                int(e["actor"] == 1),
            )
            if key in occupied:
                raise ValueError("duplicate product cell")
            occupied.add(key)
            values[key] = float(row["k"]) - e["bits"][0]
        if len(occupied) != values.size:
            raise ValueError("incomplete product design")
        result = product_components(values)
        for mask, energy in result["energies"].items():
            spectral.append(
                {
                    "reader": reader,
                    "target": target,
                    "arm": arm,
                    "mask": mask,
                    "factors": "+".join(
                        f for j, f in enumerate(FACTORS) if mask & (1 << j)
                    )
                    or "constant_bias",
                    "order": mask.bit_count(),
                    "squared_energy": energy,
                    "fraction_of_error_variance": energy / result["variance"]
                    if mask
                    else None,
                }
            )
        summaries.append(
            {
                "reader": reader,
                "target": target,
                "arm": arm,
                "cells": values.size,
                **{
                    k: result[k]
                    for k in (
                        "mean",
                        "variance",
                        "mse",
                        "max_reconstruction_error",
                        "max_orthogonality_error",
                        "variance_closure_error",
                    )
                },
                "mode_involving_error_fraction": sum(
                    v for m, v in result["energies"].items() if m & 2
                )
                / result["variance"],
                "object_style_involving_error_fraction": sum(
                    v for m, v in result["energies"].items() if m & 4
                )
                / result["variance"],
                "interaction_error_fraction": sum(
                    v for m, v in result["energies"].items() if m.bit_count() >= 2
                )
                / result["variance"],
                "separate_fraction_budgets_can_overlap": True,
            }
        )
    write_csv(args.out / "01/conditional_error_spectrum.csv", spectral)
    write_csv(args.out / "01/conditional_error_summary.csv", summaries)
    indices = json.loads((args.packet / "01/observation_indices.json").read_text())
    indexed = {}
    for row in event_rows:
        if row["arm"] in {"dag_typed", "declared_typed"} and row["account"] == "actual":
            indexed[row["reader"], row["material_id"], row["arm"]] = row
    discrepancies = []
    for i, material in enumerate(bank):
        if material["kind"] != "single":
            continue
        automatic, declared = indices["dag"][i], indices["declared"][i]
        if len(automatic) != 1 or len(declared) != 1:
            raise ValueError("single observation interface unexpectedly changed")
        automatic, declared = automatic[0], declared[0]
        for reader in ("e5", "bge"):
            a = indexed[reader, material["material_id"], "dag_typed"]
            d = indexed[reader, material["material_id"], "declared_typed"]
            discrepancies.append(
                {
                    "reader": reader,
                    "target": material["target"],
                    "material_id": material["material_id"],
                    "mode": material["events"][0]["mode"],
                    "automatic_action": automatic["action"],
                    "declared_action": declared["action"],
                    "automatic_object": automatic["object"],
                    "declared_object": declared["object"],
                    "same_action_input": automatic["action_ref"]
                    == declared["action_ref"],
                    "same_object_input": automatic["object_ref"]
                    == declared["object_ref"],
                    "k_automatic": float(a["k"]),
                    "k_declared": float(d["k"]),
                    "delta_k": float(a["k"]) - float(d["k"]),
                    "reference_is_declared_not_automatic": True,
                }
            )
    write_csv(args.out / "01/representation_discrepancies.csv", discrepancies)
    write_json(
        args.out / "02/summary.json",
        {
            "status": "result_after_diagnostic_complete",
            "primary_hashes_verified": len(hashes),
            "profiles": profiles,
            "parser_complete_programs": sum(r["parser_complete"] for r in profiles),
            "numerically_scored_programs": sum(
                r["level_and_response_available"] for r in profiles
            ),
            "missed_expected_single_observations": len(failures),
            "missed_observations_by_layer": {
                k: sum(r["failed_layer"] == k for r in failures)
                for k in {r["failed_layer"] for r in failures}
            },
            "automatic_expected_observation_failures": sum(
                r["arm"] == "dag_typed" for r in failures
            ),
            "declared_reference_observation_failures": sum(
                r["arm"] == "declared_typed" for r in failures
            ),
            "single_interface_comparisons": len(discrepancies),
            "action_input_discrepancies": sum(
                not r["same_action_input"] for r in discrepancies
            ),
            "object_input_discrepancies": sum(
                not r["same_object_input"] for r in discrepancies
            ),
            "nonzero_k_difference_with_identical_inputs": sum(
                abs(r["delta_k"]) > 1e-12
                and r["same_action_input"]
                and r["same_object_input"]
                for r in discrepancies
            ),
            "conditional_decompositions": summaries,
            "new_fit_or_encoding_or_parser_calls": 0,
            "full_goal_complete": False,
        },
    )
    write_json(
        args.out / "02/artifact_hashes.json",
        {
            str(p.relative_to(args.out)): sha(p)
            for p in sorted(args.out.rglob("*"))
            if p.is_file()
            and not p.name.startswith("._")
            and p != args.out / "02/artifact_hashes.json"
        },
    )
    print(
        f"HM-07 diagnostic: {sum(r['level_and_response_available'] for r in profiles)}/{sum(r['programs'] for r in profiles)} author programs scored; {len(failures)} missed target events"
    )


if __name__ == "__main__":
    main()
