#!/usr/bin/env python3
"""One finite context/program ambiguity experiment, without neural calls or fits."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_conditional_mean_fiber as M
from grain_research.construction import math_context_program_fiber as C
from scripts import run_grain_same_evidence_hm137 as H

SPEC = ROOT / "configs/grain_context_program_fibers_hm138.json"
CODE = (
    "configs/grain_context_program_fibers_hm138.json",
    "docs/GRAIN_CONTEXT_ACCOUNT_AND_PROGRAM_FIBER_PLAN_HM138.md",
    "grain_research/construction/math_context_program_fiber.py",
    "grain_research/construction/math_conditional_mean_fiber.py",
    "scripts/run_grain_context_program_fibers_hm138.py",
    "tests/test_grain_context_program_fiber.py",
)


def fixed_inputs(spec):
    """Authenticate the complete frozen parent before touching scientific values."""
    parent = ROOT / spec["parent"]
    if (
        H.read(parent / "verification.json")["status"]
        != "VERIFIED_FINITE_HM137_NOT_GLOBAL"
    ):
        raise ValueError("Verified HM137 required")
    for path, digest in H.read(parent / "02/seal.json").items():
        if H.sha(path) != digest:
            raise ValueError("Changed parent output: " + path)
    manifest, _, g, native, _ = H.arrays(parent)
    if spec["events"] != 96 or spec["strata"] != 24:
        raise ValueError("Do not silently reduce support")
    return manifest, g.reshape(96, 15), native[..., :2].reshape(96, 2, 2, 7, 2)


def compute(spec):
    """Evaluate every fixed summary/contract and integrate all readout consequences."""
    started = time.monotonic()
    manifest, g, y = fixed_inputs(spec)
    d = M.conditional_contrasts(24)
    address = np.indices((2, 2, 6, 2, 2))
    corpus, wording = address[0].ravel(), address[1].ravel()
    conditional = C.context_account_features(g, corpus * 2 + wording)
    views = {
        "global_mean": g,
        "corpus_joint_mean": C.context_account_features(g, corpus),
        "wording_joint_mean": C.context_account_features(g, wording),
        "corpus_wording_joint_mean": conditional,
        "global_mean_second_moments": M.summary_columns(g, True),
    }
    if list(views) != spec["summary_views"]:
        raise ValueError("Declared summary battery changed")
    targets = manifest["previous_targets"] + ["action_fact"]
    rows, ranks, saved = [], [], {}
    errors = {
        "constraint": 0.0,
        "margin": 0.0,
        "normalization": 0.0,
        "target_integration": 0.0,
        "alternate_projection": 0.0,
    }
    counts, numerical_warnings = {}, []
    for section, names in (
        ("summary", spec["summary_views"]),
        ("contract", spec["observer_contracts"]),
    ):
        for name in names:
            if section == "summary":
                features, agreed, neutral = views[name], [], []
                values = y.reshape(96, 56)
                channels = list(np.ndindex(2, 2, 7, 2))
            else:
                extra, agreed, neutral = C.observer_contract(y, name)
                features = np.column_stack([g, extra])
                values = y.mean(1).reshape(96, 28)
                channels = [(None, a, q, u) for a, q, u in np.ndindex(2, 7, 2)]
            null = C.constraint_fiber(features, d, spec["rtol"])
            key = section + "__" + name
            saved[key + "__basis"] = null["basis"]
            ranks.append(
                {
                    "section": section,
                    "name": name,
                    "columns": features.shape[1],
                    "rank": null["rank"],
                    "raw_rank": null["raw_rank"],
                    "nullity": null["nullity"],
                    "cutoff": null["cutoff"],
                    "singular_values": null["singular_values"].tolist(),
                    "raw_singular_values": null["raw_singular_values"].tolist(),
                }
            )
            if null["rank"] != null["raw_rank"]:
                numerical_warnings.append(key + ": raw/normalized ranks differ")
            # Independent least-squares projection onto the constraint column space.
            a = null["normalized_constraints"].T
            alternate = (
                np.eye(24) - a @ np.linalg.lstsq(a, np.eye(24), rcond=spec["rtol"])[0]
            )
            errors["alternate_projection"] = max(
                errors["alternate_projection"],
                float(np.max(abs(alternate - null["basis"] @ null["basis"].T))),
            )
            counts[key] = 0
            for j, (program, actor, query, unit) in enumerate(channels):
                if time.monotonic() - started > spec["max_seconds"]:
                    raise TimeoutError("Fixed HM138 numerical cap reached")
                w = M.positive_witness(
                    values[:, j], d, null["basis"], spec["tolerance"]
                )
                row = {
                    "section": section,
                    "view": name,
                    "program": "average" if program is None else ("H", "full")[program],
                    "actor": ("A", "B")[actor],
                    "target": targets[query],
                    "unit": ("score", "logodds")[unit],
                    "target_agreement_required": (actor, query, unit) in agreed,
                    "status": w["status"],
                    "residual_norm": w["residual_norm"],
                    "delta_target": w["difference"],
                    "delta_H": None,
                    "delta_full": None,
                    "agreement_error_max": None,
                    "required_fact_error_max": None,
                    "A_fact_score_change_max": None,
                    "A_fact_logodds_change_max": None,
                    "B_trait_score_change_max": None,
                    "context_account_change_max": None,
                    "minimum_probability": None,
                    "mean_error_max": None,
                }
                if w["plus"] is not None:
                    change = w["plus"] - w["minus"]
                    reading = (change @ y.reshape(96, 56)).reshape(2, 2, 7, 2)
                    error = float(np.max(abs(change @ features)))
                    errors["constraint"] = max(errors["constraint"], error)
                    margins = change.reshape(24, 2, 2)
                    errors["margin"] = max(
                        errors["margin"],
                        float(np.max(abs(margins.sum(1)))),
                        float(np.max(abs(margins.sum(2)))),
                    )
                    errors["normalization"] = max(
                        errors["normalization"],
                        abs(float(w["plus"].sum()) - 1),
                        abs(float(w["minus"].sum()) - 1),
                    )
                    direct = sum(float(change[i] * values[i, j]) for i in range(96))
                    errors["target_integration"] = max(
                        errors["target_integration"], abs(direct - w["difference"])
                    )
                    row.update(
                        delta_H=float(reading[0, actor, query, unit]),
                        delta_full=float(reading[1, actor, query, unit]),
                        agreement_error_max=max(
                            (
                                abs(
                                    float(
                                        reading[0, aa, qq, uu] - reading[1, aa, qq, uu]
                                    )
                                )
                                for aa, qq, uu in agreed
                            ),
                            default=0.0,
                        ),
                        required_fact_error_max=max(
                            (
                                abs(float(reading[rr, aa, qq, uu]))
                                for rr, aa, qq, uu in neutral
                            ),
                            default=0.0,
                        ),
                        A_fact_score_change_max=float(np.max(abs(reading[:, 0, 6, 0]))),
                        A_fact_logodds_change_max=float(
                            np.max(abs(reading[:, 0, 6, 1]))
                        ),
                        B_trait_score_change_max=float(
                            np.max(abs(reading[:, 1, :6, 0]))
                        ),
                        context_account_change_max=float(
                            np.max(abs(change @ conditional))
                        ),
                        minimum_probability=float(
                            min(w["plus"].min(), w["minus"].min())
                        ),
                        mean_error_max=float(np.max(abs(change @ g))),
                    )
                    if (
                        row["minimum_probability"] <= 0
                        or row["agreement_error_max"] > spec["tolerance"]
                        or row["required_fact_error_max"] > spec["tolerance"]
                    ):
                        raise ValueError("Positive-law contract violated")
                    saved[key + f"__q{j}__plus"] = w["plus"]
                    saved[key + f"__q{j}__minus"] = w["minus"]
                    saved[key + f"__q{j}__all_native_changes"] = reading
                    counts[key] += 1
                rows.append(row)
    if len(rows) != 420 or max(errors.values()) > spec["tolerance"]:
        raise ValueError("Incomplete battery or failed arithmetic")
    result = {
        "status": "FINITE_CONTEXT_PROGRAM_FIBERS_COMPUTED",
        "rank_cases": ranks,
        "rows": len(rows),
        "positive_law_counts": counts,
        "errors": errors,
        "numerical_warnings": numerical_warnings,
        "seconds": time.monotonic() - started,
        "new_neural_calls": 0,
        "predictor_fits": 0,
        "existing_data_diagnostic": True,
        "original_scope_complete": False,
    }
    return result, rows, saved


def execute(spec, out):
    """Freeze source hashes before running; preserve a failed run without rescue."""
    out.mkdir(parents=True, exist_ok=False)
    parent = ROOT / spec["parent"]
    H.write(
        out / "manifest.json",
        {
            "spec": spec,
            "code": {str(ROOT / p): H.sha(ROOT / p) for p in CODE},
            "parent_verification": H.sha(parent / "verification.json"),
            "parent_output_seal": H.sha(parent / "02/seal.json"),
            "post_HM137_existing_data_diagnostic": True,
        },
    )
    try:
        result, rows, arrays = compute(spec)
        H.table(out / "all_results.csv", rows)
        np.savez(out / "positive_laws.npz", **arrays)
        H.write(out / "summary.json", result)
        H.write(out / "seal.json", {str(p.resolve()): H.sha(p) for p in out.iterdir()})
    except BaseException as error:
        H.write(
            out / "failure.json",
            {
                "status": "FAILED_NOT_RETRIED",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    print(result, flush=True)


def verify(spec, out):
    """Read-only complete replay plus scalar integration of every saved witness."""
    manifest = H.read(out / "manifest.json")
    for path, digest in manifest["code"].items():
        if H.sha(path) != digest:
            raise ValueError("Changed frozen HM138 source")
    for path, digest in H.read(out / "seal.json").items():
        if H.sha(path) != digest:
            raise ValueError("Changed HM138 artifact")
    result, rows, arrays = compute(spec)
    expected = H.read(out / "summary.json")
    result.pop("seconds")
    expected.pop("seconds")
    if result != expected:
        raise ValueError("Scientific replay changed")
    with (out / "all_results.csv").open(newline="") as f:
        stored = list(csv.DictReader(f))
    for row, saved_row in zip(rows, stored, strict=True):
        for key, value in row.items():
            if saved_row[key] != ("" if value is None else str(value)):
                raise ValueError("Changed result table field")
    _, _, native = fixed_inputs(spec)
    max_error, pairs = 0.0, 0
    with np.load(out / "positive_laws.npz", allow_pickle=False) as z:
        if set(z.files) != set(arrays):
            raise ValueError("Missing or extra positive law")
        for key, value in arrays.items():
            np.testing.assert_array_equal(z[key], value)
            if not key.endswith("__plus"):
                continue
            prefix = key.removesuffix("__plus")
            change = z[key] - z[prefix + "__minus"]
            # Scalar summation is separate from matrix integration in compute().
            direct = np.empty((2, 2, 7, 2))
            for channel in np.ndindex(2, 2, 7, 2):
                direct[channel] = sum(
                    float(change[i] * native[(i,) + channel]) for i in range(96)
                )
            max_error = max(
                max_error,
                float(np.max(abs(direct - z[prefix + "__all_native_changes"]))),
            )
            pairs += 1
    if max_error > spec["tolerance"]:
        raise ValueError("Independent scalar integration failed")
    print(
        {
            "status": "VERIFIED_FINITE_HM138_READONLY",
            "rows": len(rows),
            "positive_pairs": pairs,
            "scalar_integration_max_error": max_error,
            "alternate_projection_error": result["errors"]["alternate_projection"],
            "same_agent_not_blind_replication": True,
        },
        flush=True,
    )


def main():
    """Use separate immutable collection and repeatable read-only verification."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    spec = H.read(SPEC)
    with threadpool_limits(4):
        (verify if args.verify else execute)(spec, ROOT / spec["out"])


if __name__ == "__main__":
    main()
