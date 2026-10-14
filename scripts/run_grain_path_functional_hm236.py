#!/usr/bin/env python3
"""Construct source-fixed path accounts and cold feasible ranges without IPF."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_path_functional as M
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_source_meaning_bridge_hm233 as S

CONFIG = "configs/grain_path_functional_hm236.json"
DEFAULT = ROOT / "results/grain_math_response_v1/path_functional_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_PATH_FUNCTIONAL_PLAN_HM236.md",
    "grain_research/construction/math_path_functional.py",
    "grain_research/construction/math_observed_path_memory.py",
    "scripts/run_grain_path_functional_hm236.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_source_meaning_bridge_hm233.py",
    "tests/test_grain_path_functional.py",
)
KEYS = (
    "reference",
    "point_field",
    "lower_field",
    "upper_field",
    "target_AB",
    "target_BC",
    "circuit_index",
    "circuit_amplitude",
)


def bundles(out):
    """Load only compiled aggregate laws, with no SOURCE/application label rows."""
    z = S.selected(
        out / "cold_bundle.npz", tuple(f"{k}_{i}" for i in range(2) for k in KEYS)
    )
    return [{k: z[f"{k}_{i}"] for k in KEYS} for i in range(2)]


def prepare(out):
    """Bind scope and existing source-only inputs before new scientific outcomes."""
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    R.check(parent)
    failure = R.read(parent / "compute_failure.json")
    if "Fixed IPF budget exhausted" not in failure.get("message", ""):
        raise ValueError("HM235 original failure must remain present")
    z = S.selected(
        parent / "observations.npz",
        ("source_paths", "targets", "source_indices", "held"),
    )
    if (
        z["source_paths"].shape != (386, 2, 18)
        or z["targets"].shape != (2, 386, 5)
        or len(z["held"]) != 120
        or np.intersect1d(z["source_indices"], z["held"]).size
    ):
        raise ValueError("Changed complete source/held author scope")
    count, identities, keys = 0, set(), set()
    with (parent / "cold_inputs.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            M.history_mass(rec)
            if len(rec["events"]) != 18 or rec["author_key"] in keys:
                raise ValueError("Changed window length or duplicate author-window")
            keys.add(rec["author_key"])
            for event in rec["events"]:
                if event["event_id"] in identities:
                    raise ValueError("Repeated evidence across held windows")
                identities.add(event["event_id"])
            count += 1
    if count != cfg["application_windows"]:
        raise ValueError("Must preserve all240 cold windows")
    out.mkdir(parents=True, exist_ok=False)
    inputs = (
        "manifest.json",
        "prepared_seal.json",
        "observations.npz",
        "cold_inputs.jsonl",
        "compute_failure.json",
    )
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {
                str((parent / p).relative_to(ROOT)): R.digest(parent / p)
                for p in inputs
            },
            "new_question": "functional identification on the actual path constraint fiber,not reconstruction of HM235 H",
            "cohort_reuse": "historical representation/held panel;not untouched lockbox",
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "source_inputs.npz", source_paths=z["source_paths"], targets=z["targets"]
    )
    quality = {
        "SOURCE_authors": 386,
        "held_authors": 120,
        "held_windows": count,
        "held_unique_events": len(identities),
        "SOURCE_held_overlap": 0,
        "SOURCE_overlapping_triples": 12352,
        "overlap_not_independence": True,
        "old_actual_and_single_permutation": True,
        "new_neural_API_private_train_application_label_calls": 0,
        "HM235_primary": "NOT_COMPUTED;failure untouched",
    }
    R.write(out / "input_quality.json", quality)
    R.seal(out, "prepared_seal.json", ("source_inputs.npz", "input_quality.json"))
    print(json.dumps(quality), flush=True)


def compile_source(out):
    """Compile fixed pair margins and a feasible circuit catalogue once."""
    cfg = R.check(out)["config"]
    z = S.selected(out / "source_inputs.npz", ("source_paths", "targets"))
    start = time.perf_counter()
    mass = np.array(
        [[M.triple_mass(path) for path in row] for row in z["source_paths"]]
    )
    pooled = mass.mean(1)
    tables, arrays, rows = [], {}, []
    bits = M.profile_bits()
    for i, name in enumerate(cfg["laws"]):
        table = np.zeros((32, 8, 8, 8))
        codes = z["targets"][i] @ (1 << np.arange(5))
        if not np.isin(z["targets"][i], (0, 1)).all():
            raise ValueError("Changed binary SOURCE assignment")
        np.add.at(table, codes, pooled / 386)
        law = M.compile_law(table)
        tables.append(table)
        arrays.update({f"{k}_{i}": v for k, v in law.items()})
        r = law["reference"]
        point = np.einsum("gq,gabc->qabc", bits, table)
        lower, upper = law["lower_field"] * r, law["upper_field"] * r
        used = r > 0
        for q, target in enumerate(cfg["targets"]):
            rows.append(
                {
                    "label_law": name,
                    "target": target,
                    "source_reference_target": float(
                        (bits[:, q] * table.sum((1, 2, 3))).sum()
                    ),
                    "positive_circuits": int((law["circuit_index"][:, 0] == q).sum()),
                    "supported_path_cells": int(used.sum()),
                    "point_outer_violation": max(
                        0.0,
                        float(np.max(lower[q, used] - point[q, used])),
                        float(np.max(point[q, used] - upper[q, used])),
                    ),
                    "source_weighted_outer_width": float(
                        np.sum(
                            r[used]
                            * (
                                law["upper_field"][q, used]
                                - law["lower_field"][q, used]
                            )
                        )
                    ),
                }
            )
    seconds = time.perf_counter() - start
    if seconds > cfg["compute_seconds_cap"]:
        raise TimeoutError("Source compile exceeded finite cap;no automatic retry")
    R.archive(out / "source_laws.npz", full_tables=np.array(tables), profiles=bits)
    R.archive(out / "cold_bundle.npz", **arrays)
    R.table(out / "source_functional_accounts.csv", rows)
    receipt = {
        "status": "SOURCE_COMPILED",
        "CPU_wall_seconds": seconds,
        "all_target_accounts": rows,
        "total_candidate_cubes_per_law": 501760,
        "no_IPF_LP_or_target_predictor_fit": True,
        "full_profile_and_choice_conditions_retained": True,
        "original_scope_complete": False,
    }
    R.write(out / "source_receipt.json", receipt)
    R.seal(
        out,
        "compiled_seal.json",
        (
            "source_laws.npz",
            "cold_bundle.npz",
            "source_functional_accounts.csv",
            "source_receipt.json",
        ),
    )
    print(json.dumps(receipt), flush=True)


def cold(out):
    """Emit ranges on all held histories in a process reading no individual labels."""
    cfg = R.check(out, ("compiled_seal.json",))["config"]
    law = bundles(out)
    start = time.perf_counter()
    count, supported = 0, np.zeros(2, int)
    with (
        (ROOT / cfg["parent"] / "cold_inputs.jsonl").open() as f,
        (out / "cold_outputs.jsonl").open("x") as g,
    ):
        for line in f:
            if time.perf_counter() - start > cfg["compute_seconds_cap"]:
                raise TimeoutError("Cold finite cap exhausted;no automatic retry")
            rec = json.loads(line)
            h = M.history_mass(rec)
            readings = {
                name: M.measure_mass(law[i], h) for i, name in enumerate(cfg["laws"])
            }
            supported += [int(readings[name]["supported"]) for name in cfg["laws"]]
            g.write(
                json.dumps(
                    {
                        "author_key": rec["author_key"],
                        "unique_events": len(rec["events"]),
                        "overlapping_triples": len(rec["events"]) - 2,
                        "laws": readings,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            count += 1
    if count != cfg["application_windows"]:
        raise ValueError("Cold scope incomplete")
    receipt = {
        "status": "COLD_COMPLETE",
        "windows": count,
        "supported_counts": supported.tolist(),
        "CPU_wall_seconds": time.perf_counter() - start,
        "individual_SOURCE_arrays_or_application_labels_read": False,
        "only_cold_bundle_and_observed_events": True,
        "new_neural_API_private_train_calls": 0,
        "ranges_are_not_CI_or_psychological_validity": True,
    }
    R.write(out / "cold_receipt.json", receipt)
    R.seal(out, "cold_seal.json", ("cold_outputs.jsonl", "cold_receipt.json"))
    print(json.dumps(receipt), flush=True)


def analyze(out):
    """Retain all target/law readings; do not select a predictor or sign."""
    cfg = R.check(out, ("compiled_seal.json", "cold_seal.json"))["config"]
    values = [
        json.loads(line)
        for line in (out / "cold_outputs.jsonl").read_text().splitlines()
    ]
    reference = {
        (r["label_law"], r["target"]): r["source_reference_target"]
        for r in R.read(out / "source_receipt.json")["all_target_accounts"]
    }
    detailed, summary = [], []
    for name in cfg["laws"]:
        for q, target in enumerate(cfg["targets"]):
            rows = []
            for value in values:
                result = value["laws"][name]
                if not result["supported"]:
                    detailed.append(
                        {
                            "author_key": value["author_key"],
                            "label_law": name,
                            "target": target,
                            "supported": False,
                        }
                    )
                    continue
                row = result["targets"][q]
                rows.append(row)
                detailed.append(
                    {
                        "author_key": value["author_key"],
                        "label_law": name,
                        "target": target,
                        "supported": True,
                        **row,
                    }
                )
            if not rows:
                summary.append(
                    {"label_law": name, "target": target, "supported_windows": 0}
                )
                continue
            gap = np.array([r["certified_feasible_width"] for r in rows])
            outer = np.array([r["outer_width"] for r in rows])
            center = reference[name, target]
            summary.append(
                {
                    "label_law": name,
                    "target": target,
                    "supported_windows": len(rows),
                    "positive_certificates": int((gap > 1e-12).sum()),
                    "certified_width_mean": float(gap.mean()),
                    "certified_width_median": float(np.median(gap)),
                    "certified_width_max": float(gap.max()),
                    "outer_width_mean": float(outer.mean()),
                    "outer_width_median": float(np.median(outer)),
                    "outer_width_max": float(outer.max()),
                    "witness_straddles_source_reference": sum(
                        min(r["witness_plus_score"], r["witness_minus_score"])
                        < center
                        < max(r["witness_plus_score"], r["witness_minus_score"])
                        for r in rows
                    ),
                    "retained_joint_account_mean": float(
                        np.mean([r["retained_joint_account"] for r in rows])
                    ),
                    "retained_joint_account_RMS": float(
                        np.sqrt(
                            np.mean([r["retained_joint_account"] ** 2 for r in rows])
                        )
                    ),
                    "fixed_rectangle_max_error": max(
                        r["fixed_rectangle_max_error"] for r in rows
                    ),
                    "eligible_positive_circuits": rows[0]["eligible_positive_circuits"],
                }
            )
    # A uniform schema keeps failed/unsupported rows without narrowing the scope.
    columns = tuple(dict.fromkeys(k for row in detailed for k in row))
    R.table(
        out / "all_cold_functionals.csv",
        [{k: row.get(k) for k in columns} for row in detailed],
    )
    R.table(out / "functional_summary.csv", summary)
    result = {
        "status": "ANALYZED_PENDING_DIRECT_WITNESS_VERIFICATION",
        "all_laws_targets": summary,
        "units": "target-posterior functional differences,notr/AUC/humantrait effects",
        "M2_finite_certificate_found": any(
            r.get("positive_certificates", 0) > 0 for r in summary
        ),
        "HM235_primary": "NOT_COMPUTED;this is not F-H or a substitute",
        "outer_not_globally_sharp": True,
        "circuits_not_complete_basis": True,
        "original_scope_complete": False,
    }
    R.write(out / "summary.json", result)
    R.seal(
        out,
        "analyzed_seal.json",
        ("all_cold_functionals.csv", "functional_summary.csv", "summary.json"),
    )
    print(json.dumps(result), flush=True)


def verify(out):
    """Directly sum both full laws for every saved witness and every target."""
    cfg = R.check(out, ("compiled_seal.json", "cold_seal.json", "analyzed_seal.json"))[
        "config"
    ]
    z = S.selected(out / "source_laws.npz", ("full_tables", "profiles"))
    law = bundles(out)
    errors = dict.fromkeys(
        (
            "point",
            "fixed_account",
            "pair_margin",
            "pooled_path",
            "witness_score",
            "other_target",
            "outer_containment",
            "mass_total",
        ),
        0.0,
    )
    start, checked, minimum = time.perf_counter(), 0, 1.0
    with (
        (ROOT / cfg["parent"] / "cold_inputs.jsonl").open() as f,
        (out / "cold_outputs.jsonl").open() as g,
    ):
        for input_line, output_line in zip(f, g, strict=True):
            rec, saved = json.loads(input_line), json.loads(output_line)
            if rec["author_key"] != saved["author_key"]:
                raise ValueError("Misbound author-window output")
            h = M.history_mass(rec)
            for i, name in enumerate(cfg["laws"]):
                f0, bundle = z["full_tables"][i], law[i]
                result = saved["laws"][name]
                if not result["supported"]:
                    if not np.any((h > 0) & (bundle["reference"] == 0)):
                        raise ValueError("False unsupported reading")
                    continue
                r, fitted, _, _ = M.additive_account(h, bundle["reference"])
                direct = np.einsum("gq,gabc,abc->q", z["profiles"], f0, r)
                fixed = np.einsum("gq,gabc,abc->q", z["profiles"], f0, fitted)
                for q, row in enumerate(result["targets"]):
                    errors["point"] = max(
                        errors["point"], abs(direct[q] - row["empirical_point"])
                    )
                    errors["fixed_account"] = max(
                        errors["fixed_account"],
                        abs(fixed[q] - row["fixed_additive_account"]),
                    )
                    index = row["witness_index"]
                    if index is None:
                        continue
                    plus, minus = M.witness_laws(
                        f0,
                        bundle["circuit_index"][index],
                        bundle["circuit_amplitude"][index],
                    )
                    minimum = min(minimum, float(plus.min()), float(minus.min()))
                    for p in (plus, minus):
                        errors["mass_total"] = max(
                            errors["mass_total"], abs(p.sum() - 1)
                        )
                        errors["pair_margin"] = max(
                            errors["pair_margin"],
                            *(
                                float(np.max(abs(p.sum(axis) - f0.sum(axis))))
                                for axis in (1, 3)
                            ),
                        )
                        errors["pooled_path"] = max(
                            errors["pooled_path"],
                            float(np.max(abs(p.sum(0) - f0.sum(0)))),
                        )
                    sp = np.einsum("gq,gabc,abc->q", z["profiles"], plus, r)
                    sm = np.einsum("gq,gabc,abc->q", z["profiles"], minus, r)
                    errors["witness_score"] = max(
                        errors["witness_score"],
                        abs(sp[q] - row["witness_plus_score"]),
                        abs(sm[q] - row["witness_minus_score"]),
                        abs(sp[q] - sm[q] - row["signed_witness_plus_minus_gap"]),
                    )
                    errors["other_target"] = max(
                        errors["other_target"],
                        float(np.max(abs(np.delete(sp - sm, q)))),
                    )
                    errors["outer_containment"] = max(
                        errors["outer_containment"],
                        row["outer_low"] - min(sp[q], sm[q]),
                        max(sp[q], sm[q]) - row["outer_high"],
                    )
                    checked += 1
    passed = minimum >= 0 and max(errors.values()) <= cfg["audit_absolute_tolerance"]
    receipt = {
        "PASS": bool(passed),
        "errors": errors,
        "selected_witness_accounts": checked,
        "minimum_full_law_mass": minimum,
        "CPU_wall_seconds": time.perf_counter() - start,
        "same_agent_not_blind": True,
        "no_rerun_or_optimization": True,
        "original_scope_complete": False,
    }
    R.write(out / "verification.json", receipt)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(receipt), flush=True)
    if not passed:
        raise ValueError("Direct finite witness verification failed")


def main():
    """Run exactly one bounded stage; preserve any failure without auto-retry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "compile", "cold", "analyze", "verify", "check")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    functions = {
        "prepare": prepare,
        "compile": compile_source,
        "cold": cold,
        "analyze": analyze,
        "verify": verify,
    }
    if args.stage == "check":
        R.check(
            args.output,
            (
                "compiled_seal.json",
                "cold_seal.json",
                "analyzed_seal.json",
                "verification_seal.json",
            ),
        )
        print(json.dumps({"PASS": True, "science_seals_unchanged": True}))
    else:
        try:
            functions[args.stage](args.output)
        except Exception as error:
            if args.output.exists():
                R.write(
                    args.output / f"{args.stage}_failure.json",
                    {
                        "type": type(error).__name__,
                        "message": str(error),
                        "no_auto_retry": True,
                    },
                )
            raise


if __name__ == "__main__":
    main()
