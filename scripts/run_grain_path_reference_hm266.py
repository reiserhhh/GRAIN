#!/usr/bin/env python3
"""Locate stable path-reference components without a precision or target grid."""

from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_excursion_order as E
from grain_research.construction import math_path_reference as M
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_path_reference_hm266.json"
DEFAULT = ROOT / "results/grain_math_response_v1/path_reference_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_PATH_REFERENCE_PLAN_HM266.md",
    "grain_research/construction/math_path_reference.py",
    "grain_research/construction/math_excursion_order.py",
    "grain_research/construction/math_path_completion.py",
    "scripts/run_grain_path_reference_hm266.py",
    "scripts/score_grain_path_reference_cold.py",
    "scripts/run_grain_observed_family_hm170.py",
    "tests/test_grain_path_reference.py",
)


def prepare(out):
    """Freeze numeric cohort and the old SOURCE field; never rerun its stage."""
    config = R.read(ROOT / CONFIG)
    parent = ROOT / config["parent"]
    R.check(parent, ("computed_seal.json", "verification_seal.json"))
    arrays = R.prepared(parent)
    if (
        arrays["source_paths"].shape != (386, 2, 18)
        or arrays["held_paths"].shape != (120, 2, 18)
        or arrays["targets"].shape != (2, 386, 5)
        or np.intersect1d(arrays["source_indices"], arrays["held"]).size
    ):
        raise ValueError("Changed registered cohort")
    records = [
        json.loads(line)
        for line in (parent / "cold_inputs.jsonl").read_text().splitlines()
    ]
    ids = []
    for i, row in enumerate(records):
        u, w = divmod(i, 2)
        if not np.array_equal(E.parse_events(row), arrays["held_paths"][u, w]):
            raise ValueError("Changed cold event path")
        ids.extend(e["event_id"] for e in row["events"])
    if len(records) != 240 or len(set(ids)) != len(ids):
        raise ValueError("Changed scope or reused event identity")
    out.mkdir(parents=True, exist_ok=False)
    names = (
        "observations.npz",
        "cold_inputs.jsonl",
        "source_bundle.json",
        "accounts.npz",
        "prepared_seal.json",
        "computed_seal.json",
        "verification_seal.json",
    )
    R.write(
        out / "manifest.json",
        {
            "config": config,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {
                str((parent / p).relative_to(ROOT)): R.digest(parent / p) for p in names
            },
            "original_scope_complete": False,
        },
    )
    R.archive(out / "observations.npz", **arrays)
    with (out / "cold_inputs.jsonl").open("x") as file:
        for row in records:
            file.write(json.dumps(row, allow_nan=False) + "\n")
    R.write(
        out / "input_quality.json",
        {
            "source_authors": 386,
            "held_authors": 120,
            "whole_paths": 1012,
            "source_held_overlap": 0,
            "source_field_reused_without_fitting": True,
            "historical_public_derivatives_not_untouched_lockbox": True,
            "raw_private_held_labels_neural_API_training_calls": 0,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "cold_inputs.jsonl", "input_quality.json"),
    )
    print("PREPARED HM266 full386/120 scope; old fields unchanged", flush=True)


def correlation(a, b):
    """Describe finite SOURCE associations, with no score or threshold selection."""
    x, y = np.asarray(a), np.asarray(b)
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def covariance(a, b):
    """Keep the signed finite-population covariance, not a signal fraction."""
    x, y = np.asarray(a), np.asarray(b)
    return float(np.mean((x - x.mean()) * (y - y.mean()))) if len(x) else None


def source_accounts(scores, z):
    """All readings use one common complete set and opposite-bank scoring."""
    selected = scores[np.arange(386), :, 2 - z["bank"]]
    contacts, parts, sizes, errors = [], [], [], []
    for assignment in range(2):
        full = selected[:, :, assignment]
        valid = np.isfinite(full).all((1, 2, 3))
        pair = full[valid]
        sizes.append(int(valid.sum()))
        for kind, reading in enumerate(M.READINGS):
            for p in range(5):
                for q in range(5):
                    values = pair[:, :, kind, q].mean(1)
                    labels = z["targets"][assignment, valid, p]
                    contacts.append(
                        {
                            "assignment": E.ASSIGNMENTS[assignment],
                            "reading": reading,
                            "source_anchor": E.TARGETS[p],
                            "field_target": E.TARGETS[q],
                            "joint_complete_authors": int(valid.sum()),
                            "covariance": covariance(values, labels),
                            "Pearson_SOURCE_only": correlation(values, labels),
                            "cross_window_r": correlation(
                                pair[:, 0, kind, q], pair[:, 1, kind, q]
                            ),
                        }
                    )
        for q in range(5):
            contributions = []
            for left in (2, 3, 4):
                for right in (2, 3, 4):
                    value = covariance(pair[:, 0, left, q], pair[:, 1, right, q])
                    contributions.append(value)
                    parts.append(
                        {
                            "assignment": E.ASSIGNMENTS[assignment],
                            "field_target": E.TARGETS[q],
                            "left_component": M.READINGS[left],
                            "right_component": M.READINGS[right],
                            "joint_complete_authors": int(valid.sum()),
                            "cross_window_covariance": value,
                        }
                    )
            raw = covariance(pair[:, 0, 0, q], pair[:, 1, 0, q])
            errors.append(abs(sum(contributions) - raw) if raw is not None else None)
    return contacts, parts, sizes, errors


def geometry_toy():
    """One fixed indicator separates configuration and order on designed words."""
    result = []
    for path in ((0, 4, 0, 1, 0, 2, 0, 5, 0, 3), (0, 4, 0, 2, 0, 1, 0, 5, 0, 3)):
        fine, coarse = E.orbit_counts(path), M.family_counts(path)
        index, length = (1, 0, 2), len(path) - 2
        s = fine["observed_counts"][index] / length
        a = fine["numerator"][index] / (fine["denominator"] * length)
        c = coarse["numerator"][index] / (coarse["denominator"] * length)
        result.append(
            {
                "S": s,
                "A": a,
                "C": c,
                "Q": a - c,
                "D": s - a,
                "count_family_size": coarse["denominator"],
                "orbit_size": fine["orbit_size"],
            }
        )
    return {"readings": E.json_safe(result), "designed_not_human_target": True}


def compute(out):
    """One bounded pass over all paths; preserve signed accounts and missingness."""
    config = R.check(out)["config"]
    R.write(
        out / "compute_started.json",
        {"state": "RUNNING", "cap": config["compute_seconds_cap"]},
    )
    start = time.perf_counter()
    try:
        z = R.prepared(out)
        bundle = R.read(ROOT / config["parent"] / "source_bundle.json")
        bundle = {
            **bundle,
            "version": "HM266-v1",
            "source_field_version": bundle["version"],
            "readings": list(M.READINGS),
            "maximum_events": M.MAX_EVENTS,
        }
        paths = np.concatenate(
            (z["source_paths"].reshape(-1, 18), z["held_paths"].reshape(-1, 18))
        )
        accounts, rows = [], []
        for i, path in enumerate(paths):
            if time.perf_counter() - start > config["compute_seconds_cap"]:
                raise TimeoutError("Nonrenewable primary computation cap reached")
            a = M.evaluate(bundle, path)
            accounts.append(a)
            rows.append(
                {
                    "path": i,
                    "panel": "SOURCE" if i < 772 else "held",
                    "count_family_size": a["family_size"],
                    "orbit_size": a["orbit_size"],
                    **{
                        f"count_coverage_{ref}": float(a["count_coverage"][j])
                        for j, ref in enumerate(E.REFERENCES)
                    },
                    **{
                        f"orbit_coverage_{ref}": float(a["orbit_coverage"][j])
                        for j, ref in enumerate(E.REFERENCES)
                    },
                }
            )
        scores = np.array([a["scores"] for a in accounts])
        profiles = np.array([a["profile_readings"] for a in accounts])
        contacts, parts, sizes, errors = source_accounts(
            scores[:772].reshape(386, 2, 3, 2, 5, 5), z
        )
        summary = {
            "state": "COMPLETE_NOJOB",
            "whole_paths": len(paths),
            "source_crossfit_joint_complete_authors": sizes,
            "held_complete_count_reference_by_reference": np.isfinite(
                scores[772:, :, 0, 2, 0]
            )
            .sum(0)
            .tolist(),
            "held_original_orbit_reference_support": np.isfinite(
                scores[772:, :, 0, 1, 0]
            )
            .sum(0)
            .tolist(),
            "maximum_count_family_size": max(a["family_size"] for a in accounts),
            "strictly_larger_count_families": sum(
                a["family_size"] > a["orbit_size"] for a in accounts
            ),
            "cross_window_covariance_reconstruction_max_error": max(
                e for e in errors if e is not None
            ),
            "toy": geometry_toy(),
            "scientific_CPU_seconds": time.perf_counter() - start,
            "no_parameter_search_source_refit_or_component_deletion": True,
            "not_prediction_accuracy_or_psychological_variance_fraction": True,
            "original_scope_complete": False,
        }
        if summary["cross_window_covariance_reconstruction_max_error"] > 1e-12:
            raise ArithmeticError("Signed covariance decomposition failed")
        R.write(out / "source_bundle.json", bundle)
        R.archive(
            out / "accounts.npz",
            scores=scores,
            profiles=profiles,
            family_numerators=np.array([a["family_numerator"] for a in accounts]),
            family_sizes=np.array([a["family_size"] for a in accounts]),
            orbit_sizes=np.array([a["orbit_size"] for a in accounts]),
            count_coverage=np.array([a["count_coverage"] for a in accounts]),
            orbit_coverage=np.array([a["orbit_coverage"] for a in accounts]),
        )
        R.table(out / "path_geometry.csv", rows)
        R.table(out / "source_contacts.csv", contacts)
        R.table(out / "source_covariance_parts.csv", parts)
        R.write(out / "summary.json", summary)
        R.seal(
            out,
            "computed_seal.json",
            (
                "source_bundle.json",
                "accounts.npz",
                "path_geometry.csv",
                "source_contacts.csv",
                "source_covariance_parts.csv",
                "summary.json",
            ),
        )
        print(json.dumps(summary, allow_nan=False), flush=True)
    except Exception as error:
        R.write(
            out / "compute_failure.json",
            {"state": "FAILED_NOT_RETRIED", "error": repr(error)},
        )
        raise


def verify(out):
    """Focused alternative accounts and nested expectations, not precision tuning."""
    config = R.check(out, ("computed_seal.json",))["config"]
    z, bundle = R.prepared(out), R.read(out / "source_bundle.json")
    paths = np.concatenate(
        (z["source_paths"].reshape(-1, 18), z["held_paths"].reshape(-1, 18))
    )
    fields = np.asarray(bundle["fields"])
    margin_checks, enumerated, words_checked = 0, 0, 0
    scalar_error, old_error = 0.0, 0.0
    with (
        np.load(out / "accounts.npz", allow_pickle=False) as stored,
        np.load(ROOT / config["parent"] / "accounts.npz", allow_pickle=False) as old,
    ):
        for i, path in enumerate(paths):
            coarse = {
                "numerator": stored["family_numerators"][i],
                "denominator": int(stored["family_sizes"][i]),
            }
            if not M.family_margin_check(path, coarse):
                raise ArithmeticError("Full family changed cropped pair margins")
            margin_checks += 1
            if coarse["denominator"] <= config["verification_enumeration_bound"]:
                words = M.family_words(
                    path, maximum=config["verification_enumeration_bound"]
                )
                total = np.zeros((8, 8, 8), int)
                tower = np.zeros((8, 8, 8), object)
                for word in words:
                    total += E.triple_counts(word)
                    fine = E.orbit_counts(word)
                    for index, n in np.ndenumerate(fine["numerator"]):
                        if n:
                            tower[index] += Fraction(int(n), fine["denominator"])
                if not np.array_equal(total, coarse["numerator"]) or not np.array_equal(
                    tower, total
                ):
                    raise ArithmeticError(
                        "Full enumeration or conditional tower failed"
                    )
                enumerated += 1
                words_checked += len(words)
            old_values = old["profiles"][i]
            current = stored["profiles"][i][:, :, [0, 1, 4]]
            if not np.array_equal(np.isfinite(old_values), np.isfinite(current)):
                raise ArithmeticError("Old reading support was changed")
            valid = np.isfinite(old_values)
            if valid.any():
                old_error = max(
                    old_error, float(np.max(abs(old_values[valid] - current[valid])))
                )
            fine = E.orbit_counts(path)
            mass = [
                fine["observed_counts"] / 16,
                fine["numerator"] / (16 * fine["denominator"]),
                coarse["numerator"] / (16 * coarse["denominator"]),
            ]
            mass += [mass[1] - mass[2], mass[0] - mass[1]]
            for ref in range(3):
                for assignment in range(2):
                    for kind in range(5):
                        expected = stored["profiles"][i, ref, assignment, kind]
                        if not np.isfinite(expected).all():
                            continue
                        value = sum(
                            float(n) * fields[ref, assignment, :, a, b, c]
                            for (a, b, c), n in np.ndenumerate(mass[kind])
                            if n
                        )
                        scalar_error = max(
                            scalar_error, float(np.max(abs(value - expected)))
                        )
        if max(old_error, scalar_error) > 1e-12:
            raise ArithmeticError("Old fields or direct profile contraction changed")
    result = {
        "PASS": True,
        "all_path_exact_mass_and_pair_margin_checks": margin_checks,
        "small_full_families_enumerated_with_tower": enumerated,
        "numeric_words_not_new_subjects": words_checked,
        "alternate_profile_sum_max_error": scalar_error,
        "unchanged_old_S_A_D_readback_max_error": old_error,
        "same_agent_not_blind": True,
    }
    R.write(out / "verification.json", result)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(result), flush=True)


def main():
    """Immutable stages with no automatic retry, model fitting or next queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "compute", "verify", "check"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(args.out, ("computed_seal.json", "verification_seal.json"))
        print("ALL HM266 SEALS PASS")
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
