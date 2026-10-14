#!/usr/bin/env python3
"""Run a finite SOURCE-qualified complete-path conditional order measurement."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_excursion_order as M
from grain_research.construction.math_path_completion import count_matrix
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_excursion_order_hm265.json"
DEFAULT = ROOT / "results/grain_math_response_v1/excursion_order_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_EXCURSION_ORDER_PLAN_HM265.md",
    "grain_research/construction/math_excursion_order.py",
    "grain_research/construction/math_path_completion.py",
    "scripts/run_grain_excursion_order_hm265.py",
    "scripts/score_grain_excursion_order_cold.py",
    "scripts/run_grain_observed_family_hm170.py",
    "tests/test_grain_excursion_order.py",
)


def prepare(out):
    """Use sealed numeric derivatives only; never call an old scientific stage."""
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
    event_ids = []
    for i, row in enumerate(records):
        u, w = divmod(i, 2)
        if not np.array_equal(M.parse_events(row), arrays["held_paths"][u, w]):
            raise ValueError("Changed cold event address")
        event_ids.extend(e["event_id"] for e in row["events"])
    if len(records) != 240 or len(set(event_ids)) != len(event_ids):
        raise ValueError("Changed complete held scope or repeated events")
    out.mkdir(parents=True, exist_ok=False)
    names = ("observations.npz", "cold_inputs.jsonl", "prepared_seal.json")
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
            "whole_windows": 1012,
            "source_held_overlap": 0,
            "source_association_other_bank_only": True,
            "historical_public_numeric_not_untouched_lockbox": True,
            "raw_text_held_labels_new_neural_API_private_training_calls": 0,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "cold_inputs.jsonl", "input_quality.json"),
    )
    print("PREPARED full386/120 scope", flush=True)


def correlation(x, y):
    """Describe a finite association; return null for too few/constant values."""
    a, b = np.asarray(x), np.asarray(y)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or a[keep].std() == 0 or b[keep].std() == 0:
        return None
    return float(np.corrcoef(a[keep], b[keep])[0, 1])


def source_contacts(scores, z):
    """All target contacts use opposite-bank scoring, retaining both assignments."""
    rows = []
    selected = scores[np.arange(386), :, 2 - z["bank"]]
    for assignment in range(2):
        for kind, name in enumerate(("raw", "orbit_reference", "order_contrast")):
            pair = selected[:, :, assignment, kind]
            valid = np.isfinite(pair).all((1, 2))
            for p in range(5):
                for q in range(5):
                    values = pair[valid, :, q].mean(1)
                    labels = z["targets"][assignment, valid, p]
                    covariance = (
                        float(
                            np.mean((values - values.mean()) * (labels - labels.mean()))
                        )
                        if valid.any()
                        else None
                    )
                    rows.append(
                        {
                            "assignment": M.ASSIGNMENTS[assignment],
                            "reading": name,
                            "source_anchor": M.TARGETS[p],
                            "field_target": M.TARGETS[q],
                            "complete_authors": int(valid.sum()),
                            "covariance": covariance,
                            "Pearson_SOURCE_only": correlation(values, labels),
                            "cross_window_r": correlation(pair[:, 0, q], pair[:, 1, q]),
                        }
                    )
    return rows


def geometry_toy():
    """A designed conditional-order contrast, not a fitted personality example."""
    paths = ((0, 4, 0, 1, 0, 2, 0, 5, 0, 3), (0, 4, 0, 2, 0, 1, 0, 5, 0, 3))
    result = []
    for path in paths:
        account = M.orbit_counts(path)
        result.append(
            float(
                (
                    account["observed_counts"][1, 0, 2]
                    - account["numerator"][1, 0, 2] / account["denominator"]
                )
                / (len(path) - 2)
            )
        )
    return {
        "equal_full_transition_counts": bool(
            np.array_equal(count_matrix(paths[0]), count_matrix(paths[1]))
        ),
        "contrasts": result,
        "designed_target_not_human": True,
    }


def compute(out):
    """One bounded pass; keep all paths, SOURCE references and signed profiles."""
    config = R.check(out)["config"]
    R.write(
        out / "compute_started.json",
        {"state": "RUNNING", "cap": config["compute_seconds_cap"]},
    )
    start = time.perf_counter()
    try:
        z = R.prepared(out)
        bundle = M.compile_bundle(z["source_paths"], z["targets"], z["bank"])
        paths = np.concatenate(
            (z["source_paths"].reshape(-1, 18), z["held_paths"].reshape(-1, 18))
        )
        accounts, rows = [], []
        for i, path in enumerate(paths):
            if time.perf_counter() - start > config["compute_seconds_cap"]:
                raise TimeoutError("Nonrenewable primary cap reached")
            a = M.evaluate(bundle, path)
            accounts.append(a)
            rows.append(
                {
                    "path": i,
                    "panel": "SOURCE" if i < 772 else "held",
                    "anchor": a["anchor"],
                    "interior_blocks": a["blocks"],
                    "orbit_size": a["orbit_size"],
                    **{
                        f"coverage_{ref}": float(a["coverage"][j])
                        for j, ref in enumerate(M.REFERENCES)
                    },
                }
            )
        scores = np.array([a["scores"] for a in accounts])
        profiles = np.array([a["profile_readings"] for a in accounts])
        source_scores = scores[:772].reshape(386, 2, 3, 2, 3, 5)
        contacts = source_contacts(source_scores, z)
        summary = {
            "state": "COMPLETE_NOJOB",
            "whole_paths": len(paths),
            "source_crossfit_complete_authors": sorted(
                {r["complete_authors"] for r in contacts}
            ),
            "held_complete_paths_by_reference": np.isfinite(scores[772:, :, 0, 2, 0])
            .sum(0)
            .tolist(),
            "held_orbit_size_gt_one": sum(a["orbit_size"] > 1 for a in accounts[772:]),
            "SOURCE_orbit_size_gt_one": sum(
                a["orbit_size"] > 1 for a in accounts[:772]
            ),
            "maximum_orbit_size": max(a["orbit_size"] for a in accounts),
            "held_order_contrast_RMS_by_reference_assignment_target": M.json_safe(
                np.sqrt(np.nanmean(scores[772:, :, :, 2] ** 2, axis=0))
            ),
            "toy": geometry_toy(),
            "scientific_CPU_seconds": time.perf_counter() - start,
            "not_whole_person_posterior_accuracy_or_human_causal_effect": True,
            "no_kernel_fit_smoothing_or_parameter_search": True,
            "original_scope_complete": False,
        }
        R.write(out / "source_bundle.json", bundle)
        R.archive(
            out / "accounts.npz",
            scores=scores,
            profiles=profiles,
            numerators=np.array([a["numerator"] for a in accounts]),
            denominators=np.array([a["denominator"] for a in accounts]),
            orbit_sizes=np.array([a["orbit_size"] for a in accounts]),
            coverage=np.array([a["coverage"] for a in accounts]),
        )
        R.table(out / "path_geometry.csv", rows)
        R.table(out / "source_contacts.csv", contacts)
        R.write(out / "summary.json", summary)
        R.seal(
            out,
            "computed_seal.json",
            (
                "source_bundle.json",
                "accounts.npz",
                "path_geometry.csv",
                "source_contacts.csv",
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
    """Alternative finite enumeration and scalar accounts; not blind replication."""
    R.check(out, ("computed_seal.json",))
    z, bundle = R.prepared(out), R.read(out / "source_bundle.json")
    paths = np.concatenate(
        (z["source_paths"].reshape(-1, 18), z["held_paths"].reshape(-1, 18))
    )
    with np.load(out / "accounts.npz", allow_pickle=False) as stored:
        fields = np.array(bundle["fields"])
        margin_checks, enumerated, words_checked, scalar_error = 0, 0, 0, 0.0
        for i, path in enumerate(paths):
            geometry = M.orbit_counts(path)
            if not M.adjacent_margin_check(path, geometry):
                raise ArithmeticError("Cropped pair margins changed")
            margin_checks += 1
            if geometry["orbit_size"] <= 720:
                words = M.orbit_words(path)
                total = np.zeros((8, 8, 8), np.int64)
                for word in words:
                    if not np.array_equal(count_matrix(word), count_matrix(path)):
                        raise ArithmeticError("Orbit changed pair counts")
                    total += M.triple_counts(word)
                if not np.array_equal(
                    total * geometry["denominator"], geometry["numerator"] * len(words)
                ):
                    raise ArithmeticError("Exact orbit mean disagrees with enumeration")
                enumerated += 1
                words_checked += len(words)
            for ref in range(3):
                for assignment in range(2):
                    expected = stored["profiles"][i, ref, assignment]
                    if not np.isfinite(expected).all():
                        continue
                    raw = (
                        sum(
                            int(n) * fields[ref, assignment, :, a, b, c]
                            for (a, b, c), n in np.ndenumerate(M.triple_counts(path))
                        )
                        / 16
                    )
                    orbit = sum(
                        int(n) * fields[ref, assignment, :, a, b, c]
                        for (a, b, c), n in np.ndenumerate(geometry["numerator"])
                    ) / (16 * geometry["denominator"])
                    alternate = np.array([raw, orbit, raw - orbit])
                    scalar_error = max(
                        scalar_error, float(np.max(abs(alternate - expected)))
                    )
        if scalar_error > 1e-12:
            raise ArithmeticError("Direct profile sum disagreement")
    result = {
        "PASS": True,
        "all_path_exact_pair_margin_checks": margin_checks,
        "exhaustive_small_orbits": enumerated,
        "enumerated_numeric_words_not_new_subjects": words_checked,
        "alternate_profile_sum_max_error": scalar_error,
        "same_agent_not_blind": True,
    }
    R.write(out / "verification.json", result)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(result), flush=True)


def main():
    """Run an immutable stage; there is no automatic retry or next-packet queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "compute", "verify", "check"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(args.out, ("computed_seal.json", "verification_seal.json"))
        print("ALL SEALS PASS")
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
