#!/usr/bin/env python3
"""Run one finite complete-path construction on sealed public numeric material."""

from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from math import factorial, prod
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_path_completion as M
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_path_completion_hm264.json"
DEFAULT = ROOT / "results/grain_math_response_v1/path_completion_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_PATH_COMPLETION_PLAN_HM264.md",
    "grain_research/construction/math_path_completion.py",
    "scripts/run_grain_path_completion_hm264.py",
    "scripts/score_grain_path_completion_cold.py",
    "tests/test_grain_path_completion.py",
    "scripts/run_grain_observed_family_hm170.py",
)
STAGES = ("computed_seal.json", "verification_seal.json")


def records(path):
    """Read only pre-attested numeric records, rejecting duplicate event use."""
    result = [json.loads(row) for row in Path(path).read_text().splitlines()]
    identifiers = []
    for row in result:
        M.parse_events(row)
        identifiers.extend(e["event_id"] for e in row["events"])
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Event appears in multiple held records")
    if len({r["author_key"] for r in result}) != len(result):
        raise ValueError("Repeated held record")
    return result


def prepare(out):
    """Bind the existing prepared stage; do not invoke the failed HM235 primary."""
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    sealed = R.read(parent / "prepared_seal.json")
    names = ("observations.npz", "cold_inputs.jsonl", "input_quality.json")
    for name in names:
        if R.digest(parent / name) != sealed[name]:
            raise ValueError("Changed prepared parent " + name)
    keys = (
        "source_paths",
        "targets",
        "bank",
        "source_indices",
        "held",
        "held_paths",
        "held_times",
        "held_units",
    )
    with np.load(parent / "observations.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in keys}
    if (
        arrays["source_paths"].shape != (386, 2, 18)
        or arrays["targets"].shape != (2, 386, 5)
        or arrays["held_paths"].shape != (120, 2, 18)
        or np.intersect1d(arrays["source_indices"], arrays["held"]).size
        or not np.array_equal(np.bincount(arrays["bank"]), [203, 183])
        or np.any(np.diff(arrays["held_times"], axis=-1) <= 0)
    ):
        raise ValueError("Changed SOURCE/held scope or chronology")
    cold = records(parent / "cold_inputs.jsonl")
    if len(cold) != 240:
        raise ValueError("Changed full held scope")
    for index, row in enumerate(cold):
        u, w = divmod(index, 2)
        expected = f"held/{int(arrays['held'][u])}/window/{w}"
        if row["author_key"] != expected or not np.array_equal(
            M.parse_events(row), arrays["held_paths"][u, w]
        ):
            raise ValueError("Changed held event address")
    out.mkdir(parents=True, exist_ok=False)
    input_names = tuple(
        str((parent / name).relative_to(ROOT))
        for name in names + ("manifest.json", "prepared_seal.json")
    )
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {p: R.digest(ROOT / p) for p in input_names},
            "original_scope_complete": False,
            "cohort_reuse": "historical numeric public cohorts;not untouched lockbox",
        },
    )
    R.archive(out / "observations.npz", **arrays)
    with (out / "cold_inputs.jsonl").open("x") as f:
        for row in cold:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    quality = {
        "source_authors": 386,
        "source_catalogues": 772,
        "held_authors": 120,
        "held_whole_paths": 240,
        "events_per_path": 18,
        "source_held_overlap": 0,
        "SOURCE_only_old_binary_anchors_and_single_permutation": True,
        "raw_text_application_labels_new_model_API_private_training_calls": 0,
        "sampled_publication_not_conversation": True,
        "old_primary_IPF_failure_preserved_not_called": True,
        "original_scope_complete": False,
    }
    R.write(out / "input_quality.json", quality)
    with (out / "data_schema.md").open("x") as f:
        f.write(
            "# HM264 numeric schema\n\nSOURCE386 authors/two18-state windows, actual and old-permuted five-bit anchors; fixed banks203/183. Held120 authors/two18-state windows, unique event ID and strict publication time. Historical eight states derive from a neural encoder and coarse partition, not newly discovered personality classes. No held anchors, raw text or questionnaire body. Complete ordered windows are the units; no overlapping-triple independence.\n"
        )
    R.seal(
        out,
        "prepared_seal.json",
        (
            "observations.npz",
            "cold_inputs.jsonl",
            "input_quality.json",
            "data_schema.md",
        ),
    )
    print(json.dumps(quality), flush=True)


def aggregate(scores, support):
    """Retain coverage and all signed target differences by every prefix."""
    rows = []
    for ref in range(3):
        for assignment in range(2):
            for length in range(scores.shape[3]):
                for law in range(3):
                    for q in range(5):
                        a = scores[:, ref, assignment, length, law, q]
                        b = scores[:, ref, assignment, length, 0, q]
                        valid = np.isfinite(a)
                        shared = valid & np.isfinite(b)
                        rows.append(
                            {
                                "reference": M.REFERENCES[ref],
                                "assignment": ("actual", "old_permutation")[assignment],
                                "prefix_events": length,
                                "law": M.LAWS[law],
                                "target": M.TARGETS[q],
                                "supported_paths": int(valid.sum()),
                                "common_with_local": int(shared.sum()),
                                "mean_score": float(a[valid].mean())
                                if valid.any()
                                else None,
                                "mean_difference_from_local": float(
                                    (a - b)[shared].mean()
                                )
                                if shared.any()
                                else None,
                                "RMS_difference_from_local": float(
                                    np.sqrt(np.mean((a - b)[shared] ** 2))
                                )
                                if shared.any()
                                else None,
                                "mean_support_authors": float(
                                    support[:, ref, length, law].mean()
                                ),
                            }
                        )
    return rows


def compute(out):
    """One fixed CPU pass; retain all law/reference/profile/prefix outcomes."""
    cfg = R.check(out)["config"]
    if (out / "compute_started.json").exists():
        raise ValueError("Primary stage already attempted;no automatic retry")
    R.write(
        out / "compute_started.json",
        {"state": "RUNNING", "nonrenewable_seconds_cap": cfg["compute_seconds_cap"]},
    )
    start = time.perf_counter()
    try:
        z = R.prepared(out)
        bundle = M.compile_bundle(z["source_paths"], z["targets"], z["bank"])
        R.write(out / "source_bundle.json", bundle)
        source_rows = [
            {
                "catalogue": i,
                "source_author": c["source_author_index"],
                "window": c["source_window"],
                "bank": c["bank"],
                "start": c["start"],
                "end": c["end"],
                "state_words": c["words"],
                "success_numerator": c["success"][0],
                "success_denominator": c["success"][1],
                "success_probability": c["success"][0] / c["success"][1],
            }
            for i, c in enumerate(bundle["catalogues"])
        ]
        R.table(out / "source_inventory.csv", source_rows)
        collected = []
        for i, row in enumerate(records(out / "cold_inputs.jsonl")):
            if time.perf_counter() - start > cfg["compute_seconds_cap"]:
                raise TimeoutError(
                    "One primary CPU cap exhausted;no support/precision rescue"
                )
            collected.append(M.score_record(bundle, row))
            if i % 40 == 0:
                print(
                    json.dumps(
                        {
                            "paths_completed": i + 1,
                            "scope": 240,
                            "elapsed": time.perf_counter() - start,
                        }
                    ),
                    flush=True,
                )
        duration = time.perf_counter() - start
        if duration > cfg["compute_seconds_cap"]:
            raise TimeoutError("Primary exceeds the registered nonrenewable cap")
        array_names = (
            "scores",
            "departures",
            "profiles",
            "evidence_probability",
            "support_catalogues",
            "support_authors",
            "doomed_local_weight_fraction",
        )
        result = {
            name: np.array([row[name] for row in collected]) for name in array_names
        }
        R.archive(out / "prefix_accounts.npz", **result)
        R.table(
            out / "prefix_summary.csv",
            aggregate(result["scores"], result["support_authors"]),
        )
        with (out / "cold_scores.jsonl").open("x") as f:
            for row in collected:
                f.write(
                    json.dumps(
                        {
                            k: M.serializable(v)
                            for k, v in row.items()
                            if k != "profiles"
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
        h = np.array([v["success_probability"] for v in source_rows])
        complete = result["support_authors"][:, :, -1] > 0
        summary = {
            "state": "COMPLETE_NOJOB",
            "scientific_CPU_seconds": duration,
            "whole_held_paths": len(collected),
            "prefixes_per_path": 19,
            "source_success_min_median_max": [
                float(h.min()),
                float(np.median(h)),
                float(h.max()),
            ],
            "source_success_equal_one_count": int(np.sum(h == 1)),
            "prior_actual_by_reference_and_law": M.serializable(
                result["scores"][0, :, 0, 0]
            ),
            "full18_support_count_reference_by_law": complete.sum(0).tolist(),
            "all_prefix_support_count_reference_by_law": (result["support_authors"] > 0)
            .sum(0)
            .tolist(),
            "exact_fraction_prefix_checks": sum(
                r["exact_prefix_checks"] for r in collected
            ),
            "toy": M.toy_results(),
            "joint_law_not_human_personality_posterior": True,
            "unsupported_values_kept_null": True,
            "no_smoothing_prefix_selection_parameter_grid": True,
            "original_scope_complete": False,
        }
        R.write(out / "summary.json", summary)
        R.seal(
            out,
            "computed_seal.json",
            (
                "source_bundle.json",
                "source_inventory.csv",
                "prefix_accounts.npz",
                "prefix_summary.csv",
                "cold_scores.jsonl",
                "summary.json",
            ),
        )
        R.write(
            out / "compute_completed.json",
            {"state": "COMPLETE_NOJOB", "CPU_seconds": duration},
        )
        print(json.dumps(summary), flush=True)
    except Exception as error:
        R.write(
            out / "compute_error.json",
            {
                "state": "FAILED_NOJOB",
                "type": type(error).__name__,
                "message": str(error),
                "seconds": time.perf_counter() - start,
            },
        )
        raise


def determinant_fraction(matrix):
    """Independent rational Gaussian elimination, not Bareiss or NumPy det."""
    a = [[Fraction(int(x)) for x in row] for row in matrix]
    d = Fraction(1)
    for k in range(len(a)):
        pivot = next((i for i in range(k, len(a)) if a[i][k]), None)
        if pivot is None:
            return 0
        if pivot != k:
            a[k], a[pivot] = a[pivot], a[k]
            d *= -1
        p = a[k][k]
        d *= p
        for i in range(k + 1, len(a)):
            ratio = a[i][k] / p
            for j in range(k + 1, len(a)):
                a[i][j] -= ratio * a[k][j]
    if d.denominator != 1:
        raise ArithmeticError("Integer matrix determinant became noninteger")
    return d.numerator


def verify(out):
    """Audit source counts and every recorded posterior without rescoring a grid."""
    R.check(out, ("computed_seal.json",))
    bundle, z = R.read(out / "source_bundle.json"), R.prepared(out)
    counts, likelihood_error = 0, 0.0
    for c in bundle["catalogues"]:
        n = np.array(c["counts"], int)
        alternative = M.word_count(n, c["start"], c["end"], determinant_fraction)
        if alternative != c["words"]:
            raise ArithmeticError("Alternate determinant count disagreement")
        path = z["source_paths"][c["source_author_index"], c["source_window"]]
        v, _ = M.likelihoods(path, c["start"], c["end"], n)
        local = Fraction(
            prod(factorial(int(i)) for i in n.ravel()),
            prod(factorial(int(i)) for i in n.sum(1)),
        )
        likelihood_error = max(
            likelihood_error,
            abs(v[-1, 0] - float(local)),
            abs(v[-1, 1] - 1 / alternative),
        )
        counts += 1
    with np.load(out / "prefix_accounts.npz", allow_pickle=False) as f:
        a = {k: f[k] for k in f.files}
    support = a["support_catalogues"] > 0
    valid = np.isfinite(a["profiles"])
    if not np.array_equal(
        valid.all(-1), np.broadcast_to(support[:, :, None], valid.shape[:-1])
    ):
        raise ArithmeticError("Missing profile/support disagreement")
    alphabet = ((np.arange(32)[:, None] >> np.arange(5)) & 1).astype(float)
    reconstruction = a["profiles"] @ alphabet
    score_error = float(np.nanmax(abs(reconstruction - a["scores"])))
    departure_error = float(
        np.nanmax(abs(a["scores"] - a["scores"][:, :, :, :1] - a["departures"]))
    )
    probability_error = float(np.max(abs(a["profiles"][valid.all(-1)].sum(-1) - 1)))
    first = records(out / "cold_inputs.jsonl")[0]
    cold = M.score_record(bundle, first)
    cold_error = float(np.nanmax(abs(cold["scores"] - a["scores"][0])))
    if (
        max(
            likelihood_error,
            score_error,
            departure_error,
            probability_error,
            cold_error,
        )
        > 1e-12
    ):
        raise ArithmeticError("Recorded numeric account mismatch")
    value = {
        "PASS": True,
        "alternate_exact_SOURCE_counts": counts,
        "source_fullpath_factorial_likelihood_error": likelihood_error,
        "profile_to_five_score_error": score_error,
        "own_law_prior_departure_error": departure_error,
        "profile_probability_error": probability_error,
        "one_fixed_cold_replay_error": cold_error,
        "profile_rows": int(valid.all(-1).sum()),
        "full240_path_scope": len(a["scores"]) == 240,
        "same_agent_not_blind": True,
        "old_IPF_and_all_old_seals_untouched": True,
    }
    R.write(out / "verification.json", value)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(value), flush=True)


def main():
    """Run only an explicitly selected immutable stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "compute", "verify", "check"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(args.out, STAGES)
        print(
            json.dumps(
                {
                    "PASS": True,
                    "state": "VERIFIED_COMPLETE_NOJOB",
                    "original_scope_complete": False,
                }
            )
        )
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
