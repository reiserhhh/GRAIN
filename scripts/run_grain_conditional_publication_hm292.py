#!/usr/bin/env python3
"""Build and evaluate fixed conditional-publication measurements,not forecasts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_conditional_publication as C
from scripts import run_grain_behavior_pushforward_hm291 as R

CONFIG = "configs/grain_conditional_publication_hm292.json"
CODE = (
    CONFIG,
    "docs/GRAIN_CONDITIONAL_PUBLICATION_PLAN_HM292.md",
    "grain_research/construction/math_conditional_publication.py",
    "scripts/run_grain_conditional_publication_hm292.py",
    "scripts/score_grain_conditional_publication_cold.py",
    "tests/test_grain_conditional_publication.py",
    "scripts/run_grain_behavior_pushforward_hm291.py",
)
DEFAULT = ROOT / "results/grain_math_response_v1/conditional_publication_r1"
TRAITS = ("E", "N", "A", "C", "O")


def outcomes(metadata, indices):
    """Bind all34 eligible publication pairs to the old author/window addresses."""
    return np.array(
        [
            [
                C.pair_observations(
                    metadata["lengths"][u, a:b],
                    metadata["times"][u, a:b],
                    metadata["venue_codes"][u, a:b],
                )
                for a, b in ((0, 18), (18, 36))
            ]
            for u in indices
        ]
    )


def check(out, stages=()):
    """Reuse ordinary source/artifact checks,not old scientific stages."""
    return R.check(out, stages)


def prepare(out):
    """Seal the new behavioral/criterion construction before measuring contacts."""
    cfg = R.read(ROOT / CONFIG)
    R.check(ROOT / cfg["parent"], ("computed_seal.json", "verification_seal.json"))
    out.mkdir(parents=True, exist_ok=False)
    inputs = (cfg["paths"], cfg["metadata"])
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "code": {p: R.digest(ROOT / p) for p in CODE},
            "inputs": {p: R.digest(ROOT / p) for p in inputs},
            "historical_source_not_fresh_external_validation": True,
            "original_scope_complete": False,
        },
    )
    z, metadata = R.load(ROOT / cfg["paths"]), R.load(ROOT / cfg["metadata"])
    if (
        len(z["source_indices"]) != 386
        or len(z["held"]) != 120
        or np.intersect1d(z["source_indices"], z["held"]).size
    ):
        raise ValueError("Declared complete disjoint author roles required")
    if not np.array_equal(
        metadata["times"][z["held"]].reshape(120, 2, 18), z["held_times"]
    ):
        raise ValueError("Different vector/metadata observations")
    if not np.array_equal(
        metadata["original_units"][z["held"]].reshape(120, 2, 18), z["held_units"]
    ):
        raise ValueError("Different event addresses")
    R.save(
        out / "observations.npz",
        {
            "source_states": z["source_paths"][:, :, :-1],
            "source_outputs": outcomes(metadata, z["source_indices"]),
            "source_labels": z["targets"],
            "bank": z["bank"],
            "held_states": z["held_paths"][:, :, :-1],
            "held_outputs": outcomes(metadata, z["held"]),
            "donors": z["donors"],
            "source_indices": z["source_indices"],
            "held_indices": z["held"],
        },
    )
    u = int(z["held"][0])
    record = {
        "author_key": "opaque-example",
        "pairs": [
            {
                "origin_id": f"event-{int(metadata['original_units'][u, t])}",
                "destination_id": f"event-{int(metadata['original_units'][u, t + 1])}",
                "origin_time": float(metadata["times"][u, t]),
                "destination_time": float(metadata["times"][u, t + 1]),
                "origin_cell": int(z["held_paths"][0, 0, t]),
                "next_span": float(metadata["lengths"][u, t + 1]),
                "same_venue": bool(
                    metadata["venue_codes"][u, t] == metadata["venue_codes"][u, t + 1]
                ),
            }
            for t in range(17)
        ],
    }
    R.write(out / "example_record.json", record)
    quality = {
        "source_authors": 386,
        "held_authors": 120,
        "source_pairs": 13124,
        "held_pairs": 4080,
        "source_banks": [int(np.sum(z["bank"] == b)) for b in (0, 1)],
        "pair_fields": C.BEHAVIORS,
        "source_questionnaire_representation": "OLD binary SOURCE labels only",
        "held_questionnaires_raw_neural_API_private_reads": 0,
        "cached_record_not_immediate_action": True,
        "window_span_median_days": [
            float(
                np.median(
                    np.diff(
                        metadata["times"][z["held"]].reshape(120, 2, 18)[:, w, [0, -1]],
                        axis=1,
                    )
                )
                / 86400
            )
            for w in (0, 1)
        ],
        "original_scope_complete": False,
    }
    R.write(out / "input_quality.json", quality)
    R.seal(
        out,
        "prepared_seal.json",
        ["observations.npz", "example_record.json", "input_quality.json"],
    )
    print(json.dumps(quality))


def array_rule(rule):
    """Convert all immutable SOURCE parameters into JSON-compatible records."""
    return {k: v.tolist() for k, v in rule.items()}


def run(out):
    """Construct two disjoint calibrations and one cold reference with no grid."""
    cfg = check(out)["config"]
    z, start, rows, artifacts, rules, recon = (
        R.load(out / "observations.npz"),
        time.monotonic(),
        [],
        {},
        {},
        [],
    )
    for train_bank in (0, 1):
        train, test = z["bank"] == train_bank, z["bank"] != train_bank
        rule = C.compile_rule(
            z["source_states"][train],
            z["source_outputs"][train],
            z["source_labels"][:, train],
        )
        rules[f"bank{train_bank}"] = array_rule(rule)
        f = C.fields(rule, z["source_states"][test], z["source_outputs"][test])
        score = C.score(rule, f)
        y = z["source_labels"][0, test]
        for part in C.PARTS:
            artifacts[f"bank{train_bank}/{part}"] = score[part]
            for view in (0, 1):
                for assignment in (0, 1):
                    for q, trait in enumerate(TRAITS):
                        x = score[part][:, view, assignment, q]
                        cov = float(
                            np.mean((x - x.mean()) * (y[:, q] - y[:, q].mean()))
                        )
                        rows.append(
                            {
                                "calibration_bank": train_bank,
                                "evaluation_bank": 1 - train_bank,
                                "view": view,
                                "calibration_assignment": ("actual", "old_permuted")[
                                    assignment
                                ],
                                "part": part,
                                "trait": trait,
                                "n_authors": len(y),
                                "pearson_actual_binary_criterion": C.correlation(
                                    x, y[:, q]
                                ),
                                "signed_covariance_actual": cov,
                            }
                        )
        recon.append(
            float(
                np.max(abs(score["joint"] - score["opportunity"] - score["response"]))
            )
        )
    pooled = C.compile_rule(z["source_states"], z["source_outputs"], z["source_labels"])
    rules["pooled"] = array_rule(pooled)
    measured = C.fields(pooled, z["held_states"], z["held_outputs"])
    held_scores = C.score(pooled, measured)
    donor = z["donors"]
    if len(np.unique(donor)) != 120 or np.any(donor == np.arange(120)):
        raise ValueError("Matched donor bijection incomplete")
    boot = np.random.default_rng(cfg["seed"]).integers(0, 120, (cfg["bootstrap"], 120))
    held = {}
    for part in C.PARTS:
        early, late = measured[part][:, 0], measured[part][:, 1]
        own = np.sum((early - late) ** 2, 1)
        foreign = np.sum((early - late[donor]) ** 2, 1)
        gain = foreign - own
        held[part] = {
            "matched_distance_advantage": float(gain.mean()),
            "descriptive_CI": np.quantile(gain[boot].mean(1), [0.025, 0.975]).tolist(),
            "positive_people": int(np.sum(gain > 0)),
            "actual_field_units_not_accuracy_or_personality": True,
            "score_continuity_actual_criteria": [
                C.correlation(
                    held_scores[part][:, 0, 0, q], held_scores[part][:, 1, 0, q]
                )
                for q in range(5)
            ],
        }
        artifacts[f"held/{part}"] = measured[part]
        artifacts[f"held/{part}_scores"] = held_scores[part]
    artifacts["held/support"] = measured["support"]
    recon.append(
        float(
            np.max(
                abs(
                    held_scores["joint"]
                    - held_scores["opportunity"]
                    - held_scores["response"]
                )
            )
        )
    )
    elapsed = time.monotonic() - start
    if elapsed > cfg["compute_seconds_cap"]:
        raise TimeoutError("No automatic renewed compute budget")
    R.write(out / "source_rules.json", rules)
    R.save(out / "measured_fields.npz", artifacts)
    with (out / "criterion_contacts.csv").open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    means = {
        part: {
            view: [
                C.correlation(
                    artifacts[f"bank{b}/{part}"][:, view, 0, q],
                    z["source_labels"][0, z["bank"] != b, q],
                )
                for b in (0, 1)
                for q in range(5)
            ]
            for view in (0, 1)
        }
        for part in C.PARTS
    }
    summary = {
        "compute_seconds": elapsed,
        "held_field_alignment": held,
        "source_raw_actual_r_by_part_view_bank_ENACO": means,
        "score_reconstruction_error": max(recon),
        "held_address_support_counts": np.bincount(
            measured["support"].sum(-1).ravel(), minlength=9
        ).tolist(),
        "source_author_crossfit_not_new_held_questionnaires": True,
        "original_scope_complete": False,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "computed_seal.json",
        [
            "source_rules.json",
            "measured_fields.npz",
            "criterion_contacts.csv",
            "summary.json",
        ],
    )
    print(json.dumps(summary))


def verify(out):
    """Read-only alternate moment sums and cold scoring,not another science run."""
    check(out, ("computed_seal.json",))
    z = R.load(out / "observations.npz")
    rule = {
        k: np.array(v) for k, v in R.read(out / "source_rules.json")["pooled"].items()
    }
    frozen = R.load(out / "measured_fields.npz")
    errors = []
    for u in range(120):
        for w in range(2):
            joint = np.zeros((8, 3))
            mass = np.zeros(8)
            for a, b in zip(
                z["held_states"][u, w], z["held_outputs"][u, w], strict=True
            ):
                mass[a] += 1 / 17
                joint[a] += (b - rule["mean"]) / (17 * rule["scale"])
            opportunity = mass[:, None] * rule["conditional_mean"]
            j0 = rule["address_mass"][:, None] * rule["conditional_mean"]
            fields = (joint - j0, opportunity - j0, joint - opportunity)
            for part, value in zip(C.PARTS, fields, strict=True):
                literal = (value / np.sqrt(rule["address_mass"][:, None])).ravel()
                errors.append(
                    float(np.max(abs(literal - frozen[f"held/{part}"][u, w])))
                )
    cold = C.cold(rule, R.read(out / "example_record.json"))
    for part in C.PARTS:
        errors.append(
            float(
                np.max(abs(cold["fields"][part][0, 0] - frozen[f"held/{part}"][0, 0]))
            )
        )
    # Signed criterion covariances,not r,obey the same retained-part account.
    with (out / "criterion_contacts.csv").open() as f:
        rows = list(csv.DictReader(f))
    keys = {
        (r["calibration_bank"], r["view"], r["calibration_assignment"], r["trait"])
        for r in rows
    }
    for key in keys:
        parts = {
            r["part"]: float(r["signed_covariance_actual"])
            for r in rows
            if (
                r["calibration_bank"],
                r["view"],
                r["calibration_assignment"],
                r["trait"],
            )
            == key
        }
        errors.append(abs(parts["joint"] - parts["opportunity"] - parts["response"]))
    if max(errors) > 1e-10:
        raise ArithmeticError("Alternate measured-field construction disagrees")
    verdict = {
        "status": "PASS",
        "literal_and_cold_max_error": max(errors),
        "same_agent_not_blind": True,
        "original_scope_complete": False,
    }
    R.write(out / "verification.json", verdict)
    R.seal(out, "verification_seal.json", ["verification.json"])
    print(json.dumps(verdict))


def main():
    """Run exclusive finite stages without replacing existing outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "check"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.out)
    elif args.stage == "run":
        run(args.out)
        verify(args.out)
    else:
        check(args.out, ("computed_seal.json", "verification_seal.json"))
        print("PASS")


if __name__ == "__main__":
    main()
