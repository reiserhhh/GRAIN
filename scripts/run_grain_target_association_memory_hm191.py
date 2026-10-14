#!/usr/bin/env python3
"""One finite source-association memory experiment, without answer fitting."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_joint_evidence as J
from grain_research.construction import math_target_association_memory as M
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_target_association_memory_hm191.json"
DEFAULT = ROOT / "results/grain_math_response_v1/target_association_memory_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_TARGET_ASSOCIATION_MEMORY_PLAN_HM191.md",
    "grain_research/construction/math_target_association_memory.py",
    "scripts/run_grain_target_association_memory_hm191.py",
    "tests/test_grain_target_association_memory.py",
    "grain_research/construction/math_joint_evidence.py",
    "scripts/run_grain_observed_family_hm170.py",
)


def prepare(out):
    """Seal the complete source and finite stream family before new calculation."""
    spec = R.read(ROOT / CONFIG)
    parent = ROOT / spec["parent"]
    R.check(parent, ("scored_seal.json",))
    bundle = R.read(parent / "source_bundle.json")
    p, y = J.validate_bundle(bundle)
    if p.shape != (spec["source_authors"], spec["event_cells"]) or y.shape[1] != 5:
        raise ValueError("Changed complete SOURCE target/observation support")
    M.law_quotient(p, spec["source_events_per_author"])
    names = (
        "manifest.json",
        "source_bundle.json",
        "prepared_seal.json",
        "scored_seal.json",
        "cold_input.jsonl",
        "cold_output.jsonl",
    )
    inputs = {
        str((parent / name).relative_to(ROOT)): R.digest(parent / name)
        for name in names
    }
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "config": spec,
            "code": {name: R.digest(ROOT / name) for name in SCIENCE},
            "inputs": inputs,
            "source_working_law": bundle["observation_law"],
            "witnesses": "positive_counterfactual_source_weights_NOT_realized_histories",
            "original_scope_complete": False,
        },
    )
    R.archive(out / "observations.npz", probabilities=p, targets=y)
    R.write(
        out / "input_quality.json",
        {
            "source_authors": len(p),
            "event_cells": p.shape[1],
            "source_events_per_author": spec["source_events_per_author"],
            "targets": spec["targets"],
            "profile_count": len(np.unique(y, axis=0)),
            "no_source_or_application_refit": True,
            "new_neural_calls": 0,
            "application_label_reads": 0,
            "historical_representation_overlap_and_working_iid_retained": True,
        },
    )
    R.seal(out, "prepared_seal.json", ("observations.npz", "input_quality.json"))
    print(json.dumps(R.read(out / "input_quality.json")))


def profile_state(p, y, w):
    """Compute full joint-profile and one-event conditional profile probabilities."""
    profiles, ids = np.unique(y, axis=0, return_inverse=True)
    c = np.eye(len(profiles))[ids]
    mass = w @ c
    pi = w @ p
    joint = (c * w[:, None]).T @ p
    return mass, joint / pi, (p * w[:, None]).T @ p / pi[:, None]


def compute(out):
    """Discriminate state hypotheses and use the exact association update interface."""
    manifest = R.check(out)
    spec, z = manifest["config"], R.prepared(out)
    p, y = z["probabilities"], z["targets"]
    start = time.perf_counter()
    rows, weight_plus, weight_minus, spaces = [], [], [], []
    for order in spec["constraint_orders"]:
        b, _ = M.constraints(p, y, order)
        basis, rank, tolerance = M.column_basis(b)
        spaces.append(
            {
                "order": order,
                "columns": b.shape[1],
                "rank": rank,
                "machine_tolerance": tolerance,
            }
        )
        cells = (
            [(a,) for a in range(p.shape[1])]
            if order == 0
            else [(a, b) for a in range(p.shape[1]) for b in range(a, p.shape[1])]
        )
        for q, target in enumerate(spec["targets"]):
            for word in cells:
                r = M.witness(
                    p,
                    y,
                    basis,
                    q,
                    list(word),
                    spec["witness_mass_fraction"],
                    spec["projection_relative_tolerance"],
                )
                wp, wm = r["plus"], r["minus"]
                s0, s1 = M.state(p, y, wp), M.state(p, y, wm)
                c0, post0, next0 = profile_state(p, y, wp)
                c1, post1, next1 = profile_state(p, y, wm)
                row = {
                    "constraint_order": order,
                    "target": target,
                    "cell_a": word[0],
                    "cell_b": word[1] if order else -1,
                    "witness_present": r["present"],
                    "relative_residual": r["relative_residual"],
                    "update_gap": r["gap"],
                    "min_positive_weight": float(min(wp.min(), wm.min())),
                    "constraint_error": float(np.max(np.abs(b.T @ (wp - wm)))),
                    "weight_sum_error": float(
                        max(abs(wp.sum() - 1), abs(wm.sum() - 1))
                    ),
                    "current_profile_error": float(np.max(np.abs(c0 - c1))),
                    "current_target_error": float(np.max(np.abs(s0["mu"] - s1["mu"]))),
                    "next_event_law_error": float(np.max(np.abs(s0["pi"] - s1["pi"]))),
                    "immediate_profile_error": float(np.max(np.abs(post0 - post1))),
                    "immediate_target_error": float(
                        np.max(np.abs(s0["after"] - s1["after"]))
                    ),
                    "after_one_next_event_law_error": float(
                        np.max(np.abs(next0 - next1))
                    ),
                }
                if row["constraint_error"] > spec["certificate_tolerance"]:
                    raise ValueError(
                        "State-matching certificate failed; no tolerance rescue"
                    )
                rows.append(row)
                weight_plus.append(wp)
                weight_minus.append(wm)
    R.table(out / "state_matched_witnesses.csv", rows)
    R.archive(
        out / "witness_weights.npz",
        plus=np.array(weight_plus),
        minus=np.array(weight_minus),
    )
    quotient = M.law_quotient(p, spec["source_events_per_author"])
    R.archive(
        out / "law_quotient.npz",
        count_vectors=quotient["count_vectors"],
        class_ids=quotient["class_ids"],
        sizes=quotient["sizes"],
    )
    R.write(
        out / "law_quotient.json",
        {
            "source_authors": len(p),
            "identical_law_classes": len(quotient["sizes"]),
            "class_sizes": quotient["sizes"].tolist(),
            "closed_observation_algebra_dimension": quotient[
                "closed_observation_algebra_dimension"
            ],
            "inference": "finite_separating_polynomial_algebra_not_numerical_rank_search",
            "not_psychological_dimension_or_general_nonlinear_memory_lower_bound": True,
        },
    )
    parent = ROOT / spec["parent"]
    with (parent / "cold_output.jsonl").open() as stream:
        old = {
            r["author_key"]: np.asarray(r["probabilities"], float)[2]
            for r in map(json.loads, stream)
        }
    trajectory, after_tables, endpoint_rows, max_error = [], [], [], 0.0
    with (parent / "cold_input.jsonl").open() as stream:
        for record in map(json.loads, stream):
            w, steps = M.replay(p, y, record["events"])
            key = record["author_key"]
            endpoint = w @ y
            endpoint_rows.append(
                {
                    "author_key": key,
                    "events": len(steps),
                    **{
                        target: float(endpoint[q])
                        for q, target in enumerate(spec["targets"])
                    },
                }
            )
            max_error = max(max_error, float(np.max(np.abs(endpoint - old[key]))))
            for step, (identity, cell, before, after) in enumerate(steps):
                after_tables.append(after["after"])
                for q, target in enumerate(spec["targets"]):
                    actual = float(after["mu"][q] - before["mu"][q])
                    predicted = float(before["gamma"][q, cell] / before["pi"][cell])
                    trajectory.append(
                        {
                            "author_key": key,
                            "step": step + 1,
                            "event_id": identity,
                            "cell": cell,
                            "target": target,
                            "before": float(before["mu"][q]),
                            "after": float(after["mu"][q]),
                            "delta": actual,
                            "association_increment": predicted,
                            "update_identity_error": abs(actual - predicted),
                            "new_event_probability": float(before["pi"][cell]),
                        }
                    )
    if len(endpoint_rows) != spec["cold_families"]:
        raise ValueError("Changed complete finite stream family")
    R.table(out / "association_trajectories.csv", trajectory)
    R.table(out / "endpoint_replay.csv", endpoint_rows)
    R.archive(
        out / "next_update_tables.npz", possible_next_scores=np.array(after_tables)
    )
    summaries = []
    for order in spec["constraint_orders"]:
        for target in spec["targets"]:
            rr = [
                r
                for r in rows
                if r["constraint_order"] == order and r["target"] == target
            ]
            summaries.append(
                {
                    "constraint_order": order,
                    "target": target,
                    "cases": len(rr),
                    "present": sum(r["witness_present"] for r in rr),
                    "update_gap_min": min(r["update_gap"] for r in rr),
                    "update_gap_max": max(r["update_gap"] for r in rr),
                }
            )
    R.table(out / "witness_summary.csv", summaries)
    summary = {
        "question": "Does state-matched target/evidence association change future target updates?",
        "constraint_spaces": spaces,
        "witnesses": summaries,
        "full_profile_and_next_law_matching_error": max(
            max(r["current_profile_error"], r["next_event_law_error"]) for r in rows
        ),
        "order1_all_immediate_matching_error": max(
            max(r["immediate_profile_error"], r["after_one_next_event_law_error"])
            for r in rows
            if r["constraint_order"] == 1
        ),
        "minimum_witness_weight": min(r["min_positive_weight"] for r in rows),
        "cold_families": len(endpoint_rows),
        "distinct_stream_events": len(after_tables),
        "target_update_accounts": len(trajectory),
        "association_update_max_error": max(
            r["update_identity_error"] for r in trajectory
        ),
        "parent_endpoint_max_error": max_error,
        "unique_source_laws": len(quotient["sizes"]),
        "compute_seconds": time.perf_counter() - start,
        "counterfactual_weights_NOT_real_histories_or_new_authors": True,
        "conditional_iid_working_law_NOT_native_LLM_mechanism": True,
        "new_neural_calls": 0,
        "new_answer_fit": False,
        "original_scope_complete": False,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "computed_seal.json",
        (
            "state_matched_witnesses.csv",
            "witness_weights.npz",
            "law_quotient.npz",
            "law_quotient.json",
            "association_trajectories.csv",
            "endpoint_replay.csv",
            "next_update_tables.npz",
            "witness_summary.csv",
            "summary.json",
        ),
    )
    print(json.dumps(summary))


def verify(out):
    """One alternate direct weighted-sum audit; do not optimize numerical precision."""
    manifest = R.check(out, ("computed_seal.json",))
    z = R.prepared(out)
    p, y = z["probabilities"], z["targets"]
    with np.load(out / "witness_weights.npz", allow_pickle=False) as z:
        wp, wm = z["plus"], z["minus"]
    import csv

    with (out / "state_matched_witnesses.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    delta_error = 0.0
    for i, row in enumerate(rows):
        q = manifest["config"]["targets"].index(row["target"])
        cells = [int(row["cell_a"])]
        if int(row["constraint_order"]) == 1:
            cells.append(int(row["cell_b"]))
        numerator, denominator = [], []
        for w in (wp[i], wm[i]):
            terms = [
                float(w[u]) * float(np.prod([p[u, a] for a in cells]))
                for u in range(len(p))
            ]
            denominator.append(sum(terms))
            numerator.append(sum(terms[u] * float(y[u, q]) for u in range(len(p))))
        gap = numerator[0] / denominator[0] - numerator[1] / denominator[1]
        delta_error = max(delta_error, abs(gap - float(row["update_gap"])))
        if row["witness_present"] == "True" and gap <= 0:
            raise ValueError("Positive construction did not give its declared witness")
    summary = R.read(out / "summary.json")
    tolerance = manifest["config"]["certificate_tolerance"]
    errors = {
        "direct_sum_witness_gap": delta_error,
        "profile_and_event_matching": summary[
            "full_profile_and_next_law_matching_error"
        ],
        "order1_immediate_matching": summary["order1_all_immediate_matching_error"],
        "update_identity": summary["association_update_max_error"],
        "parent_endpoint_replay": summary["parent_endpoint_max_error"],
    }
    if max(errors.values()) > tolerance or summary["minimum_witness_weight"] <= 0:
        raise ValueError(
            "Declared finite construction failed; retain result without rescue"
        )
    audit = {
        "witnesses_recomputed": len(rows),
        "errors": errors,
        "same_agent_alternate_not_blind_audit": True,
        "passes_fixed_tolerance": True,
        "original_scope_complete": False,
    }
    R.write(out / "alternate_audit.json", audit)
    R.seal(out, "verified_seal.json", ("alternate_audit.json",))
    print(json.dumps(audit))


def main():
    """Run only the requested exclusive stage; existing results cannot be overwritten."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "compute", "verify"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"prepare": prepare, "compute": compute, "verify": verify}[args.stage](args.out)


if __name__ == "__main__":
    main()
