#!/usr/bin/env python3
"""Alternate scalar arithmetic and input/provenance verification for actual HM47."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_evaluative_routing import OUT, check, load_arrays, write

LEVELS = ("none", "positive", "negative")


def csv_rows(path):
    """Read a full result table without treating its existence as acceptance."""
    with path.open() as stream:
        return list(csv.DictReader(stream))


def native_score(native):
    """Recompute the ten-digit expectation and full-vocabulary mass from stored logits."""
    z = native["candidate_logits"]
    offset = max(z)
    mass = sum(math.exp(v - offset) for v in z)
    probabilities = [math.exp(v - offset) / mass for v in z]
    return (
        sum(p * j / 9 for j, p in enumerate(probabilities)),
        math.exp(offset + math.log(mass) - native["full_vocab_log_normalizer"]),
    )


def verify(out):
    """Rebuild all contrasts from actual native units, checking cached and fresh evidence."""
    manifest = check(out)
    status = read_json(out / "run_status.json")
    if (
        status["status"] != "COLLECTED"
        or status["in_flight"] is not None
        or status["forward_calls"] != 864
        or status["attempted_forward_calls"] != 864
        or status["reused_units"] != 432
        or status["completed_units"] != 1296
    ):
        raise ValueError("Actual finite collection not complete")
    hashes = read_json(out / "03/artifact_hashes.json")
    for name, expected in hashes.items():
        if digest(out / name) != expected:
            raise ValueError("Changed actual result " + name)
    parent = ROOT / manifest["spec"]["parent"]
    rows = read_json(out / "00/design.json")
    scores, groups, arrays = {}, {}, {}
    errors, reuse_errors = [], []
    for row in rows:
        meta, capture = load_arrays(out / f"01/units/unit_{row['unit']:04d}.json")
        if any(meta[k] != row[k] for k in ("unit", "role", "reused", "hm46_unit")):
            raise ValueError("Wrong actual unit identity")
        value, mass = native_score(meta["native"])
        errors.extend(
            (
                abs(value - meta["native"]["conditional_score"]),
                abs(mass - meta["native"]["candidate_mass"]),
            )
        )
        scores[row["unit"]] = value
        arrays[row["unit"]] = capture
        base = tuple(
            row[k] for k in ("role", "wording", "other", "imagined", "content", "query")
        )
        cell = (row["polarity"], LEVELS.index(row["evaluation"]))
        table = groups.setdefault(base, {})
        if cell in table:
            raise ValueError("Duplicate finite response")
        table[cell] = value
        if row["reused"]:
            old, old_arrays = load_arrays(
                parent / f"01/units/unit_{row['hm46_unit']:04d}.json"
            )
            if old["native"] != meta["native"]:
                raise ValueError("Cached score changed")
            for key, values in old_arrays.items():
                reuse_errors.append(float(np.max(abs(values - capture[key]))))
    if (
        len(scores) != 1296
        or len(groups) != 216
        or any(
            set(t) != set(itertools.product(range(2), range(3)))
            for t in groups.values()
        )
    ):
        raise ValueError("Incomplete finite factorial scope")
    readouts = csv_rows(out / "03/readouts.csv")
    coefficients = csv_rows(out / "03/coefficients.csv")
    effects = csv_rows(out / "03/effects.csv")
    matrices = csv_rows(out / "03/response_matrices.csv")
    for row in readouts:
        errors.append(abs(float(row["score"]) - scores[int(row["unit"])]))
    behaviour_vectors = (
        (1 / math.sqrt(2), 1 / math.sqrt(2)),
        (1 / math.sqrt(2), -1 / math.sqrt(2)),
    )
    evaluation_vectors = (
        (1 / math.sqrt(3),) * 3,
        (0, 1 / math.sqrt(2), -1 / math.sqrt(2)),
        (-2 / math.sqrt(6), 1 / math.sqrt(6), 1 / math.sqrt(6)),
    )

    def key(row):
        return (
            row["role"],
            int(row["wording"]),
            int(row["other"]),
            int(row["imagined"]),
            row["content"],
            row["query"],
        )

    for row in coefficients:
        table = groups[key(row)]
        b, e = int(row["behaviour_component"]), int(row["appraisal_component"])
        value = sum(
            behaviour_vectors[b][p] * evaluation_vectors[e][j] * table[p, j]
            for p, j in itertools.product(range(2), range(3))
        )
        errors.append(abs(value - float(row["value"])))
    for row in effects:
        t = groups[key(row)]
        name, level = row["effect"], row["level"]
        if name == "behaviour":
            e = LEVELS.index(level)
            value = t[0, e] - t[1, e]
        elif name == "evaluation":
            p = int(level)
            value = t[p, 1] - t[p, 2]
        elif name == "presence":
            p = int(level)
            value = (t[p, 1] + t[p, 2]) / 2 - t[p, 0]
        elif name == "interaction":
            value = t[0, 1] - t[1, 1] - t[0, 2] + t[1, 2]
        else:
            raise ValueError("Unknown contrast")
        errors.append(abs(value - float(row["value"])))
    for row in matrices:
        t = groups[key(row)]
        e = LEVELS.index(row["evaluation"])
        errors.append(abs(t[0, e] - t[1, e] - float(row["behaviour_delta"])))
    group_summary = read_json(out / "03/group_summary.json")
    for row in group_summary:
        base = (row["role"], row["wording"], row["other"], row["imagined"])
        matrices_by_eval = {e: {} for e in LEVELS}
        evaluation, presence, interaction = {}, {}, {}
        for c, q in itertools.product(range(6), repeat=2):
            t = groups[(*base, manifest["targets"][c], manifest["targets"][q])]
            for e, label in enumerate(LEVELS):
                matrices_by_eval[label][c, q] = t[0, e] - t[1, e]
            evaluation[c, q] = sum(t[p, 1] - t[p, 2] for p in (0, 1)) / 2
            presence[c, q] = sum((t[p, 1] + t[p, 2]) / 2 - t[p, 0] for p in (0, 1)) / 2
            interaction[c, q] = t[0, 1] - t[1, 1] - t[0, 2] + t[1, 2]
            for name, values in (
                ("evaluation", evaluation),
                ("presence", presence),
                ("interaction", interaction),
            ):
                errors.append(abs(values[c, q] - row[name + "_matrix"][c][q]))
        reference = matrices_by_eval["none"]
        r2, e2 = (
            sum(v * v for v in reference.values()),
            sum(v * v for v in evaluation.values()),
        )
        errors.extend(
            (
                abs(math.sqrt(r2 / 36) - row["baseline_behaviour_rms"]),
                abs(math.sqrt(e2 / 36) - row["evaluation_rms"]),
                abs(math.sqrt(e2 / r2) - row["evaluation_to_behaviour_rms"]),
                abs(
                    math.sqrt(sum(v * v for v in interaction.values()) / 36)
                    - row["interaction_rms"]
                ),
            )
        )
        for e in ("positive", "negative"):
            current = matrices_by_eval[e]
            b2 = sum(v * v for v in current.values())
            gamma = sum(reference[k] * current[k] for k in reference) / r2
            residual = sum((current[k] - gamma * reference[k]) ** 2 for k in reference)
            errors.extend(
                (
                    abs(gamma - row[e]["projection"]),
                    abs(residual / b2 - row[e]["residual_squared_share"]),
                )
            )
    barrier = read_json(out / "01/source_barrier.json")
    for name, sha in barrier["files"].items():
        if digest(out / name) != sha:
            raise ValueError("Changed source barrier")
    if len(barrier["files"]) != 864 or any(
        read_json(out / f"01/units/unit_{r['unit']:04d}.json")["role"] != "source"
        for r in rows
        if r["role"] == "source"
    ):
        raise ValueError("Incomplete source barrier")
    if (
        (
            len(readouts),
            len(coefficients),
            len(effects),
            len(matrices),
            len(group_summary),
        )
        != (1296, 1296, 1728, 648, 6)
        or max(errors) > 1e-12
        or max(reuse_errors) != 0
    ):
        raise ValueError("Incomplete output scope or alternate scalar mismatch")
    result = {
        "status": "VERIFIED_FINITE_HM47_NOT_GLOBAL",
        "native_scores": len(scores),
        "reused_scores": status["reused_units"],
        "fresh_actual_forwards": status["forward_calls"],
        "coefficients_effects_matrix_cells": [
            len(coefficients),
            len(effects),
            len(matrices),
        ],
        "group_summaries": len(group_summary),
        "artifact_hashes": len(hashes),
        "maximum_alternate_scalar_error": max(errors),
        "maximum_reused_carrier_error": max(reuse_errors),
        "source_scores_sealed_before_application": barrier["source_scores"],
        "verifier_sha256": digest(Path(__file__).resolve()),
        "same_agent_alternate_arithmetic_not_independent_replication": True,
        "new_model_calls": 0,
        "full_original_goal_complete": False,
    }
    write(out / "verification.json", result)
    print(result)


def main():
    """Verify actual data without modifying scientific results or launching a model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    verify(args.out)


if __name__ == "__main__":
    main()
