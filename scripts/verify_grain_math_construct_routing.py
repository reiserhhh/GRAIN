#!/usr/bin/env python3
"""Verify HM46 native readouts and finite effects by separate scalar arithmetic."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_construct_routing import OUT, check, write


def verify(out):
    """Recompute all scores, cube effects, matrix entries and actual carrier changes."""
    manifest = check(out)
    hashes = read_json(out / "03/artifact_hashes.json")
    for name, expected in hashes.items():
        if digest(out / name) != expected:
            raise ValueError("Changed actual artifact " + name)
    summary = read_json(out / "03/summary.json")
    status = read_json(out / "run_status.json")
    if (
        status["status"] != "COLLECTED"
        or status["in_flight"] is not None
        or status["forward_calls"] != 1302
    ):
        raise ValueError("Incomplete actual acquisition")
    design = read_json(out / "00/design.json")
    targets = manifest["spec"]["targets"]
    points = tuple(itertools.product(range(2), repeat=3))
    scores, arrays = {}, {}
    errors = []
    for row in design:
        path = out / f"01/units/unit_{row['unit']:04d}.json"
        meta = read_json(path)
        if meta["unit"] != row["unit"] or meta["role"] != row["role"]:
            raise ValueError("Wrong acquired material identity")
        native = meta["native"]
        z = np.asarray(native["candidate_logits"], float)
        p = np.exp(z - z.max())
        p /= p.sum()
        value = sum(float(p[j]) * j / 9 for j in range(10))
        errors.append(abs(value - native["conditional_score"]))
        errors.append(
            abs(float(np.exp(native["candidate_log_mass"])) - native["candidate_mass"])
        )
        key = tuple(
            row[k]
            for k in (
                "role",
                "wording",
                "content",
                "query",
                "polarity",
                "other",
                "imagined",
            )
        )
        if key in scores:
            raise ValueError("Duplicate actual cube cell")
        scores[key] = value
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as a:
            arrays[row["unit"]] = {k: a[k].copy() for k in a.files}
    with (out / "03/cube_coefficients.csv").open() as stream:
        coefficients = list(csv.DictReader(stream))
    with (out / "03/response_matrices.csv").open() as stream:
        matrices = list(csv.DictReader(stream))
    axes = ("polarity", "other", "imagined")
    for row in coefficients:
        terms = set(row["term"].split("*"))
        mask = tuple(int(a in terms) for a in axes)
        base = (row["role"], int(row["wording"]), row["content"], row["query"])
        value = (
            sum(
                (-1) ** sum(a * b for a, b in zip(mask, bits, strict=True))
                * scores[base + bits]
                for bits in points
            )
            / 8
        )
        errors.append(abs(value - float(row["value"])))
    for row in matrices:
        base = (row["role"], int(row["wording"]), row["content"], row["query"])
        o, a = int(row["other"]), int(row["imagined"])
        value = scores[base + (0, o, a)] - scores[base + (1, o, a)]
        errors.append(abs(value - float(row["polarity_delta"])))
    carrier_error, random_norm_error, noop_error = [], [], []
    for path in (out / "02/patches").glob("*.json"):
        if path.name.startswith("._"):
            continue
        meta = read_json(path)
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as a:
            if path.name.startswith("noop"):
                carrier_error.append(
                    float(np.max(abs(a["property"] - arrays[meta["unit"]]["property"])))
                )
                noop_error.append(
                    abs(
                        meta["native"]["conditional_score"]
                        - meta["receiver_native"]["conditional_score"]
                    )
                )
                continue
            receiver, donor = arrays[meta["receiver_unit"]], arrays[meta["donor_unit"]]
            if meta["arm"] != "random_property":
                key = meta["arm"]
                carrier_error.append(float(np.max(abs(a[key] - donor[key]))))
            else:
                actual = float(
                    np.linalg.norm(
                        a["property"].astype(float) - receiver["property"].astype(float)
                    )
                )
                intended = float(
                    np.linalg.norm(donor["property"] - receiver["property"])
                )
                random_norm_error.append(
                    abs(actual - intended) / intended if intended else actual
                )
    if (
        len(scores) != 1152
        or len(coefficients) != 1152
        or len(matrices) != 576
        or len(noop_error) != 6
    ):
        raise ValueError("Incomplete finite analysis")
    if (
        max(errors) > 1e-12
        or max(carrier_error) != 0
        or max(noop_error) > 1e-6
        or max(random_norm_error) > 1e-5
    ):
        raise ValueError("Arithmetic, address, sham or norm-control mismatch")
    result = {
        "status": "VERIFIED_FINITE_HM46_NOT_GLOBAL",
        "native_scores_checked": len(scores),
        "cube_coefficients_checked": len(coefficients),
        "response_matrix_entries_checked": len(matrices),
        "artifact_hashes_checked": len(hashes),
        "max_scalar_arithmetic_error": max(errors),
        "maximum_actual_carrier_error": max(carrier_error),
        "maximum_noop_score_error": max(noop_error),
        "maximum_random_relative_norm_error": max(random_norm_error),
        "source_sealed_before_application": len(
            read_json(out / "01/source_barrier.json")["files"]
        )
        == 1152,
        "same_agent_alternate_arithmetic_not_independent_neural_replication": True,
        "no_new_model_calls": True,
        "scientific_status": summary["status"],
        "targets": targets,
    }
    write(out / "verification.json", result)
    print(result)


def main():
    """Validate one actual completed packet, without changing its scientific results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    verify(args.out)


if __name__ == "__main__":
    main()
