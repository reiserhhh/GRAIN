#!/usr/bin/env python3
"""Recompute conditional response restrictions by independent scalar products."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_construct_routing import OUT, write


def discrepancy(products):
    """Return the raw and scale-relative discrepancy, keeping zero denominators null."""
    differences = sum((left - right) ** 2 for left, right in products)
    norm = sum(left * left + right * right for left, right in products)
    return (
        math.sqrt(differences / len(products)),
        math.sqrt(differences / norm) if norm else None,
    )


def verify(parent, out):
    """Check the sealed inputs, all 18 restrictions and their actual arithmetic."""
    manifest = read_json(out / "manifest.json")
    for name, sha in manifest["code"].items():
        if digest(ROOT / name) != sha or digest(out / "code" / name) != sha:
            raise ValueError("Changed companion code")
    results = read_json(out / "results.json")
    if digest(parent / "03/response_matrices.csv") != results["parent_matrix_sha256"]:
        raise ValueError("Changed parent response matrices")
    targets = read_json(parent / "00/manifest.json")["spec"]["targets"]
    matrices = {}
    with (parent / "03/response_matrices.csv").open() as stream:
        for row in csv.DictReader(stream):
            key = (
                row["role"],
                int(row["wording"]),
                int(row["other"]),
                int(row["imagined"]),
            )
            cell = (targets.index(row["content"]), targets.index(row["query"]))
            matrix = matrices.setdefault(key, {})
            if cell in matrix:
                raise ValueError("Duplicate response cell")
            matrix[cell] = float(row["polarity_delta"])
    expected = set(itertools.product(range(6), repeat=2))
    if len(matrices) != 16 or any(set(m) != expected for m in matrices.values()):
        raise ValueError("Missing response matrix cell")
    errors, constraints = [], 0
    for record in results["conditions"]:
        role, wording = record["role"], record["wording"]

        def matrix(other, imagined, role=role, wording=wording):
            if wording == "mean":
                return {
                    cell: sum(matrices[role, w, other, imagined][cell] for w in (0, 1))
                    / 2
                    for cell in expected
                }
            return matrices[role, wording, other, imagined]

        a = matrix(0, 0)
        b = matrix(record["other"], record["imagined"])
        pairs = tuple(itertools.combinations(range(6), 2))
        products = {
            "row_only": [
                (b[i, j] * a[i, k], b[i, k] * a[i, j])
                for i in range(6)
                for j, k in pairs
            ],
            "query_only": [
                (b[j, i] * a[k, i], b[k, i] * a[j, i])
                for i in range(6)
                for j, k in pairs
            ],
            "row_and_query": [
                (
                    b[i, j] * b[k, l] * a[i, l] * a[k, j],
                    b[i, l] * b[k, j] * a[i, j] * a[k, l],
                )
                for i, k in pairs
                for j, l in pairs
            ],
        }
        for name, values in products.items():
            raw, relative = discrepancy(values)
            reported = record[name]
            if len(values) != reported["constraints"]:
                raise ValueError("Wrong polynomial count")
            errors.append(abs(raw - reported["raw_polynomial_rms"]))
            if relative is None:
                if reported["relative_polynomial_norm"] is not None:
                    raise ValueError("Undefined normalized value was filled")
            else:
                errors.append(abs(relative - reported["relative_polynomial_norm"]))
            constraints += len(values)
        a2, b2 = sum(v * v for v in a.values()), sum(v * v for v in b.values())
        scalar = sum(a[cell] * b[cell] for cell in expected) / a2
        residual = sum((b[cell] - scalar * a[cell]) ** 2 for cell in expected)
        errors.extend(
            (
                abs(scalar - record["scalar_projection"]),
                abs(residual / b2 - record["scalar_residual_squared_share"]),
            )
        )
    if len(results["conditions"]) != 18 or max(errors) > 1e-12:
        raise ValueError("Incomplete conditions or scalar arithmetic mismatch")
    output = {
        "status": "VERIFIED_FINITE_HM46A_NOT_GLOBAL",
        "conditions": len(results["conditions"]),
        "polynomials_recomputed": constraints,
        "maximum_scalar_arithmetic_error": max(errors),
        "results_sha256": digest(out / "results.json"),
        "parent_matrix_sha256": results["parent_matrix_sha256"],
        "verifier_sha256": digest(Path(__file__).resolve()),
        "new_model_calls": 0,
        "same_agent_alternate_arithmetic_not_neural_replication": True,
    }
    write(out / "verification.json", output)
    print(output)


def main():
    """Verify one frozen companion without fitting or modifying its results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=OUT)
    parser.add_argument(
        "--out", type=Path, default=OUT.parent / "construct_routing_invariants_r2"
    )
    args = parser.parse_args()
    verify(args.parent, args.out)


if __name__ == "__main__":
    main()
