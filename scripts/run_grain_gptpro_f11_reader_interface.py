#!/usr/bin/env python3
"""Join actual F08 readers on fixed materials; descriptive method contrasts only."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def correlation(a, b):
    """Return null for degenerate vectors rather than fabricating a correlation."""
    return (
        float(np.corrcoef(a, b)[0, 1])
        if np.std(a) > 1e-12 and np.std(b) > 1e-12
        else None
    )


def crossed_decomposition(matrix):
    """Balanced two-reader fixed-effects decomposition, not latent trait variance."""
    y = np.asarray(matrix, dtype=float)
    grand = y.mean()
    material, reader = y.mean(axis=1) - grand, y.mean(axis=0) - grand
    residual = y - grand - material[:, None] - reader[None, :]
    terms = np.array([np.mean(material**2), np.mean(reader**2), np.mean(residual**2)])
    np.testing.assert_allclose(terms.sum(), np.mean((y - grand) ** 2), atol=1e-12)
    return terms


def main():
    """Measure agreement, matched qualification effects and wording-half stability."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexical", required=True, type=Path)
    parser.add_argument("--nli", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = {"lexical": args.lexical, "nli": args.nli}
    banks = {name: list(csv.DictReader(path.open())) for name, path in inputs.items()}
    indexes = {
        name: {r["material_id"]: r for r in rows} for name, rows in banks.items()
    }
    if any(len(x) != 96 for x in indexes.values()) or set(indexes["lexical"]) != set(
        indexes["nli"]
    ):
        raise ValueError("require identical 96 unique materials")
    metadata = (
        "role",
        "family",
        "pole",
        "subject_status",
        "qualification",
        "variant",
        "text",
    )
    long = []
    for material_id in sorted(indexes["lexical"]):
        left, right = indexes["lexical"][material_id], indexes["nli"][material_id]
        if any(left[k] != right[k] for k in metadata):
            raise ValueError("material content or role mismatch")
        for reader in inputs:
            r = indexes[reader][material_id]
            p, q = float(r["pole_score"]), float(r["qualification_score"])
            for target, value, truth in (
                ("pole", p, float(r["pole"])),
                ("qualification", q, float(r["qualification"])),
                (
                    "signed_qualified",
                    q * (2 * p - 1),
                    float(r["qualification"]) * (2 * float(r["pole"]) - 1),
                ),
            ):
                long.append(
                    {
                        "material_id": material_id,
                        "role": r["role"],
                        "pole": int(r["pole"]),
                        "subject_status": r["subject_status"],
                        "variant": int(r["variant"]),
                        "reader": reader,
                        "target": target,
                        "score": value,
                        "designed_target": truth,
                    }
                )
    agreement, halves, decomposition, signatures = [], [], [], []
    for role in ("source", "validation", "application"):
        for target in ("pole", "qualification", "signed_qualified"):
            selected = [r for r in long if r["role"] == role and r["target"] == target]
            by_reader = {
                reader: [r for r in selected if r["reader"] == reader]
                for reader in inputs
            }
            a, b = [
                np.array([r["score"] for r in by_reader[reader]]) for reader in inputs
            ]
            truth = np.array([r["designed_target"] for r in by_reader["lexical"]])
            agreement.append(
                {
                    "role": role,
                    "target": target,
                    "n_materials": len(a),
                    "reader_r": correlation(a, b),
                    "reader_rmse": float(np.sqrt(np.mean((a - b) ** 2))),
                    "lexical_mse": float(np.mean((a - truth) ** 2)),
                    "nli_mse": float(np.mean((b - truth) ** 2)),
                    "error_r": correlation(a - truth, b - truth),
                }
            )
            parts = crossed_decomposition(np.column_stack([a, b]))
            decomposition.append(
                {
                    "role": role,
                    "target": target,
                    "material_variance": parts[0],
                    "reader_fixed_offset_variance": parts[1],
                    "material_reader_interaction_variance": parts[2],
                    "total_variance": sum(parts),
                }
            )
            for reader, rows in by_reader.items():
                keys = sorted({(r["pole"], r["subject_status"]) for r in rows})
                even, odd = [], []
                for pole, status in keys:
                    group = [
                        r
                        for r in rows
                        if r["pole"] == pole and r["subject_status"] == status
                    ]
                    even.append(
                        np.mean([r["score"] for r in group if r["variant"] % 2 == 0])
                    )
                    odd.append(
                        np.mean([r["score"] for r in group if r["variant"] % 2 == 1])
                    )
                halves.append(
                    {
                        "role": role,
                        "target": target,
                        "reader": reader,
                        "n_designed_cells": len(keys),
                        "wording_half_r": correlation(even, odd),
                        "wording_half_rmse": float(
                            np.sqrt(np.mean((np.array(even) - odd) ** 2))
                        ),
                    }
                )
                lookup = {
                    (r["pole"], r["subject_status"], r["variant"]): r for r in rows
                }
                for pole in (0, 1):
                    for variant in range(4):
                        baseline = lookup[pole, "actual", variant]
                        for status in ("quoted", "hypothetical", "negated"):
                            # F08 uses status name 'negated' for a correction affirming the designed pole.
                            if (pole, status, variant) not in lookup:
                                raise ValueError("missing prescribed matched edit")
                            changed = lookup[pole, status, variant]
                            signatures.append(
                                {
                                    "role": role,
                                    "target": target,
                                    "reader": reader,
                                    "pole": pole,
                                    "variant": variant,
                                    "edit": "actual_to_" + status,
                                    "observed_delta": changed["score"]
                                    - baseline["score"],
                                    "designed_delta": changed["designed_target"]
                                    - baseline["designed_target"],
                                }
                            )
    for filename, rows in (
        ("paired_scores.csv", long),
        ("agreement.csv", agreement),
        ("wording_halves.csv", halves),
        ("fixed_method_decomposition.csv", decomposition),
        ("edit_signatures.csv", signatures),
    ):
        with (args.output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "status": "ACTUAL_TWO_READER_INTERFACE_COMPUTED",
        "materials": 96,
        "score_rows": len(long),
        "edit_signatures": len(signatures),
        "readers": ["lexical", "nli"],
        "missing_reader": "GRAIN: no same-material output in approved F08 caches",
        "scope": "fixed synthetic materials; not author reliability or latent trait validity",
        "method_variance": "balanced descriptive fixed offsets and interaction; not random reader population variance",
        "hashes": {
            k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in inputs.items()
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
