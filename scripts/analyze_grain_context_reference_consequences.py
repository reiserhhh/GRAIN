#!/usr/bin/env python3
"""Result-after HM77 author-composition consequences, with zero new model calls."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.construction import math_context_anchored_operation as M
from grain_research.construction import math_source_ownership_operator as S
from scripts.run_grain_context_anchored_operation import (
    OUT,
    all_groups,
    check,
    digest,
    export,
    files,
    models,
    read,
    seal,
    write,
)

SUBSETS = (
    ("x", (0,)),
    ("y", (1,)),
    ("xy", (0, 1)),
    ("g", (2,)),
    ("xg", (0, 2)),
    ("yg", (1, 2)),
    ("xyg", (0, 1, 2)),
)


def contrast(coords, vectors, subset):
    """Difference of two specified valid profiles P+/-=(1+/-0.5*chi)/8."""
    coords = np.asarray(coords, int)
    vectors = S.finite_rows(vectors)
    expected = {(x, y, g) for x in (-1, 1) for y in (-1, 1) for g in (-1, 1)}
    if (
        coords.shape != (8, 3)
        or {tuple(c) for c in coords} != expected
        or len(vectors) != 8
    ):
        raise ValueError("Complete eight-cell evaluation cube required")
    if not subset or any(i not in (0, 1, 2) for i in subset):
        raise ValueError("Declared nonempty character required")
    chi = np.prod(coords[:, subset], axis=1)
    return (0.5 / 4 * chi) @ vectors


def main():
    """Separate shared transport offset from changes in relative author profiles."""
    check()
    if read(OUT / "verification.json")["status"] != "VERIFIED_FINITE_HM77":
        raise ValueError("Primary HM77 must be verified first")
    for name, sha in read(OUT / "03/output_seal.json").items():
        if digest(OUT / name) != sha:
            raise ValueError("Frozen primary drift")
    out = OUT / "05_author_consequences"
    if out.exists():
        raise FileExistsError("Author consequence companion exists; no replacement")
    fitted = models()
    groups, inputs = all_groups()
    cubes = defaultdict(dict)
    for (bank, query, focal, companion, x, y), value in groups.items():
        cubes[bank, query, focal, companion][x, y] = value
    out.mkdir()
    write(
        out / "preanalysis_manifest.json",
        {
            "result_after": True,
            "script_sha256": digest(Path(__file__)),
            "primary_summary_sha256": digest(OUT / "03/summary.json"),
            "verification_sha256": digest(OUT / "verification.json"),
            "question": "Does large absolute operation error cancel in relative author positions?",
            "profile_amplitude": 0.5,
            "profiles_per_cube": 7,
            "new_fits": 0,
            "new_model_calls": 0,
            "private_data_reads": 0,
            "simulated_compositions_not_real_author_labels": True,
        },
    )
    rows, energy_rows, max_identity, max_zero = [], [], 0.0, 0.0
    for (bank, query, focal, companion), values in sorted(cubes.items()):
        if set(values) != {(x, y) for x in (-1, 1) for y in (-1, 1)}:
            raise ValueError("Material group lacks complete composition support")
        for arm in M.METHODS:
            conditioned = arm.startswith("anchored")
            model = fitted[query, arm.split("_", 1)[1]]
            coordinates, errors, targets, even_errors, odd_errors = [], [], [], [], []
            for (x, y), pair in sorted(values.items()):
                bp, qp = pair[1]
                bm, qm = pair[-1]
                even_q = (qp + qm) / 2 - model["origin"]
                anchor_even = (
                    M.anchor(model, ((bp + bm) / 2)[None])[0] if conditioned else 0
                )
                anchor_odd = (
                    ((bp - bm) / 2 @ model["anchor_right"].T) @ model["anchor_left"].T
                    if conditioned
                    else np.zeros_like(qp)
                )
                l = S.project(model, (even_q - anchor_even)[None])[0]
                d = (qp - qm) / 2
                missing = d - S.project(model, d[None])[0]
                contamination = S.project(model, anchor_odd[None])[0]
                for g in (-1, 1):
                    b, q = pair[g]
                    target = pair[-g][1]
                    predicted = M.operate(model, b, q, conditioned=conditioned)[1]
                    coordinates.append((x, y, g))
                    errors.append(predicted - target)
                    targets.append(target)
                    even_errors.append(-2 * l)
                    odd_errors.append(2 * g * (missing + contamination))
            errors, targets = np.array(errors), np.array(targets)
            shared = errors.mean(0)
            centered = errors - shared
            shared_sq = float(shared @ shared)
            variable_sq = float(np.mean(np.sum(centered * centered, axis=1)))
            total_sq = float(np.mean(np.sum(errors * errors, axis=1)))
            max_identity = max(max_identity, abs(total_sq - shared_sq - variable_sq))
            energy_rows.append(
                {
                    "bank": bank,
                    "query": query,
                    "arm": arm,
                    "focal": focal,
                    "companion": companion,
                    "shared_sq": shared_sq,
                    "variable_sq": variable_sq,
                    "total_sq": total_sq,
                }
            )
            for name, subset in SUBSETS:
                direct = contrast(coordinates, errors, subset)
                even = contrast(coordinates, even_errors, subset)
                odd = contrast(coordinates, odd_errors, subset)
                true = contrast(coordinates, targets, subset)
                max_identity = max(
                    max_identity, float(np.max(abs(direct - even - odd)))
                )
                forbidden = even if 2 in subset else odd
                max_zero = max(max_zero, float(np.max(abs(forbidden))))
                rows.append(
                    {
                        "bank": bank,
                        "query": query,
                        "arm": arm,
                        "focal": focal,
                        "companion": companion,
                        "profile": name,
                        "profile_type": "ownership_composition"
                        if 2 in subset
                        else "content_composition",
                        "true_contrast_sq": float(true @ true),
                        "error_contrast_sq": float(direct @ direct),
                        "context_error_contrast_sq": float(even @ even),
                        "odd_error_contrast_sq": float(odd @ odd),
                    }
                )
    summaries = []
    for bank in sorted({r["bank"] for r in rows}):
        for arm in M.METHODS:
            energies = [r for r in energy_rows if r["bank"] == bank and r["arm"] == arm]
            total = np.mean([r["total_sq"] for r in energies])
            for kind in ("content_composition", "ownership_composition"):
                selected = [
                    r
                    for r in rows
                    if r["bank"] == bank
                    and r["arm"] == arm
                    and r["profile_type"] == kind
                ]
                truth = float(np.mean([r["true_contrast_sq"] for r in selected]))
                err = float(np.mean([r["error_contrast_sq"] for r in selected]))
                summaries.append(
                    {
                        "bank": bank,
                        "arm": arm,
                        "profile_type": kind,
                        "profile_pairs": len(selected),
                        "true_contrast_rms": float(np.sqrt(truth)),
                        "error_contrast_rms": float(np.sqrt(err)),
                        "relative_contrast_error": float(np.sqrt(err / truth))
                        if truth
                        else None,
                        "shared_fraction_of_absolute_error": float(
                            np.mean([r["shared_sq"] for r in energies]) / total
                        )
                        if total
                        else None,
                    }
                )
    export(out / "profile_accounts.csv", rows)
    export(out / "cube_error_energy.csv", energy_rows)
    export(out / "summary.csv", summaries)
    write(
        out / "summary.json",
        {
            "result_after": True,
            "summaries": summaries,
            "max_contrast_identity_error": max_identity,
            "max_forbidden_character_component": max_zero,
            "profiles": len(rows),
            "cubes_across_arms": len(energy_rows),
            "all_input_objects": inputs,
            "new_model_calls": 0,
            "new_fits": 0,
            "original_scope_complete": False,
        },
    )
    if max(max_identity, max_zero) > 1e-8:
        raise ValueError("Relative author account failed")
    write(out / "output_seal.json", seal(OUT, files(out)))
    print(
        {
            "status": "AUTHOR_CONSEQUENCES_VERIFIED_RESULT_AFTER",
            "profiles": len(rows),
            "max_identity_error": max_identity,
            "max_structural_zero": max_zero,
            "new_model_calls": 0,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
