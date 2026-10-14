#!/usr/bin/env python3
"""Source exchange transport with explicit component and support omissions."""

from __future__ import annotations

import numpy as np

from grain_research.construction.math_carrier_edit_transport import apply_increment
from grain_research.construction.math_target_carrier_closure import (
    carrier_scopes,
    memory_characters,
)

OPERATIONS = ("relation_other", "full_other", "relation_closure", "full_closure")


def source_increment(cube, e, n, axis, component):
    """Compile a whole source assignment exchange or only its mixed component."""
    if e not in (-1, 1) or n not in (-1, 1) or axis not in ("e", "n"):
        raise ValueError("Registered e/n source assignment required")
    if component not in ("relation", "full"):
        raise ValueError("No fitted or interpolated component is registered")
    coefficients = memory_characters(cube)
    relation = -2 * e * n * coefficients[3]
    if component == "relation":
        return relation
    return (
        relation - 2 * (e if axis == "e" else n) * coefficients[1 if axis == "e" else 2]
    )


def operation_slots(row, axis, operation):
    """Use fixed other-clause or complete causal-descendant history support."""
    if axis not in ("e", "n") or operation not in OPERATIONS:
        raise ValueError("Registered axis and operation required")
    support = operation.split("_")[1]
    return carrier_scopes(row)[axis + "_" + support]


def mask_memory(value, slots):
    """Project a memory increment onto specified time rows without a fitted mask."""
    x = np.asarray(value, dtype=float)
    if x.ndim != 4 or not np.isfinite(x).all():
        raise ValueError("Finite layer/head/time/channel memory required")
    if (
        not slots
        or len(set(slots)) != len(slots)
        or min(slots) < 0
        or max(slots) >= x.shape[2]
    ):
        raise ValueError("Nonempty distinct native time addresses required")
    result = np.zeros_like(x)
    result[:, :, slots] = x[:, :, slots]
    return result


def transported_memory(base, cube, row, axis, operation):
    """Borrow source increments, never the application counterfactual memory."""
    component = operation.split("_")[0]
    slots = operation_slots(row, axis, operation)
    increment = source_increment(cube, row["e"], row["n"], axis, component)
    return apply_increment(base, increment, slots), slots


def defect_terms(base, natural_partner, source_cube, row, axis, operation):
    """Decompose update defect; application partner is audit-only, not a donor."""
    b, target, cube = (
        np.asarray(a, dtype=float) for a in (base, natural_partner, source_cube)
    )
    if b.shape != target.shape or cube.shape != (2, 2, *b.shape):
        raise ValueError("Matched four-corner source and application memory required")
    if not all(np.isfinite(a).all() for a in (b, target, cube)):
        raise ValueError("No nonfinite memory repair")
    e, n = row["e"], row["n"]
    i, j = (e + 1) // 2, (n + 1) // 2
    other = cube[1 - i, j] if axis == "e" else cube[i, 1 - j]
    full = source_increment(cube, e, n, axis, "full")
    used = source_increment(cube, e, n, axis, operation.split("_")[0])
    slots = operation_slots(row, axis, operation)
    return {
        "context": (b - cube[i, j]) - (target - other),
        "omitted_support": mask_memory(full, slots) - full,
        "omitted_component": mask_memory(used - full, slots),
        "total": b + mask_memory(used, slots) - target,
        "natural_increment": target - b,
    }


def energy_account(terms):
    """Keep cross terms: defect energies are not independent information shares."""
    parts = ("context", "omitted_support", "omitted_component")
    vectors = [np.asarray(terms[name], float).ravel() for name in parts]
    total = np.asarray(terms["total"], float).ravel()
    natural = np.asarray(terms["natural_increment"], float).ravel()
    if len({v.shape for v in (*vectors, total, natural)}) != 1:
        raise ValueError("Aligned defect components required")
    if not all(np.isfinite(v).all() for v in (*vectors, total, natural)):
        raise ValueError("Finite defects required")
    summed = sum(vectors)
    norm = float(natural @ natural)
    result = {
        "account_max_error": float(np.max(abs(summed - total))),
        "defect_rms": float(np.sqrt(np.mean(total**2))),
        "natural_increment_rms": float(np.sqrt(np.mean(natural**2))),
        "relative_defect": float(np.linalg.norm(total) / np.sqrt(norm))
        if norm
        else None,
        "total_energy": float(total @ total),
    }
    result.update(
        {name + "_energy": float(v @ v) for name, v in zip(parts, vectors, strict=True)}
    )
    for i, first in enumerate(parts):
        for j in range(i + 1, len(parts)):
            result[first + "_x_" + parts[j]] = float(2 * vectors[i] @ vectors[j])
    return result
