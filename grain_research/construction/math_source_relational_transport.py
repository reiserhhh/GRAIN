#!/usr/bin/env python3
"""Source-only relational changes with explicit actor-slot and query alignment."""

from __future__ import annotations

import numpy as np

ARMS = (
    "exact_suffix",
    "source_full",
    "source_bound",
    "source_shared",
    "wrong_person",
    "random",
    "noop",
)


def suffix_ids(row, capture):
    """Return actual query/instruction suffix IDs, with their complete relative order."""
    p = capture["partition"]["suffix"]
    if not p or p != list(range(p[0], len(row["input_ids"]))):
        raise ValueError("Complete contiguous suffix required")
    return [row["input_ids"][i] for i in p]


def prove_transport(source_row, source_capture, app_row, app_capture):
    """Require exact same-person query IDs, retaining natural absolute-position shifts."""
    if any(source_row[k] != app_row[k] for k in ("person", "query", "content")):
        raise ValueError("Source/application person or target mismatch")
    a, b = suffix_ids(source_row, source_capture), suffix_ids(app_row, app_capture)
    if a != b:
        raise ValueError("Source/application suffix token mismatch")
    return {
        "source_positions": source_capture["partition"]["suffix"],
        "application_positions": app_capture["partition"]["suffix"],
        "suffix_ids": a,
        "relative_slots": list(range(len(a))),
    }


def actor_slot_proof(a, b):
    """Allow precisely one A/B query-name difference, not arbitrary content changes."""
    if len(a) != len(b):
        raise ValueError("Actor-slot query length mismatch")
    changed = [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]
    if len(changed) != 1:
        raise ValueError("Only one intentional actor-name slot may differ")
    return {
        "intentional_actor_slot": changed[0],
        "actor_token_ids": [a[changed[0]], b[changed[0]]],
    }


def source_sectors(delta_a, delta_b):
    """Keep both exact shared and person-oriented source sectors, without fitting."""
    a, b = np.asarray(delta_a, float), np.asarray(delta_b, float)
    if a.shape != b.shape or a.ndim != 2 or not np.isfinite([a, b]).all():
        raise ValueError("Two finite aligned source suffix changes required")
    return {"A": a, "B": b, "shared": (a + b) / 2, "bound": (a - b) / 2}


def source_delta(sectors, person, direction, arm, pair, seed):
    """Produce a frozen additive change; random norms use source only, never outputs."""
    if person not in ("A", "B") or direction not in (0, 1):
        raise ValueError("Explicit actor and binary direction required")
    sign, actor = 1 - 2 * direction, 1 if person == "A" else -1
    if arm == "source_full":
        delta = sectors[person]
    elif arm == "source_bound":
        delta = actor * sectors["bound"]
    elif arm == "source_shared":
        delta = sectors["shared"]
    elif arm == "wrong_person":
        delta = sectors["B" if person == "A" else "A"]
    elif arm == "random":
        delta = np.random.default_rng(seed + pair // 2).normal(
            size=sectors[person].shape
        )
        delta *= (
            np.linalg.norm(sectors[person], axis=1) / np.linalg.norm(delta, axis=1)
        )[:, None]
    else:
        raise ValueError("An additive source arm is required")
    return sign * np.asarray(delta, float)


def patch_values(entry, current, exact_donor, sectors, seed):
    """Replace a complete suffix; preserve all body/prefix states outside the patch."""
    p, arm = entry["partition"]["suffix"], entry["arm"]
    if arm not in ARMS or current.shape != exact_donor.shape or current.ndim != 2:
        raise ValueError("Aligned complete receiver states and fixed arm required")
    if arm == "exact_suffix":
        values = exact_donor[p].copy()
    elif arm == "noop":
        values = current[p].copy()
    else:
        delta = source_delta(
            sectors, entry["person"], entry["direction"], arm, entry["pair"], seed
        )
        if delta.shape != current[p].shape or not np.isfinite(delta).all():
            raise ValueError("Source change does not match the complete query suffix")
        values = (current[p].astype(float) + delta).astype(current.dtype)
    return p, values


def four_cells(native, shared, bound, full):
    """Keep the shared/bound output interaction despite exact vector-state additivity."""
    if not np.isfinite([native, shared, bound, full]).all():
        raise ValueError("Finite four-cell observations required")
    a, b = shared - native, bound - native
    i = full - shared - bound + native
    return {
        "y00": float(native),
        "y10": float(shared),
        "y01": float(bound),
        "y11": float(full),
        "shared_action": float(a),
        "bound_action": float(b),
        "interaction": float(i),
        "joint_action": float(full - native),
        "reconstruction_error": float(full - native - a - b - i),
    }
