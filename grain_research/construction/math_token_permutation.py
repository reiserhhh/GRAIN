"""HM13: exact token-multiset interventions with fixed receiver context."""

from __future__ import annotations

import hashlib

import numpy as np

from grain_research.construction.math_token_mean import patch_target


def token_matrix(value):
    """Require finite float32 token states without silently rounding new inputs."""
    h = np.asarray(value)
    if (
        h.dtype != np.float32
        or h.ndim != 2
        or min(h.shape) < 1
        or not np.isfinite(h).all()
    ):
        raise ValueError("finite nonempty float32 token matrix required")
    return np.ascontiguousarray(h)


def reference_and_target(current, donor, arm):
    """Reverse original states or the exact frozen HM12 centered-only assignment."""
    h, d = token_matrix(current), token_matrix(donor)
    if h.shape != d.shape:
        raise ValueError("aligned donor and receiver required")
    if arm == "receiver_reverse":
        reference = h.copy()
    elif arm == "centered_reverse":
        reference = patch_target(h, d, "centered_only")
    else:
        raise ValueError("unknown fixed permutation arm")
    return reference, reference[::-1].copy()


def row_bytes(value):
    """Return an exact multiset certificate, preserving signed zero and all bits."""
    return sorted(row.tobytes() for row in token_matrix(value))


def state_digest(value):
    """Hash the complete ordered float32 matrix, not a projected summary."""
    h = token_matrix(value)
    return hashlib.sha256(h.tobytes()).hexdigest()


def permutation_audit(current, donor, arm):
    """Replay all-dimensional moments and byte-exact row invariance separately."""
    h = token_matrix(current)
    reference, target = reference_and_target(h, donor, arm)
    reference_rows, target_rows = row_bytes(reference), row_bytes(target)
    if reference_rows != target_rows:
        raise ValueError("token multiset changed")
    a, b = reference.astype(float), target.astype(float)
    am, bm = a.mean(axis=0), b.mean(axis=0)
    ac, bc = a - am, b - bm
    # Blockwise replay checks every covariance entry with bounded memory.
    covariance_error = 0.0
    for start in range(0, a.shape[1], 256):
        difference = (
            ac[:, start : start + 256].T @ ac - bc[:, start : start + 256].T @ bc
        ) / len(a)
        covariance_error = max(covariance_error, float(np.max(np.abs(difference))))
    return {
        "tokens": len(a),
        "dimensions": a.shape[1],
        "row_multiset_exact": True,
        "multiset_sha256": hashlib.sha256(b"".join(reference_rows)).hexdigest(),
        "reference_state_sha256": state_digest(reference),
        "target_state_sha256": state_digest(target),
        "mean_replay_max": float(np.max(np.abs(am - bm))),
        "covariance_replay_max": covariance_error,
        "is_exact_input_noop": h.tobytes() == target.tobytes(),
        "permutation_is_identity": reference.tobytes() == target.tobytes(),
        "permutation_displacement_norm": float(np.linalg.norm(b - a)),
        "total_intervention_norm": float(np.linalg.norm(b - h)),
    }


def enumerate_interventions(old_design, spec):
    """Select all multi-token pairs and six lowest-ID no-ops solely from input metadata."""
    if spec["arms"] != ["receiver_reverse", "centered_reverse"] or spec["layer"] != 16:
        raise ValueError("fixed layer and two arms required")
    by_pair = {}
    for row in old_design:
        if row["query"] == spec["queries"][0] and row["arm"] == "mean_only":
            if row["pair_id"] in by_pair:
                raise ValueError("duplicate input pair")
            by_pair[row["pair_id"]] = row
    if len(by_pair) != spec["original_pairs"]:
        raise ValueError("full original pair design required")
    selected = [r for r in by_pair.values() if r["tokens"] > 1]
    if len(selected) != spec["multi_pairs"]:
        raise ValueError("all declared multi-token pairs required")
    for role in spec["roles"]:
        for axis in spec["axes"]:
            candidates = [
                r
                for r in by_pair.values()
                if r["role"] == role and r["axis"] == axis and r["tokens"] == 1
            ]
            if not candidates:
                raise ValueError("one single-token control per role/axis required")
            selected.append(min(candidates, key=lambda r: r["pair_id"]))
    selected_ids = sorted(r["pair_id"] for r in selected)
    lookup = {
        (r["pair_id"], r["query"]): r for r in old_design if r["arm"] == "mean_only"
    }
    rows = []
    for pair_id in selected_ids:
        for query in spec["queries"]:
            row = lookup[pair_id, query]
            original = by_pair[pair_id]
            if any(row[k] != original[k] for k in ("tokens", "positions", "token_ids")):
                raise ValueError("query-independent property opportunity required")
            for arm in spec["arms"]:
                rows.append({**row, "unit": len(rows), "arm": arm})
    if len(rows) != spec["patch_forwards"]:
        raise ValueError("complete declared intervention count required")
    return rows


def endpoint_effect(actual, reference):
    """Return placement sensitivity and the exact two-endpoint set-function error bound."""
    values = np.asarray([actual, reference], float)
    if not np.isfinite(values).all():
        raise ValueError("finite native scores required")
    effect = float(values[0] - values[1])
    return {"effect": effect, "set_function_max_error_lower_bound": abs(effect) / 2}
