#!/usr/bin/env python3
"""SOURCE-constrained path functionals, fixed accounts and feasible ambiguity."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from grain_research.construction.math_observed_path_memory import triple_mass


def profile_bits():
    """Decode the already measured ENACO binary profiles, not inferred types."""
    return ((np.arange(32)[:, None] >> np.arange(5)) & 1).astype(int)


def validate_table(table):
    """Require a normalized full-profile by three-state empirical mass law."""
    f = np.asarray(table, float)
    if (
        f.ndim != 4
        or f.shape[0] != 32
        or len(set(f.shape[1:])) != 1
        or not np.isfinite(f).all()
        or np.any(f < 0)
        or abs(f.sum() - 1) > 1e-12
    ):
        raise ValueError("Normalized nonnegative full-profile triple law required")
    return f


def collapsed_bounds(table):
    """Bound each target triple mass by nonnegativity of eight binary atoms."""
    f = validate_table(table)
    r = f.sum(0)
    joint = np.einsum("gq,gabc->qabc", profile_bits(), f)
    p = joint.sum((1, 3))[:, None, :, None]
    x, z = r.sum(2)[None, :, :, None], r.sum(0)[None, None]
    t = r.sum((0, 2))[None, None, :, None]
    px, pz = joint.sum(3)[..., None], joint.sum(1)[:, None]
    xz = np.broadcast_to(r, joint.shape)
    lower = np.maximum.reduce(
        [np.zeros_like(joint), px + pz - p, px + xz - x, pz + xz - z]
    )
    upper = np.minimum.reduce(
        [
            np.broadcast_to(px, joint.shape),
            np.broadcast_to(pz, joint.shape),
            xz,
            t - p - x - z + px + pz + xz,
        ]
    )
    if np.max(lower - upper) > 1e-12:
        raise ValueError("Incompatible collapsed binary pair margins")
    return r, joint, lower, upper


def circuits(table):
    """Enumerate positive single-target checkerboards, not a full Markov basis."""
    f = validate_table(table)
    k = f.shape[1]
    pair = np.array(list(combinations(range(k), 2)), int).reshape(-1, 2)
    indexes, amplitudes = [], []
    for q in range(5):
        for b in range(k):
            for g0 in range(32):
                if g0 & (1 << q):
                    continue
                g1 = g0 | (1 << q)
                aa, cc = np.meshgrid(np.arange(len(pair)), np.arange(len(pair)))
                a0, a1 = pair[aa.ravel()].T
                c0, c1 = pair[cc.ravel()].T
                corners = np.array(
                    [
                        f[g, a, b, c]
                        for g in (g0, g1)
                        for a in (a0, a1)
                        for c in (c0, c1)
                    ]
                ).T
                minimum = corners.min(1)
                keep = minimum > 0
                if keep.any():
                    n = int(keep.sum())
                    indexes.append(
                        np.column_stack(
                            [
                                np.full(n, q),
                                np.full(n, b),
                                np.full(n, g0),
                                np.full(n, g1),
                                a0[keep],
                                a1[keep],
                                c0[keep],
                                c1[keep],
                            ]
                        )
                    )
                    amplitudes.append(minimum[keep] / 2)
    if not indexes:
        return np.empty((0, 8), int), np.empty(0)
    return np.concatenate(indexes), np.concatenate(amplitudes)


def compile_law(table):
    """Compile SOURCE margins/bounds/moves without retaining author arrays."""
    f = validate_table(table)
    r, joint, lower, upper = collapsed_bounds(f)
    index, amplitude = circuits(f)
    return {
        "reference": r,
        "point_field": np.divide(
            joint, r, out=np.full_like(joint, np.nan), where=r > 0
        ),
        "lower_field": np.divide(
            lower, r, out=np.full_like(lower, np.nan), where=r > 0
        ),
        "upper_field": np.divide(
            upper, r, out=np.full_like(upper, np.nan), where=r > 0
        ),
        "target_AB": joint.sum(3),
        "target_BC": joint.sum(1),
        "circuit_index": index,
        "circuit_amplitude": amplitude,
    }


def additive_account(mass, reference):
    """Project H/R on source-weighted A/B plus B/C functions; keep remainder."""
    h, r = np.asarray(mass, float), np.asarray(reference, float)
    if (
        h.shape != r.shape
        or h.ndim != 3
        or len(set(h.shape)) != 1
        or np.any(h < 0)
        or not np.isfinite(h).all()
        or abs(h.sum() - 1) > 1e-12
        or np.any((h > 0) & (r <= 0))
    ):
        raise ValueError("Normalized supported history mass required")
    k = len(h)
    density = np.divide(h, r, out=np.zeros_like(h), where=r > 0)
    design = np.column_stack(
        [np.repeat(np.eye(k), k, axis=0), np.tile(np.eye(k), (k, 1))]
    )
    left, right, fitted = np.zeros((k, k)), np.zeros((k, k)), np.zeros_like(h)
    for b in range(k):
        weights = r[:, b, :].ravel()
        used = weights > 0
        if not used.any():
            continue
        root = np.sqrt(weights[used])
        coef = np.linalg.lstsq(
            design[used] * root[:, None],
            density[:, b, :].ravel()[used] * root,
            rcond=None,
        )[0]
        left[:, b], right[b] = coef[:k], coef[k:]
        fitted[:, b] = (design @ coef).reshape(k, k)
    return density, fitted, left, right


def circuit_rectangle(density, index):
    """Evaluate all declared path rectangles without scoring any labels."""
    if len(index) == 0:
        return np.empty(0)
    _, b, _, _, a0, a1, c0, c1 = index.T
    return (
        density[a0, b, c0]
        - density[a0, b, c1]
        - density[a1, b, c0]
        + density[a1, b, c1]
    )


def measure_mass(law, mass):
    """Return a point, fixed account, outer range and positive-law certificates."""
    r = law["reference"]
    h = np.asarray(mass, float)
    coverage = float(h[r > 0].sum())
    if np.any((h > 0) & (r <= 0)):
        return {"supported": False, "coverage": coverage, "targets": None}
    density, fitted, left, right = additive_account(h, r)
    use = h > 0
    point = np.sum(h[use] * law["point_field"][:, use], axis=1)
    low = np.sum(h[use] * law["lower_field"][:, use], axis=1)
    high = np.sum(h[use] * law["upper_field"][:, use], axis=1)
    fixed = np.einsum("qab,ab->q", law["target_AB"], left) + np.einsum(
        "qbc,bc->q", law["target_BC"], right
    )
    indexes, amplitudes = law["circuit_index"], law["circuit_amplitude"]
    signed_gap = -2 * amplitudes * circuit_rectangle(density, indexes)
    rectangles = circuit_rectangle(fitted, indexes)
    rows = []
    for q in range(5):
        eligible = np.flatnonzero(indexes[:, 0] == q)
        witness = (
            int(eligible[np.argmax(abs(signed_gap[eligible]))])
            if len(eligible)
            else None
        )
        gap = float(signed_gap[witness]) if witness is not None else 0.0
        rows.append(
            {
                "target_index": q,
                "empirical_point": float(point[q]),
                "fixed_additive_account": float(fixed[q]),
                "retained_joint_account": float(point[q] - fixed[q]),
                "outer_low": float(low[q]),
                "outer_high": float(high[q]),
                "outer_width": float(high[q] - low[q]),
                "eligible_positive_circuits": len(eligible),
                "witness_index": witness,
                "signed_witness_plus_minus_gap": gap,
                "certified_feasible_width": abs(gap),
                "witness_plus_score": float(point[q] + gap / 2),
                "witness_minus_score": float(point[q] - gap / 2),
                "fixed_rectangle_max_error": float(np.max(abs(rectangles[eligible])))
                if len(eligible)
                else 0.0,
                "status": "FEASIBLE_AMBIGUITY_WITNESS"
                if abs(gap) > 1e-12
                else "NO_TESTED_WITNESS_NOT_IDENTIFICATION",
            }
        )
    return {"supported": True, "coverage": coverage, "targets": rows}


def witness_laws(table, index, amplitude):
    """Instantiate both positive laws for one signed fixed-margin circuit."""
    f = validate_table(table)
    _, b, g0, g1, a0, a1, c0, c1 = map(int, index)
    delta = np.zeros_like(f)
    for i, g in enumerate((g0, g1)):
        for j, a in enumerate((a0, a1)):
            for k, c in enumerate((c0, c1)):
                delta[g, a, b, c] = (-1) ** (i + j + k)
    return f + amplitude * delta, f - amplitude * delta


def history_mass(record, states=8):
    """Validate only identified chronological observations, never outcomes."""
    if set(record) != {"author_key", "events"} or not isinstance(
        record["author_key"], str
    ):
        raise ValueError("Only opaque author key and identified observations allowed")
    events = record["events"]
    if len(events) < 3 or any(
        set(e) != {"event_id", "timestamp", "cell"} for e in events
    ):
        raise ValueError("At least three qualified observations required")
    ids = [e["event_id"] for e in events]
    times = np.asarray([e["timestamp"] for e in events], float)
    if (
        any(not isinstance(i, str) or not i for i in ids)
        or len(set(ids)) != len(ids)
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError(
            "Distinct event identities and strictly increasing times required"
        )
    return triple_mass(np.asarray([e["cell"] for e in events]), states)
