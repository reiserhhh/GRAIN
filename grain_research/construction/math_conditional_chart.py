#!/usr/bin/env python3
"""Scoped opportunity kernels and finite behavior-frequency response charts."""

from __future__ import annotations

import itertools

import numpy as np

SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=int)


def material(spec, bank, family, condition, polarity, owner):
    """Return public condition/action clauses with explicit supplied identities."""
    if any(x not in (-1, 1) for x in (condition, polarity, owner)):
        raise ValueError("Binary declared support required")
    clauses = []
    for person, sign in (("A", condition), ("B", -condition)):
        phrase = spec["conditions"][bank][(sign + 1) // 2]
        clauses.append(f"On occasion K1, Person {person} {phrase}.")
    actor, other = ("A", "B") if owner == 1 else ("B", "A")
    action = spec["behaviours"][family][(polarity + 1) // 2]
    return {
        "condition_text": "\n".join(clauses),
        "action_text": f"On occasion K1, Person {actor} {action}.\nPerson {other} recorded the date.",
        "author": "A",
        "actor": actor,
        "condition_for_actor": condition if owner == 1 else -condition,
        "evidence_mode": "own_actual" if owner == 1 else "other_actual",
    }


def decompose(corners):
    """Keep all eight characters of condition, behavior and actor assignment."""
    corners = np.asarray(corners, float)
    if corners.shape[0] != 8 or not np.isfinite(corners).all():
        raise ValueError("Complete finite eight-corner support required")
    chars = np.prod(1 + 0 * SIGNS, axis=1)
    columns = []
    for mask in range(8):
        selected = [j for j in range(3) if mask & (1 << j)]
        columns.append(SIGNS[:, selected].prod(1) if selected else chars)
    return np.stack(columns) @ corners / 8


def condition_projection(chart, means, rtol=1e-10):
    """Recover identifiable coordinates by two scalar projections, not an inverse."""
    gain = np.asarray(chart["gain"], float)
    centered = np.asarray(means, float) - np.asarray(chart["offset"], float)
    norms = np.linalg.norm(gain, axis=1)
    active = norms > rtol * norms.max()
    slopes = np.zeros(2)
    for c in np.flatnonzero(active):
        slopes[c] = np.dot(centered[c], gain[c]) / norms[c] ** 2
    return np.array([slopes.mean(), (slopes[1] - slopes[0]) / 2])


def target_coupling(gains, rtol=1e-10):
    """Describe six behavior response directions; these are not author factors."""
    g = np.asarray(gains, float)
    if g.shape != (2, 6, 6) or not np.isfinite(g).all():
        raise ValueError("Two conditions/six families/six frozen target units required")
    # Each family has fixed mass 1/6 in this hypothetical mixed-evidence instrument.
    low, high = g[0].T / 6, g[1].T / 6
    d = np.block([[low, -low], [high, high]])
    _, s, vt = np.linalg.svd(d, full_matrices=False)
    rank = int(np.sum(s > rtol * s[0])) if s[0] else 0
    projector = vt[:rank].T @ vt[:rank]
    gram = np.einsum("cfq,cgq->cfg", g, g)
    norms = np.linalg.norm(g, axis=2)
    denominator = norms[:, :, None] * norms[:, None, :]
    cosines = np.divide(
        gram, denominator, out=np.zeros_like(gram), where=denominator > 0
    )
    return {
        "joint_design": d.tolist(),
        "rank": rank,
        "singular_values": s.tolist(),
        "null_basis": vt[rank:].tolist(),
        "identified_projector": projector.tolist(),
        "family_cosines_by_condition": cosines.tolist(),
        "not_psychological_factors": True,
        "equal_family_mass_is_supplied_design": True,
    }


def compile_chart(endpoints, rtol=1e-10):
    """Construct a range/null-space chart from actual own-actor controls only."""
    k = np.asarray(endpoints, float)
    if k.ndim != 3 or k.shape[:2] != (2, 2) or not np.isfinite(k).all():
        raise ValueError("Two conditions/two alternatives/finite targets required")
    m, g = k.mean(1), (k[:, 1] - k[:, 0]) / 2
    d = np.concatenate([np.column_stack((g[0], -g[0])), np.column_stack((g[1], g[1]))])
    u, s, vt = np.linalg.svd(d, full_matrices=False)
    rank = int(np.sum(s > rtol * s[0])) if s[0] else 0
    inverse = (vt[:rank].T / s[:rank]) @ u[:, :rank].T
    return {
        "offset": m.tolist(),
        "gain": g.tolist(),
        "design": d.tolist(),
        "inverse": inverse.tolist(),
        "rank": rank,
        "singular_values": s.tolist(),
        "identified_projector": (inverse @ d).tolist(),
        "null_basis": vt[rank:].tolist(),
        "response_coordinates": "declared alternative-frequency,not psychological trait",
    }


def score_chart(chart, means, counts, qualification):
    """Score qualified conditional means without any access to author parameters."""
    if qualification != "own_actual":
        return {"status": "UNAVAILABLE_OWNERSHIP", "coordinates": None}
    if len(counts) != 2 or any(n <= 0 for n in counts) or means is None:
        return {"status": "UNAVAILABLE_CONDITION", "coordinates": None}
    m = np.asarray(chart["offset"])
    y = np.asarray(means, float)
    if y.shape != m.shape or not np.isfinite(y).all():
        raise ValueError("Declared conditional target units required")
    centered = (y - m).ravel()
    coordinates = np.asarray(chart["inverse"]) @ centered
    residual = centered - np.asarray(chart["design"]) @ coordinates
    return {
        "status": "IDENTIFIED" if chart["rank"] == 2 else "PARTIAL_IDENTIFICATION",
        "coordinates": coordinates.tolist(),
        "residual": residual.tolist(),
        "choice": [n / sum(counts) for n in counts],
        "pooled_observation": np.average(y, axis=0, weights=counts).tolist(),
        "raw_condition_difference": (y[1] - y[0]).tolist(),
        "identified_projector": chart["identified_projector"],
        "null_basis": chart["null_basis"],
    }


def event_counts(level, response, share, total):
    """Build exact integer condition/alternative counts with no sampling rescue."""
    means = np.array([level - response, level + response])
    n = np.array([total * (1 - share), total * share])
    plus = n * (1 + means) / 2
    values = np.column_stack((n - plus, plus))
    if np.any(values < 0) or not np.allclose(values, np.rint(values), atol=1e-10):
        raise ValueError("Legal probabilities and exact integer counts required")
    return np.rint(values).astype(int)


def integrate(endpoints, counts):
    """Average actual vector observations using explicit condition/event counts."""
    k, n = np.asarray(endpoints, float), np.asarray(counts, int)
    if k.shape[:2] != n.shape or np.any(n < 0) or np.any(n.sum(1) == 0):
        raise ValueError("Positive observed condition support required")
    return np.einsum("cb,cbq->cq", n, k) / n.sum(1)[:, None]
