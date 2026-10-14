#!/usr/bin/env python3
"""S3 observation sectors distinguish role pairs from generic actor counts."""

from __future__ import annotations

import itertools

import numpy as np

from .math_heterogeneous_evidence_cube import TARGETS, coefficients, grid
from .math_ownership_composition import ANCHOR
from .math_passive_event_composition import action_line

VARIABLES = ("x", "y", "g", "h", "i")
STAGES = ("score", "log_odds", "centered_logits", "body_endpoint", "query_endpoint")
PERIPHERAL = "stood beside a closed door"
EDGES = ("gh", "gi", "hi")
PAIR_INDEX = (30, 29, 27)


def queries(focal, companion):
    """Return both declared targets and a deterministic unused target."""
    if focal not in TARGETS or companion not in TARGETS or focal == companion:
        raise ValueError("Two distinct frozen targets required")
    start = TARGETS.index(companion)
    other = next(
        TARGETS[(start + k) % 6]
        for k in range(1, 6)
        if TARGETS[(start + k) % 6] not in (focal, companion)
    )
    return dict(zip(("focal", "companion", "off_target"), (focal, companion, other)))


def documents(material):
    """Enumerate new four-event accounts with three independent actor slots."""
    if material["targets"] != list(TARGETS):
        raise ValueError("Frozen behaviour bank required")
    docs = []
    for family in (0, 1):
        for focal in TARGETS:
            for companion in (q for q in TARGETS if q != focal):
                for corner, point in enumerate(grid(5).astype(int)):
                    x, y, g, h, i = point.tolist()
                    actions = (
                        material["behaviours"][focal][family][int(x > 0)],
                        material["behaviours"][companion][family][int(y > 0)],
                        ANCHOR,
                        PERIPHERAL,
                    )
                    actors = (
                        "A",
                        "A" if g > 0 else "B",
                        "A" if h > 0 else "B",
                        "A" if i > 0 else "B",
                    )
                    body = "\n".join(
                        action_line(f"K{k + 1}", actor, action)
                        for k, (actor, action) in enumerate(
                            zip(actors, actions, strict=True)
                        )
                    )
                    docs.append(
                        {
                            "doc_id": len(docs),
                            "family": family,
                            "focal": focal,
                            "companion": companion,
                            "corner": corner,
                            "coordinates": point.tolist(),
                            "person_a_events": 1 + sum(v > 0 for v in (g, h, i)),
                            "body": body,
                        }
                    )
    return docs


def complete(values):
    """Require a complete finite32-cell observation with trailing channels."""
    a = np.asarray(values, dtype=float)
    if a.ndim < 1 or a.shape[0] != 32 or not np.isfinite(a).all():
        raise ValueError("Complete finite32-cell values required")
    return a


def count_projection(values):
    """Average the S3 orbit while retaining content and actor-count variation."""
    a = complete(values)
    points = grid(5).astype(int)
    result = np.empty_like(a)
    for x, y, count in itertools.product((-1, 1), (-1, 1), range(4)):
        mask = (points[:, 0] == x) & (points[:, 1] == y)
        mask &= (points[:, 2:] > 0).sum(1) == count
        result[mask] = a[mask].mean(0)
    return result


def permutation_projection(values):
    """Use a separate group average to expose the same declared projection."""
    a = complete(values)
    points = grid(5).astype(int)
    lookup = {tuple(p): k for k, p in enumerate(points)}
    images = []
    for order in itertools.permutations(range(3)):
        index = [lookup[tuple(p[:2]) + tuple(p[2 + np.asarray(order)])] for p in points]
        images.append(a[index])
    return np.mean(images, axis=0)


def profiles(first, second, epsilon=0.5):
    """Keep content-by-count distribution fixed while changing role-pair links."""
    if first not in EDGES or second not in EDGES or first == second:
        raise ValueError("Two distinct declared edges required")
    if not np.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise ValueError("Finite epsilon in (0,1] required")
    points = grid(5)
    edge = {
        name: np.prod(points[:, [VARIABLES.index(v) for v in name]], axis=1)
        for name in EDGES
    }
    delta = points[:, 0] * points[:, 1] * (edge[first] - edge[second]) / 2
    return (1 + epsilon * delta) / 32, (1 - epsilon * delta) / 32


def sectors(values):
    """Keep count-compatible and role-indexed observations, including pair terms."""
    a = complete(values)
    c = coefficients(a, 5)
    count = count_projection(a)
    role = a - count
    pairs = c[list(PAIR_INDEX)]
    pair_count = np.broadcast_to(pairs.mean(0), pairs.shape)
    return {
        "coefficients": c,
        "count": count,
        "role": role,
        "pair_coefficients": pairs,
        "pair_count": pair_count,
        "pair_role": pairs - pair_count,
    }
