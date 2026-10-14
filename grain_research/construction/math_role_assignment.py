#!/usr/bin/env python3
"""Independent role assignments and exact restriction-alias observation laws."""

from __future__ import annotations

import numpy as np

from .math_heterogeneous_evidence_cube import TARGETS, coefficients, grid
from .math_ownership_composition import ANCHOR
from .math_passive_event_composition import action_line

VARIABLES = ("x", "y", "g", "h")
STAGES = ("score", "log_odds", "centered_logits", "body_endpoint", "query_endpoint")


def documents(material):
    """Enumerate both full actor assignments without changing any behaviour text."""
    if material["targets"] != list(TARGETS):
        raise ValueError("The six frozen targets are required")
    rows = []
    for family in (0, 1):
        for focal in TARGETS:
            for companion in (q for q in TARGETS if q != focal):
                for corner, point in enumerate(grid(4).astype(int)):
                    x, y, g, h = point.tolist()
                    a = material["behaviours"][focal][family][int(x > 0)]
                    b = material["behaviours"][companion][family][int(y > 0)]
                    rows.append(
                        {
                            "doc_id": len(rows),
                            "family": family,
                            "focal": focal,
                            "companion": companion,
                            "corner": corner,
                            "coordinates": [x, y, g, h],
                            "person_a_events": 1 + int(g > 0) + int(h > 0),
                            "body": "\n".join(
                                (
                                    action_line("K1", "A", a),
                                    action_line("K2", "A" if g > 0 else "B", b),
                                    action_line("K3", "A" if h > 0 else "B", ANCHOR),
                                )
                            ),
                        }
                    )
    return rows


def complete(values):
    """Validate a full sixteen-cell observation with finite trailing channels."""
    a = np.asarray(values, dtype=float)
    if a.ndim < 1 or a.shape[0] != 16 or not np.isfinite(a).all():
        raise ValueError("Sixteen finite observations required")
    return a


def old_fraction(values):
    """Restrict the complete table to the historical h=-g design in old order."""
    a = complete(values)
    points = grid(4)
    return a[points[:, 3] == -points[:, 2]]


def aliased_coefficients(full_coefficients):
    """Compute all old coefficients from the full character restriction map."""
    c = complete(full_coefficients).reshape(
        2, 2, 2, 2, *np.shape(full_coefficients)[1:]
    )
    return np.asarray(
        [
            c[a, b, g, 0] - c[a, b, 1 - g, 1]
            for a in (0, 1)
            for b in (0, 1)
            for g in (0, 1)
        ]
    )


def effects(values):
    """Separate companion, anchor and joint gating without sparsity assumptions."""
    a = complete(values)
    t = a.reshape(2, 2, 2, 2, *a.shape[1:])
    mixed = t[1, 1] - t[1, 0] - t[0, 1] + t[0, 0]
    c = coefficients(a, 4)
    return {
        "coefficients": c,
        "mixed": mixed,
        "companion_gate_at_h": mixed[1] - mixed[0],
        "anchor_gate_at_g": mixed[:, 1] - mixed[:, 0],
        "old_gate": mixed[1, 0] - mixed[0, 1],
        "old_coefficients": coefficients(old_fraction(a), 3),
        "alias_from_full": aliased_coefficients(c),
    }


def profile_pair(epsilon=0.5):
    """Create two valid compositions sharing every marginal of order <=3."""
    if not np.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise ValueError("A finite epsilon in (0,1] is required")
    character = np.prod(grid(4), axis=1)
    return (1 + epsilon * character) / 16, (1 - epsilon * character) / 16
