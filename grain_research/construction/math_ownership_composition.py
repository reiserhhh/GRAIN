#!/usr/bin/env python3
"""Ownership group projections on fixed cross-target evidence tables."""

from __future__ import annotations

import numpy as np

from .math_heterogeneous_evidence_cube import TARGETS, basis, coefficients, decode, grid
from .math_passive_event_composition import action_line

NEUTRAL = (
    (
        "moved a blue folder from one side of a table to the other",
        "moved a green folder from one side of a table to the other",
    ),
    (
        "placed a blue ribbon beside a box on a shelf",
        "placed a green ribbon beside a box on a shelf",
    ),
)
ANCHOR = "opened a plain notebook on a table"
STAGES = (
    "score",
    "log_odds",
    "centered_logits",
    "body_endpoint",
    "score_link_control",
    "log_odds_link_control",
)
WORLDS = ("qualified", "generic", "role_marker", "link_only")


def documents(material):
    """Enumerate all directed pairs and neutral controls, with balanced owner names."""
    if material["targets"] != list(TARGETS):
        raise ValueError("Fixed six targets required")
    docs = []
    for family in (0, 1):
        for focal in TARGETS:
            for companion in (*[k for k in TARGETS if k != focal], "neutral"):
                for corner, (x, y, g) in enumerate(grid(3)):
                    actor = "A" if g == 1 else "B"
                    alternate = "B" if g == 1 else "A"
                    a = material["behaviours"][focal][family][int(x > 0)]
                    b = (
                        NEUTRAL[family]
                        if companion == "neutral"
                        else material["behaviours"][companion][family]
                    )[int(y > 0)]
                    docs.append(
                        {
                            "doc_id": len(docs),
                            "family": family,
                            "focal": focal,
                            "companion": companion,
                            "corner": corner,
                            "coordinates": [int(x), int(y), int(g)],
                            "body": "\n".join(
                                (
                                    action_line("K1", "A", a),
                                    action_line("K2", actor, b),
                                    action_line("K3", alternate, ANCHOR),
                                )
                            ),
                        }
                    )
    return docs


def tensor(values):
    """Validate a complete eight-corner observation without flattening its units."""
    a = np.asarray(values, dtype=float)
    if a.ndim < 2 or a.shape[0] != 8 or not np.isfinite(a).all():
        raise ValueError("Complete finite eight-corner table required")
    return a.reshape(2, 2, 2, *a.shape[1:])


def group_effects(values):
    """Separate mixed behaviour actions and ownership parity by exact differences."""
    a = tensor(values)
    mixed = a[1, 1] - a[1, 0] - a[0, 1] + a[0, 0]
    odd = (a[:, :, 1] - a[:, :, 0]) / 2
    even = (a[:, :, 1] + a[:, :, 0]) / 2
    c = coefficients(np.asarray(values), 3)
    return {
        "own": mixed[1],
        "other": mixed[0],
        "gate": mixed[1] - mixed[0],
        "coefficients": c,
        "even": even,
        "odd": odd,
        "focal_edges": a[1] - a[0],
        "identity_error": float(np.max(np.abs(mixed[1] - mixed[0] - 8 * c[7]))),
    }


def link_controls(centered_logits):
    """Decode logits after removing only focal-by-companion interaction terms."""
    z = np.asarray(centered_logits, dtype=float)
    if z.shape != (8, 6, 10):
        raise ValueError("Eight corners, six queries and ten candidates required")
    c = coefficients(z, 3)
    c[[6, 7]] = 0
    clean = (basis(grid(3), 3) @ c.reshape(8, -1)).reshape(z.shape)
    return clean, decode(clean)


def toy_world(kind, seed):
    """Save real numeric examples with declared ownership/semantic/link mechanisms."""
    if kind not in WORLDS:
        raise ValueError("Declared mechanism world required")
    rng = np.random.default_rng(seed)
    x, y, g = grid(3).T
    base = rng.normal(size=(6, 10))
    a, b, v = rng.normal(size=(3, 6, 10)) * 0.35
    x, y, g = (q[:, None, None] for q in (x, y, g))
    main = base + x * a + y * b
    neutral = base + x * a + 0.1 * y * b
    if kind == "qualified":
        main += 0.4 * x * y * (g + 1) / 2 * v
    elif kind == "generic":
        main += 0.4 * x * y * v
    elif kind == "role_marker":
        main += 0.4 * x * y * g * v
        neutral += 0.4 * x * y * g * v
    else:
        main += 0.3 * y * g * v
        neutral += 0.1 * y * g * v
    return {
        "main_logits": main - main.mean(-1, keepdims=True),
        "neutral_logits": neutral - neutral.mean(-1, keepdims=True),
    }
