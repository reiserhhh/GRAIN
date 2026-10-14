#!/usr/bin/env python3
"""HM29 fixed source-complement acquisitions, not outcome-selected directions."""

from __future__ import annotations

import numpy as np

from .math_source_field import moment_map

CONTEXTS = (
    "preparing a short work report",
    "checking a routine travel plan",
    "arranging a small community event",
    "organizing a simple home repair",
    "planning an ordinary class activity",
    "sorting a folder of appointment notes",
    "preparing a short club presentation",
    "checking a familiar delivery schedule",
    "arranging a library volunteer shift",
    "planning an ordinary practice session",
    "organizing a small course project",
    "reviewing a routine household schedule",
    "preparing a local workshop handout",
    "checking a standard registration form",
    "arranging a neighborhood meeting",
    "planning a short office demonstration",
)
PHRASES = (
    "deeply worried and tense",
    "very nervous and worried",
    "quite anxious and tense",
    "really worried about failure",
    "uneasy about the outcome",
    "nervous about the outcome",
    "worried about what happens",
    "anxious about the result",
    "highly anxious and nervous",
    "very tense and uneasy",
    "worried about the future",
    "extremely tense and worried",
    "deeply anxious and unsettled",
    "very worried and uneasy",
    "nervous and unable to relax",
    "worried and afraid of failure",
)
ARMS = ("unseen_source", "unseen_random", "known_mean")


def material(family, phrase):
    """Render supplied self-owned evidence and fixed trust/context controls."""
    if not 0 <= family < len(CONTEXTS) or phrase not in PHRASES:
        raise ValueError("declared family and phrase required")
    return (
        f"During {CONTEXTS[family]}, I felt {phrase} while I checked a printed list. "
        "I judged AI systems generally useful while I recorded a routine note."
    )


def unit_vector(b):
    """Normalize a finite predeclared hidden vector, failing on zero evidence."""
    b = np.asarray(b, float)
    if b.ndim != 1 or not np.isfinite(b).all() or np.linalg.norm(b) == 0:
        raise ValueError("nonzero finite hidden direction required")
    return b / np.linalg.norm(b)


def directions(source_b, random_b):
    """Invert the4-token map for a declared complement or known mean coordinate."""
    source_b, random_b = unit_vector(source_b), unit_vector(random_b)
    if source_b.shape != random_b.shape:
        raise ValueError("matched hidden dimensions required")
    matrix = moment_map(4)
    n, mean = np.array([0.0, -1.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])
    return {
        "unseen_source": np.outer(np.linalg.solve(matrix, n), source_b),
        "unseen_random": np.outer(np.linalg.solve(matrix, n), random_b),
        "known_mean": np.outer(np.linalg.solve(matrix, mean), source_b),
    }


def physical_patches(h, direction, radius=0.01):
    """Return the actual float32 endpoints and both realized physical deltas."""
    h, direction = np.asarray(h, np.float32), np.asarray(direction, float)
    if (
        h.shape != direction.shape
        or h.ndim != 2
        or len(h) != 4
        or not np.isfinite(h).all()
        or not np.isfinite(direction).all()
        or np.linalg.norm(direction) == 0
        or np.linalg.norm(h) == 0
        or not np.isfinite(radius)
        or radius <= 0
    ):
        raise ValueError("finite4-token state, nonzero direction and radius required")
    intended = direction / np.linalg.norm(direction) * radius * np.linalg.norm(h)
    plus = (h + intended.astype(np.float32)).astype(np.float32)
    minus = (h - intended.astype(np.float32)).astype(np.float32)
    return {
        "intended": intended,
        "plus": plus,
        "minus": minus,
        "delta_plus": plus.astype(float) - h.astype(float),
        "delta_minus": minus.astype(float) - h.astype(float),
        "half_delta": (plus.astype(float) - minus.astype(float)) / 2,
    }


def contrast(base, plus, minus):
    """Keep native symmetric action, even displacement and full one-sided changes."""
    values = np.array([base, plus, minus], float)
    if not np.isfinite(values).all():
        raise ValueError("finite native scores required")
    return {
        "odd_action": (plus - minus) / 2,
        "even_change": (plus + minus) / 2 - base,
        "plus_change": plus - base,
        "minus_change": minus - base,
    }


def summarize(rows):
    """Compute physical finite-family diagnostics without tiny-ratio promotion."""
    if not rows:
        raise ValueError("addressed finite action rows required")
    rms = lambda a: float(np.sqrt(np.mean(np.asarray(a, float) ** 2)))
    y = np.array([r["odd_action"] for r in rows])
    output = {
        "families": len(rows),
        "native_action_rms": rms(y),
        "even_change_rms": rms([r["even_change"] for r in rows]),
    }
    for name in ("old_joint", "old_quotient", "new_source", "local_derivative"):
        p = np.array([r[name] for r in rows])
        output[name + "_rmse"] = rms(p - y)
        output[name + "_prediction_rms"] = rms(p)
        output[name + "_signs"] = int(sum(p * y > 0))
    return output
