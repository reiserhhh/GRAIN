#!/usr/bin/env python3
"""Person-qualified reader fields, separate from persona-conditioned writing."""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np

AXES = ("extraversion", "neuroticism")
FAMILIES = ("named", "described", "anchored")
SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)))
CHARACTERS = np.stack(
    [
        np.prod(SIGNS[:, [j for j in range(3) if mask & (1 << j)]], axis=1)
        for mask in range(8)
    ]
)
QUERIES = tuple(itertools.product(AXES, FAMILIES, (-1, 1)))
BANKS = (
    (
        (
            "Person {hi} started conversations with several people at the gathering.\n"
            "Person {lo} stayed apart and spoke only when approached at the gathering.\n"
        ),
        (
            "Person {hi} worried about several possible problems before the appointment.\n"
            "Person {lo} remained calm about several possible problems before the appointment.\n"
        ),
    ),
    (
        (
            "Person {hi} approached unfamiliar teammates and kept the conversation going.\n"
            "Person {lo} spent the break alone and gave brief replies to teammates.\n"
        ),
        (
            "Person {hi} repeatedly anticipated mistakes while waiting for the result.\n"
            "Person {lo} felt relaxed and stopped thinking about mistakes while waiting for the result.\n"
        ),
    ),
)
CONDITIONS = (
    "The social behavior was chosen freely by each person.\n",
    "The social behavior was required by the organiser.\n",
)
SYSTEM = (
    "Read the fictional record. Evaluate only the person asked about. "
    "Answer the question with Yes or No. Do not explain."
)


def materials():
    """Return the complete32 records, with names exchanged but words retained."""
    rows = []
    for bank, condition, order, e, n in itertools.product(
        range(2), range(2), range(2), (-1, 1), (-1, 1)
    ):
        parts = [
            BANKS[bank][j].format(
                hi="A" if sign == 1 else "B", lo="B" if sign == 1 else "A"
            )
            for j, sign in enumerate((e, n))
        ]
        body = "Fictional record:\n" + CONDITIONS[condition]
        intervals = {}
        for j in (0, 1) if order == 0 else (1, 0):
            start = len(body)
            body += parts[j]
            intervals[AXES[j]] = [start, len(body)]
        body += "Record ends.\n"
        rows.append(
            {
                "id": len(rows),
                "bank": bank,
                "condition": condition,
                "order": order,
                "e": e,
                "n": n,
                "body": body,
                "intervals": intervals,
                "role": "source" if bank == 0 else "held_expression",
            }
        )
    for bank, condition, order in itertools.product(range(2), repeat=3):
        group = [
            r
            for r in rows
            if (r["bank"], r["condition"], r["order"]) == (bank, condition, order)
        ]
        if len({tuple(sorted(Counter(r["body"].split()).items())) for r in group}) != 1:
            raise ValueError("Assignment edits changed words")
    return rows


def questions():
    """Freeze three descriptions per target and both named people, before data."""
    claims = {
        "extraversion": (
            "has high extraversion",
            "generally seeks social contact",
            "is more likely to initiate conversation than stay apart",
        ),
        "neuroticism": (
            "has high neuroticism",
            "is generally prone to worry and tension",
            "is more likely to remain worried than stay calm",
        ),
    }
    return [
        f"Does this record provide evidence that Person {'A' if p == 1 else 'B'} "
        f"{claims[axis][FAMILIES.index(family)]}? Answer Yes or No."
        for axis, family, p in QUERIES
    ]


def characters(cube):
    """Expand all eight e/n/person characters without erasing context terms."""
    x = np.asarray(cube, dtype=np.float64)
    if x.shape[:3] != (2, 2, 2) or not np.isfinite(x).all():
        raise ValueError("Complete e/n/queried-person support required")
    return np.einsum("ma,a...->m...", CHARACTERS, x.reshape(8, *x.shape[3:])) / 8


def reconstruct(coefficients):
    """Invert the complete finite transform, including every mixed term."""
    c = np.asarray(coefficients, dtype=np.float64)
    if c.shape[0] != 8 or not np.isfinite(c).all():
        raise ValueError("Eight finite coefficients required")
    return np.einsum("ma,m...->a...", CHARACTERS, c).reshape(2, 2, 2, *c.shape[1:])


def composition(base, e, n, both, whole):
    """Keep local effects, their nonlinear interaction and nonlocal remainder."""
    values = [np.asarray(a, dtype=np.float64) for a in (base, e, n, both, whole)]
    if len({a.shape for a in values}) != 1 or not all(
        np.isfinite(a).all() for a in values
    ):
        raise ValueError("Aligned finite intervention endpoints required")
    b, x, y, xy, w = values
    return {
        "e": x - b,
        "n": y - b,
        "interaction": xy - x - y + b,
        "nonlocal_remainder": w - xy,
        "whole": w - b,
    }


def action_account(effect, reference):
    """Describe signed transport and orthogonal remainder, not prediction accuracy."""
    x, y = (np.asarray(a, dtype=np.float64).ravel() for a in (effect, reference))
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Aligned finite action vectors required")
    xx, yy, xy = float(x @ x), float(y @ y), float(x @ y)
    return {
        "rms": float(np.sqrt(xx / len(x))),
        "reference_rms": float(np.sqrt(yy / len(y))),
        "signed_projection": xy / yy if yy > 0 else None,
        "cosine": xy / np.sqrt(xx * yy) if xx > 0 and yy > 0 else None,
        "relative_remainder": float(np.linalg.norm(x - y) / np.sqrt(yy))
        if yy > 0
        else None,
    }


def cross_wording_support(gram):
    """Return net three-family agreement, without calling it psychological validity."""
    g = np.asarray(gram, dtype=np.float64)
    if g.shape != (3, 3) or not np.isfinite(g).all():
        raise ValueError("Three finite family fields required")
    if not np.allclose(g, g.T, atol=1e-9) or np.linalg.eigvalsh(g).min() < -1e-8:
        raise ValueError("Invalid field Gram; never clip into PSD")
    return float((2 / 3) * (g[0, 1] + g[0, 2] + g[1, 2]))
