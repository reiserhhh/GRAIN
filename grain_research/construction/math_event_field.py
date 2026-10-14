#!/usr/bin/env python3
"""Event-indexed fields with a lossless finite presentation decomposition."""

from __future__ import annotations

import itertools

import numpy as np

from .math_cold_relational import PAIRS, cosine
from .math_heterogeneous_evidence_cube import grid

EVENTS = ("K1", "K2", "K3")
FEATURES = (
    "body_end",
    "event_end",
    "event_mean",
    "action_mean",
    "pooled_event_mean",
)
MODES = ("forward", "reverse", "even", "odd")
CHARACTERS = ("x", "y", "xy", "g", "xg", "yg", "xyg")


def documents(source_rows, cold_rows):
    """Make both exact sentence orders on the full declared source/new support."""
    result = []
    for bank, rows in (("source", source_rows), ("new", cold_rows)):
        unique = {}
        for row in rows:
            if bank == "source" and row["family"] != 0:
                continue
            if bank == "new" and row["form"] != "paraphrase":
                continue
            if (row["focal"], row["companion"]) not in PAIRS:
                continue
            pair = PAIRS.index((row["focal"], row["companion"]))
            key = pair, row["corner"]
            if key in unique and unique[key]["body"] != row["body"]:
                raise ValueError(
                    "Same cell must have the identical body across queries"
                )
            unique.setdefault(key, row)
        if set(unique) != set(itertools.product(range(30), range(8))):
            raise ValueError("All30 directed pairs/all8 cells required")
        for (pair, cell), row in sorted(unique.items()):
            lines = row["body"].split("\n")
            if len(lines) != 3 or any(
                not line.startswith(f"On occasion {event}, Person ")
                for event, line in zip(EVENTS, lines, strict=True)
            ):
                raise ValueError("Explicit ordered event identities required")
            for order in (0, 1):
                result.append(
                    {
                        "unit": len(result),
                        "bank": bank,
                        "pair": pair,
                        "cell": cell,
                        "order": order,
                        "body": "\n".join(lines if order == 0 else lines[::-1]),
                    }
                )
    return result


def positions(rendered, body, offsets):
    """Bind each actual contextual event/action span without deleting its actor."""
    if rendered.count(body) != 1:
        raise ValueError("Unique body required")
    origin = rendered.index(body)
    spans = {}
    cursor = 0
    for line in body.split("\n"):
        event = line.split(",", 1)[0].removeprefix("On occasion ")
        header = f"On occasion {event}, Person "
        if event not in EVENTS or event in spans or not line.startswith(header):
            raise ValueError("Unique K1/K2/K3 identities required")
        actor_start = len(header)
        if line[actor_start : actor_start + 2] not in ("A ", "B "):
            raise ValueError("Explicit A/B actor required")
        start, end = origin + cursor, origin + cursor + len(line)
        action_start = start + actor_start + 2
        address = {}
        for name, left in (("event", start), ("action", action_start)):
            address[name] = [
                i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > left
            ]
            if not address[name]:
                raise ValueError("No actual event/action tokens")
        address["actor"] = line[actor_start]
        address["raw"] = line
        spans[event] = address
        cursor += len(line) + 1
    if set(spans) != set(EVENTS):
        raise ValueError("Complete event support required")
    return spans


def extract(hidden, spans):
    """Keep event-aligned contextual vectors and a label-forgetting pool."""
    h = np.asarray(hidden)
    if h.ndim != 2 or not np.isfinite(h).all():
        raise ValueError("Finite token-by-vector field required")
    result = {}
    for name, slot in (("event_mean", "event"), ("action_mean", "action")):
        result[name] = np.stack(
            [h[spans[e][slot]].mean(axis=0, dtype=np.float64) for e in EVENTS]
        )
    result["event_end"] = np.stack([h[spans[e]["event"][-1]] for e in EVENTS])
    result["body_end"] = h[max(spans[e]["event"][-1] for e in EVENTS)]
    result["pooled_event_mean"] = result["event_mean"].mean(axis=0)
    return result


def decompose(values):
    """Retain both even and odd fields; orbit invariance is not discovered."""
    a = np.asarray(values, dtype=float)
    if a.ndim < 2 or a.shape[0] != 2 or not np.isfinite(a).all():
        raise ValueError("Two complete finite presentation orders required")
    return dict(zip(MODES, (*a, (a[0] + a[1]) / 2, (a[0] - a[1]) / 2), strict=True))


def observe(probabilities, field):
    """Contract fixed profile masses with event-indexed observations, without fit."""
    p, h = np.asarray(probabilities, float), np.asarray(field, float)
    if (
        p.shape != (2, 8)
        or h.shape[0] != 8
        or not np.isfinite(p).all()
        or not np.isfinite(h).all()
        or np.any(p < 0)
        or np.max(abs(p.sum(axis=1) - 1)) > 1e-12
    ):
        raise ValueError("Normalized paired profiles on eight fixed cells required")
    return np.tensordot(p[1] - p[0], h, axes=1)


def characters():
    """Return all nonconstant Boolean contrasts in declared x,y,g cell order."""
    cells = np.asarray(grid(3), float)
    return np.stack(
        [
            np.prod(cells[:, [j for j in range(3) if mask & (1 << j)]], axis=1)
            for mask in range(1, 8)
        ]
    )


def relation_row(source, new):
    """Report direction/magnitude separately; a zero direction remains undefined."""
    source, new = np.asarray(source, float), np.asarray(new, float)
    wrong_slot = np.roll(new, 1, axis=0) if new.ndim == 2 else None
    return {
        "source_norm": float(np.linalg.norm(source)),
        "new_norm": float(np.linalg.norm(new)),
        "error_norm": float(np.linalg.norm(new - source)),
        "direction_cosine": cosine(source, new),
        "wrong_slot_cosine": None if wrong_slot is None else cosine(source, wrong_slot),
    }
