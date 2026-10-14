#!/usr/bin/env python3
"""Source-observed text anchors, distinct from persona labels and model answers."""

from __future__ import annotations

import re

import numpy as np

TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def lexical_observations(text, definitions):
    """Count registered surface forms; do not infer experienced emotion or agency."""
    tokens = TOKEN.findall(text.lower().replace("\u2019", "'"))
    counts = []
    for definition in definitions:
        if definition["kind"] == "words":
            vocabulary = set(definition["forms"])
            value = sum(token in vocabulary for token in tokens)
        elif definition["kind"] == "negation":
            value = sum(
                token in {"no", "not", "never", "cannot", "nobody", "nothing"}
                or token.endswith("n't")
                for token in tokens
            )
        elif definition["kind"] == "punctuation":
            value = text.count(definition["symbol"])
        else:
            raise ValueError("Unknown surface observation definition")
        counts.append(value)
    counts = np.asarray(counts, dtype=np.int64)
    rates = 100 * counts.astype(float) / max(len(tokens), 1)
    return counts, rates, len(tokens)


def source_units(values, source_mask):
    """Freeze per-anchor source units, without application outcomes or centering."""
    y = np.asarray(values, float)
    s = np.asarray(source_mask, bool)
    if y.ndim != 2 or s.shape != (len(y),) or not s.any():
        raise ValueError("A disjoint, nonempty source cohort is required")
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite observable input")
    center, spread = y[s].mean(0), y[s].std(0)
    support = spread > 0
    scale = np.where(support, spread, 1.0)
    return (y - center) / scale, center, scale, support


def anchor_program(features, observations, conditions):
    """Construct observed cross-moments and retain within AND between conditions."""
    x, y = np.asarray(features, float), np.asarray(observations, float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError(
            "Aligned source events with at least two observations required"
        )
    if (
        len(conditions) != len(x)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Finite, qualified source observations required")
    mx, my = x.mean(0), y.mean(0)
    total = (y - my).T @ (x - mx) / len(x)
    within, between = np.zeros_like(total), np.zeros_like(total)
    keys = sorted(set(conditions))
    for key in keys:
        mask = np.asarray([c == key for c in conditions])
        xc, yc = x[mask], y[mask]
        mxc, myc = xc.mean(0), yc.mean(0)
        within += (yc - myc).T @ (xc - mxc) / len(x)
        between += len(xc) / len(x) * np.outer(myc - my, mxc - mx)
    norm = np.linalg.norm(total, axis=1)
    return {
        "origin": mx,
        "total": total,
        "within": within,
        "between": between,
        "norm": norm,
        "supported": norm > 0,
        "decomposition_error": float(np.max(abs(total - within - between))),
    }


def cold_score(program, features, arm):
    """Apply a source-only signed witness, NOT a calibrated response expectation."""
    if arm not in {"total", "within", "between"}:
        raise ValueError("Unknown registered cross-moment component")
    x = np.asarray(features, float)
    if x.ndim != 2 or x.shape[1] != program["origin"].shape[0]:
        raise ValueError("Observation representation differs from source")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite cold representation")
    numerator = (x - program["origin"]) @ program[arm].T
    result = np.full_like(numerator, np.nan)
    active = program["supported"]
    result[:, active] = numerator[:, active] / program["norm"][active]
    return result


def persona_pairs(rows):
    """Pair actual high/low articles within task, style, family and replicate."""
    table = {}
    for i, row in enumerate(rows):
        if row["axis"] == "none":
            continue
        key = tuple(row[k] for k in ("axis", "family", "situation", "style", "seed"))
        poles = table.setdefault(key, {})
        if row["pole"] in poles:
            raise ValueError("Duplicate observation at a persona corner")
        poles[row["pole"]] = i
    pairs, metadata = [], []
    for key, poles in sorted(table.items()):
        if set(poles) != {-1, 1}:
            raise ValueError("Missing observed persona corner; no synthetic filling")
        lo, hi = poles[-1], poles[1]
        if rows[lo]["role"] != rows[hi]["role"]:
            raise ValueError("Pair crosses collection roles")
        pairs.append((hi, lo))
        metadata.append(
            dict(
                zip(("axis", "family", "situation", "style", "seed"), key, strict=True)
            )
            | {"role": rows[hi]["role"]}
        )
    return np.asarray(pairs, int), metadata


def pair_changes(values, pairs):
    """Use complete actual paired observations; preserve zeros and reversals."""
    x, p = np.asarray(values, float), np.asarray(pairs, int)
    if p.ndim != 2 or p.shape[1] != 2 or np.any(p < 0) or np.any(p >= len(x)):
        raise ValueError("Invalid pair addresses")
    return x[p[:, 0]] - x[p[:, 1]]


def signed_comparison(observed, reading, tolerance=1e-12):
    """Separate contrary directions from exactly absent registered observations."""
    y, r = np.asarray(observed, float), np.asarray(reading, float)
    if y.shape != r.shape:
        raise ValueError("Different observed and scored endpoint shapes")
    active = (
        np.isfinite(y) & np.isfinite(r) & (abs(y) > tolerance) & (abs(r) > tolerance)
    )
    same = active & ((y > 0) == (r > 0))
    opposite = active & ~same
    return {
        "same": int(same.sum()),
        "opposite": int(opposite.sum()),
        "observable_zero": int((np.isfinite(y) & (abs(y) <= tolerance)).sum()),
        "reading_zero": int((np.isfinite(r) & (abs(r) <= tolerance)).sum()),
        "unavailable": int((~np.isfinite(y) | ~np.isfinite(r)).sum()),
    }
