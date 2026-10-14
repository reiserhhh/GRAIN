#!/usr/bin/env python3
"""Design and finite-effect summaries for article-to-trait reader interventions."""

from __future__ import annotations

import hashlib

import numpy as np


def paired_materials(rows):
    """Choose the declared pairs by metadata only, never by measured readback."""
    lookup = {
        (r["axis"], r["family"], r["situation"], r["seed"], r["pole"]): r for r in rows
    }
    pairs = []
    for axis in ("extraversion", "neuroticism"):
        for family in ("named", "described"):
            for situation in ("opinion", "advice", "neighbor"):
                for seed in (11, 23) if axis == "extraversion" else (11,):
                    key = (axis, family, situation, seed)
                    high, low = lookup[(*key, 1)], lookup[(*key, -1)]
                    pairs.append(
                        {
                            "pair": "|".join(map(str, key)),
                            "axis": axis,
                            "family": family,
                            "situation": situation,
                            "seed": seed,
                            "role": "discovery"
                            if situation == "opinion"
                            else "application",
                            "high": high,
                            "low": low,
                        }
                    )
    return pairs


def splice_context(prefix, suffix, high, low, cap=128):
    """Equalize token positions without pretending ordinal tokens align semantically."""
    n = min(cap, len(high), len(low))
    if n < 32:
        raise ValueError("Insufficient paired article tokens")
    positions = list(range(len(prefix), len(prefix) + n))
    return (
        list(prefix) + list(high[:n]) + list(suffix),
        list(prefix) + list(low[:n]) + list(suffix),
        positions,
    )


def equal_norm_random(low, high, key):
    """Make a reproducible per-row norm-matched perturbation, independent of resume."""
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    low, high = np.asarray(low), np.asarray(high)
    direction = rng.standard_normal(low.shape)
    norm = np.linalg.norm(high - low, axis=-1, keepdims=True)
    direction *= norm / np.maximum(
        np.linalg.norm(direction, axis=-1, keepdims=True), 1e-15
    )
    return low + direction


def effect_summary(rows, repeats=2000):
    """Summarize raw reader effects, resampling family/task blocks with seeds together."""
    if not rows:
        return {"n": 0}
    d = np.asarray([r["effect"] for r in rows], dtype=float)
    contrast = np.asarray([r["contrast"] for r in rows], dtype=float)
    blocks = [r["family"] + "|" + r["situation"] for r in rows]
    unique = sorted(set(blocks))
    groups = [np.flatnonzero(np.asarray(blocks) == b) for b in unique]
    rng = np.random.default_rng(20260914)
    means = []
    for _ in range(repeats):
        ix = np.concatenate(
            [groups[j] for j in rng.integers(len(groups), size=len(groups))]
        )
        means.append(float(d[ix].mean()))
    denom = float(contrast @ contrast)
    return {
        "n": len(rows),
        "blocks": len(groups),
        "mean_effect": float(d.mean()),
        "mean_contrast": float(contrast.mean()),
        "mean_effect_ci": np.quantile(means, [0.025, 0.975]).tolist(),
        "donor_mse": float(np.mean((d - contrast) ** 2)),
        "zero_mse": float(np.mean(contrast**2)),
        "slope": float(contrast @ d / denom) if denom > 1e-12 else None,
        "positive_count": int((d > 0).sum()),
    }


def select_discovery(rows, field):
    """Choose donor reconstruction by E/opinion only; never application outcomes."""
    groups = {}
    for row in rows:
        if (
            row["axis"] == "extraversion"
            and row["query"] == "extraversion"
            and row["role"] == "discovery"
            and row["kind"] == "donor"
        ):
            groups.setdefault(row[field], []).append(
                (row["effect"] - row["contrast"]) ** 2
            )
    if not groups:
        raise ValueError("No eligible discovery rows")
    return min(groups, key=lambda k: (np.mean(groups[k]), str(k)))
