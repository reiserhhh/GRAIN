#!/usr/bin/env python3
"""Derived statistics for review cards (author-paired, seed-stratified)."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

BOOT = 2000
SEED = 20260915


def load(root: Path, path: str, **filters) -> list[dict]:
    """CSV rows matching all filters (string comparison)."""
    with (root / path).open() as f:
        return [r for r in csv.DictReader(f) if all(r.get(k) == str(v) for k, v in filters.items())]


def stratified(values: np.ndarray, seeds: np.ndarray, levels=(0.025, 0.975, 0.05, 0.95)):
    """Seed-stratified author bootstrap of a mean; returns quantiles."""
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(seeds == s) for s in sorted(set(seeds.tolist()))]
    draws = np.empty(BOOT)
    for b in range(BOOT):
        idx = np.concatenate([rng.choice(g, g.size, replace=True) for g in groups])
        draws[b] = values[idx].mean()
    return dict(zip(("lo95", "hi95", "lo90", "hi90"), np.quantile(draws, levels).tolist(), strict=True))


def decide(lo95, hi95):
    """COMMON loss-difference rule."""
    if hi95 < 0:
        return "SUPPORTED_LOWER_LOSS"
    if lo95 > 0:
        return "HIGHER_LOSS"
    return "INCONCLUSIVE"


def paired(root, path, key_fields, arm_field, a, b, value_field="value", margin=None, **filters):
    """Mean of a-b over matched units with 95%/90% CIs and optional equivalence."""
    units = defaultdict(dict)
    for r in load(root, path, **filters):
        units[tuple(r[k] for k in key_fields)][r[arm_field]] = r[value_field]
    keys = sorted(k for k, v in units.items() if a in v and b in v and v[a] != "" and v[b] != "")
    diff = np.array([float(units[k][a]) - float(units[k][b]) for k in keys])
    seeds = np.array([k[0] for k in keys])
    q = stratified(diff, seeds)
    verdict = "IDENTICAL" if np.all(diff == 0) else decide(q["lo95"], q["hi95"])
    out = {"n_units": len(keys), "delta": float(diff.mean()), **q, "decision": verdict}
    if margin is not None:
        out["equivalence_margin"] = margin
        out["equivalent_90"] = bool(-margin < q["lo90"] and q["hi90"] < margin)
    return out


def mean_ci(root, path, key_fields, value_field="value", **filters):
    """Seed-stratified bootstrap CI of a plain mean over units."""
    rows = [r for r in load(root, path, **filters) if r[value_field] != ""]
    values = np.array([float(r[value_field]) for r in rows])
    seeds = np.array([r[key_fields[0]] for r in rows])
    return {"n_units": len(rows), "mean": float(values.mean()), **stratified(values, seeds)}


def wilson(successes: float, n: int, z: float = 1.959963984540054):
    """Wilson 95% interval."""
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)
