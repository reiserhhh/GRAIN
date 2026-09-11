#!/usr/bin/env python3
"""Source-disjoint selection and fixed-unit twelve-source coordinate teaching."""

import hashlib

import numpy as np

from . import compiled_compatible_reader as previous

VARIANTS = ("original12_native_coordinates", "original12_compatible_coordinates")
SALT = "GRAIN_NEXT88_SOURCE_DIVERSITY_V1"


def select(material, old_sources, recipients):
    """Choose four per provenance by ID hash only; refuse silent replacement."""
    old_sources, recipients = set(old_sources), set(recipients)
    rows = [r for r in material if r["view"] == 0]
    ids = {r["author_id"] for r in rows}
    if len(ids) != len(rows) or len(old_sources) != 4 or len(recipients) != 8:
        raise ValueError("Unique view0, original4 and recipient8 required")
    if old_sources & recipients or not (old_sources | recipients) <= ids:
        raise ValueError("Disjoint existing memberships must remain present")
    chosen, counts = [], {}
    for group in ("mbti_only", "big5_only"):
        pool = [
            r
            for r in rows
            if r["source_group"] == group
            and r["author_id"] not in old_sources | recipients
        ]
        counts[group] = len(pool)
        if len(pool) < 4:
            raise ValueError("Four unused source authors per provenance required")
        pool.sort(
            key=lambda r: hashlib.sha256(
                f"{SALT}|{r['author_id']}".encode()
            ).hexdigest()
        )
        chosen.extend(pool[:4])
    if sum(counts.values()) != 12 or len(rows) != 24:
        raise ValueError("Frozen24-author source material inventory changed")
    return chosen, counts


def fit(old, points, target):
    """Fit12 values with old4 units/weight/lambda; no target normalization."""
    x, target = np.asarray(points, float), np.asarray(target, float)
    if x.shape != (12, old["x"].shape[1]) or target.shape != (12, 5):
        raise ValueError("Exactly12 source positions and five value coordinates")
    if not np.isfinite(x).all() or not np.isfinite(target).all():
        raise ValueError("Finite source-only values required")
    np.testing.assert_array_equal(x[:4], old["x"][previous.ORIGINAL])
    oldw = old["weights"][previous.ORIGINAL]
    np.testing.assert_array_equal(oldw, np.repeat(oldw[0], 4))
    w = np.repeat(oldw[0], 12)
    v, kinds = np.zeros_like(x), np.zeros(12, bool)
    k = previous.base.gram(x, v, kinds, x, v, kinds)
    lam = float(old["regularizer"])
    a = k + np.diag(lam / w**2)
    beta = np.linalg.solve(a, target)
    error = np.linalg.norm(a @ beta - target) / max(
        np.linalg.norm(target), np.finfo(float).tiny
    )
    if error > 1e-9 or not np.isfinite(beta).all():
        raise ValueError("Source coordinate solve failed")
    return {
        "x": x,
        "v": v,
        "kinds": kinds,
        "weights": w,
        "target": target,
        "fitted": k @ beta,
        "beta": beta,
        "d0": np.zeros(5),
        "regularizer": np.asarray(lam),
        "normal_residual_relative": np.asarray(error),
        "regularized_condition": np.asarray(np.linalg.cond(a)),
        "coefficients": np.asarray(beta.size),
        "moving": np.asarray(True),
        "units": old["units"],
        "fixed_frame": old["fixed_frame"],
    }
