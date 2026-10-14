#!/usr/bin/env python3
"""Fixed event contrasts joining source meaning to historical model actions."""

from __future__ import annotations

from itertools import combinations

import numpy as np


def source_exclusion(source, application, total):
    """Exclude whole application authors, never just their queried comments."""
    s, a = np.asarray(source), np.asarray(application)
    for values in (s, a):
        if (
            values.ndim != 1
            or not np.issubdtype(values.dtype, np.integer)
            or len(set(values.tolist())) != len(values)
            or np.any((values < 0) | (values >= total))
        ):
            raise ValueError("Unique in-range author indices required")
    return s[~np.isin(s, a)]


def unit_vectors(x):
    """Normalize existing embeddings without smoothing or deleting components."""
    x = np.asarray(x, float)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    if x.ndim < 2 or not np.isfinite(x).all() or np.any(n <= 0):
        raise ValueError("Finite nonzero vector observations required")
    return x / n


def recover_controls(vectors, poles, stored_contrast, k=4, tolerance=2e-12):
    """Restore historical identities from input geometry, never native outputs."""
    v, p = unit_vectors(vectors), unit_vectors(poles)
    if v.ndim != 2 or p.shape != (2, v.shape[1]) or len(v) != 3 * k:
        raise ValueError("Fixed12-event/two-pole observation required")
    if not np.isfinite(stored_contrast) or tolerance <= 0:
        raise ValueError("Finite saved input-only contrast required")
    signed = v @ (p[1] - p[0])
    selected = np.sort(np.argsort(-np.abs(signed), kind="stable")[:k])
    rest = np.flatnonzero(~np.isin(np.arange(len(v)), selected))
    matches = [
        c
        for c in combinations(rest, k)
        if abs(
            np.abs(signed[selected]).mean()
            - np.abs(signed[list(c)]).mean()
            - stored_contrast
        )
        <= tolerance
    ]
    if len(matches) != 1:
        raise ValueError(f"Historical control must be unique;found{len(matches)}")
    return selected, np.asarray(matches[0]), signed


def event_variants(values, selected, control):
    """Keep original order and identities in five historical evidence conditions."""
    x, s, c = np.asarray(values, float), np.asarray(selected), np.asarray(control)
    if (
        x.ndim != 2
        or len(x) != 12
        or not np.isfinite(x).all()
        or s.shape != (4,)
        or c.shape != (4,)
        or not np.issubdtype(s.dtype, np.integer)
        or not np.issubdtype(c.dtype, np.integer)
        or len(set(s.tolist()) | set(c.tolist())) != 8
        or np.any((np.r_[s, c] < 0) | (np.r_[s, c] >= 12))
    ):
        raise ValueError("Require12 finite events and disjoint4-event subsets")
    ids = np.arange(12)
    groups = {
        "full": ids,
        "remove_selected": ids[~np.isin(ids, s)],
        "remove_control": ids[~np.isin(ids, c)],
        "selected_only": np.sort(s),
        "control_only": np.sort(c),
    }
    return {name: x[ix].mean(0) for name, ix in groups.items()}


def deletion_accounts(values, selected, control):
    """Return exact normalized changes; a removed mean is not a deletion effect."""
    v = event_variants(values, selected, control)
    ds = v["remove_selected"] - v["full"]
    dc = v["remove_control"] - v["full"]
    return v | {
        "selected_removal_delta": ds,
        "control_removal_delta": dc,
        "removal_contrast": ds - dc,
        "subset_contrast": v["selected_only"] - v["control_only"],
        "identity_error": np.maximum(
            np.abs(ds - 0.5 * (v["full"] - v["selected_only"])),
            np.abs(dc - 0.5 * (v["full"] - v["control_only"])),
        ),
    }


def pearson(x, y):
    """Return association only for nonconstant paired observations."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 1 or not np.isfinite([x, y]).all():
        raise ValueError("Finite paired vectors required")
    xc, yc = x - x.mean(), y - y.mean()
    den = np.linalg.norm(xc) * np.linalg.norm(yc)
    return None if len(x) < 3 or den == 0 else float(xc @ yc / den)


def alignment(x, y, draws=999, seed=17320261012):
    """Describe differently scaled changes without fitting a calibration slope."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = pearson(x, y)
    den = np.linalg.norm(x) * np.linalg.norm(y)
    eligible = (x != 0) & (y != 0)
    boot = []
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        ix = rng.integers(len(x), size=len(x))
        value = pearson(x[ix], y[ix])
        if value is not None:
            boot.append(value)
    ci = np.quantile(boot, [0.025, 0.975]).tolist() if boot else None
    return {
        "n_authors": len(x),
        "pearson": r,
        "paired_author_bootstrap_ci": ci,
        "valid_bootstrap_draws": len(boot),
        "unscaled_cosine": None if den == 0 else float(x @ y / den),
        "signed_concordant": int(np.sum(x[eligible] * y[eligible] > 0)),
        "signed_discordant": int(np.sum(x[eligible] * y[eligible] < 0)),
        "sign_comparable": int(eligible.sum()),
        "mathematical_zero": int(np.sum(x == 0)),
        "native_zero": int(np.sum(y == 0)),
        "mathematical_mean": float(x.mean()),
        "native_mean": float(y.mean()),
    }
