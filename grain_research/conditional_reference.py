#!/usr/bin/env python3
"""Frozen condition profiles and joint vector-completion envelopes."""

from __future__ import annotations

from itertools import combinations

import numpy as np


def cell_means(conditions, scores):
    """Keep all observed cells with their counts and finite evidence means."""
    c, x = np.asarray(conditions, str), np.asarray(scores, float)
    if c.ndim != 1 or x.ndim != 2 or len(c) != len(x) or not len(c):
        raise ValueError("Aligned nonempty conditions and evidence required")
    if not np.isfinite(x).all() or np.any(c == ""):
        raise ValueError("Missing condition/evidence must be resolved explicitly")
    return {key: (int(np.sum(c == key)), x[c == key].mean(0)) for key in sorted(set(c))}


def fit_reference(conditions, scores, authors, source):
    """Learn author-balanced cell frequencies and cell means from source only."""
    authors, source = np.asarray(authors), np.asarray(source)
    conditions, scores = np.asarray(conditions, str), np.asarray(scores, float)
    if len(authors) != len(conditions) or len(authors) != len(scores):
        raise ValueError("Unit address mismatch")
    if len(source) == 0 or len(set(source)) != len(source):
        raise ValueError("Unique nonempty source authors required")
    masses, profiles = {}, {}
    for uid in source:
        ix = authors == uid
        if not ix.any():
            raise ValueError("Source author has no observations")
        cells = cell_means(conditions[ix], scores[ix])
        total = int(ix.sum())
        for key, (n, mean) in cells.items():
            masses[key] = masses.get(key, 0.0) + n / total / len(source)
            profiles.setdefault(key, []).append(mean)
    keys = sorted(masses)
    weights = np.array([masses[k] for k in keys])
    weights /= weights.sum()
    counts = np.array([len(profiles[k]) for k in keys])
    top = sorted(range(len(keys)), key=lambda i: (-counts[i], keys[i]))[:5]
    core = np.zeros_like(weights)
    core[top] = weights[top] / weights[top].sum()
    return {
        "keys": keys,
        "weights": weights,
        "common5_weights": core,
        "mean": np.vstack([np.mean(profiles[k], axis=0) for k in keys]),
        "source_authors_per_cell": counts,
        "source_indices": source.copy(),
    }


def score_reference(conditions, scores, reference, target="full"):
    """Return actual, conditional-departure and partially observed fixed scores."""
    if target not in ("full", "common5"):
        raise ValueError("Unknown fixed target")
    scores = np.asarray(scores, float)
    cells = cell_means(conditions, scores)
    lookup = {k: i for i, k in enumerate(reference["keys"])}
    nu = reference["weights" if target == "full" else "common5_weights"]
    width = scores.shape[1]
    context, departure, unseen, center = [np.zeros(width) for _ in range(4)]
    observed = np.zeros(len(nu), bool)
    unknown_mass, selections = 0.0, []
    for key, (n, mean) in cells.items():
        mass = n / len(scores)
        known = key in lookup
        selections.append(
            {"condition": key, "count": n, "fraction": mass, "reference_known": known}
        )
        if known:
            j = lookup[key]
            context += mass * reference["mean"][j]
            departure += mass * (mean - reference["mean"][j])
            center += nu[j] * mean
            observed[j] = True
        else:
            unseen += mass * mean
            unknown_mass += mass
    coverage = float(nu[observed].sum())
    missing = max(0.0, float(nu[~observed].sum()))
    complete = missing <= 1e-14
    reference_mean = nu @ reference["mean"]
    return {
        "actual": scores.mean(0),
        "context": context,
        "departure": departure,
        "unseen": unseen,
        "center": center,
        "reference_target_mean": reference_mean,
        "point": center.copy() if complete else None,
        "supported_target": center / coverage if coverage > 0 else None,
        "coverage": coverage,
        "missing": missing,
        "observed": observed,
        "unknown_material_mass": unknown_mass,
        "complete": complete,
        "selection": selections,
        "missing_cells_ranked": sorted(
            [i for i in range(len(nu)) if not observed[i] and nu[i] > 0],
            key=lambda i: (-nu[i], reference["keys"][i]),
        ),
    }


def contrasts(facets):
    """Fixed coordinate, domain-mean and domain-difference targets; no label use."""
    facets = list(facets)
    if len(facets) != 30 or len(set(facets)) != 30:
        raise ValueError("Thirty distinct facet reference keys required")
    directions = [v for v in np.eye(30)]
    names = facets.copy()
    domain = {}
    for key in "ACENO":
        v = np.array([float(f.startswith(key)) for f in facets])
        if v.sum() != 6:
            raise ValueError("Six facets per domain required")
        domain[key] = v / v.sum()
        names.append(f"mean_{key}")
        directions.append(domain[key])
    for a, b in combinations(domain, 2):
        names.append(f"mean_{a}_minus_{b}")
        directions.append(domain[a] - domain[b])
    return np.column_stack(directions), names


def completion_envelope(center, missing, anchor_directions, targets):
    """Support function of the linear image of a unit-ball missing-cell class."""
    d, t = np.asarray(anchor_directions, float), np.asarray(targets, float)
    a = np.asarray(center, float)
    if d.ndim != 2 or t.ndim != 2 or a.shape != (d.shape[1],) or t.shape[0] != len(a):
        raise ValueError("Aligned evidence and target coordinates required")
    if not 0 <= missing <= 1 + 1e-12 or not all(
        np.isfinite(x).all() for x in (d, t, a)
    ):
        raise ValueError("Finite maps and valid missing mass required")
    mid = a @ t
    halfwidth = missing * np.linalg.norm(d @ t, axis=0)
    box_halfwidth = missing * (np.linalg.norm(d, axis=0) @ abs(t))
    return mid - halfwidth, mid + halfwidth, halfwidth, box_halfwidth
