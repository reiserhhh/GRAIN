#!/usr/bin/env python3
"""Source-observed conditional functions and author-relative numeric measurements.

Histories are continuous anonymous GRAIN coordinates, not exact toy prototypes.
Outcomes must be distinct, observed later records. This finite partition makes
no claim to identify an unrestricted text-to-personality function.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

VERSION = "HM170_observed_author_family_v1"
INDICES = (0, 5)
TARGETS = (
    "next_record_same_venue",
    "next_author_coordinate_high",
    "next_response_coordinate_high",
)


def code_hash():
    """Bind the cold observation rule to its scientific implementation."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def states(coordinates, thresholds):
    """Apply two frozen source cutoffs to arbitrary finite15 coordinates."""
    x, cut = np.asarray(coordinates, float), np.asarray(thresholds, float)
    if x.ndim != 2 or x.shape[1] != 15 or cut.shape != (2,):
        raise ValueError("Anonymous15 coordinates and two frozen cutoffs required")
    if not np.isfinite(x).all() or not np.isfinite(cut).all():
        raise ValueError("Nonfinite observation")
    bits = x[:, INDICES] > cut
    return bits[:, 0].astype(int) + 2 * bits[:, 1].astype(int)


def compile_law(coordinates, conditions, outcomes, authors):
    """Estimate a finite conditional law; every source author has equal mass."""
    x, c, y, ids = (
        np.asarray(coordinates, float),
        np.asarray(conditions),
        np.asarray(outcomes, float),
        np.asarray(authors),
    )
    if (
        x.ndim != 2
        or x.shape[1] != 15
        or c.shape != (len(x),)
        or y.shape != (len(x), 3)
        or ids.shape != (len(x),)
        or len(x) == 0
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.isin(c, (0, 1)).all()
        or not np.isin(y, (0, 1)).all()
    ):
        raise ValueError(
            "Aligned source history/condition/observed binary outcomes required"
        )
    c = c.astype(int)
    cut = np.median(x[:, INDICES], axis=0)
    s = states(x, cut)
    _, ix, counts = np.unique(ids, return_inverse=True, return_counts=True)
    weight = 1.0 / (len(counts) * counts[ix])
    support, mass, positives = (
        np.zeros((2, 4), int),
        np.zeros((2, 4)),
        np.zeros((2, 4, 3)),
    )
    np.add.at(support, (c, s), 1)
    np.add.at(mass, (c, s), weight)
    np.add.at(positives, (c, s), weight[:, None] * y)
    probability = np.divide(
        positives,
        mass[:, :, None],
        out=np.full((2, 4, 3), np.nan),
        where=mass[:, :, None] > 0,
    )
    c_mass = mass.sum(1)
    c_positive = positives.sum(1)
    conditional = np.divide(
        c_positive,
        c_mass[:, None],
        out=np.full((2, 3), np.nan),
        where=c_mass[:, None] > 0,
    )
    return {
        "version": VERSION,
        "code_hash": code_hash(),
        "targets": list(TARGETS),
        "coordinate_indices": list(INDICES),
        "thresholds": cut.tolist(),
        "source_authors": len(counts),
        "source_events": len(x),
        "support": support.tolist(),
        "mass": mass.tolist(),
        "positive_mass": positives.tolist(),
        "probabilities": [
            [[None if not np.isfinite(v) else float(v) for v in row] for row in group]
            for group in probability
        ],
        "condition_reference": [
            [None if not np.isfinite(v) else float(v) for v in row]
            for row in conditional
        ],
        "reference": c_positive.sum(0).tolist(),
        "regime": "self_selected_sampled_publications",
    }


def verify_law(bundle):
    """Verify probability units, support and the fixed source reference."""
    fields = {
        "version",
        "code_hash",
        "targets",
        "coordinate_indices",
        "thresholds",
        "source_authors",
        "source_events",
        "support",
        "mass",
        "positive_mass",
        "probabilities",
        "condition_reference",
        "reference",
        "regime",
    }
    if (
        set(bundle) != fields
        or bundle["version"] != VERSION
        or bundle["code_hash"] != code_hash()
        or bundle["targets"] != list(TARGETS)
        or bundle["coordinate_indices"] != list(INDICES)
        or bundle["regime"] != "self_selected_sampled_publications"
    ):
        raise ValueError("Unknown rule, target, regime or latent/teacher field")
    m, pos, p, mu, ref, support = (
        np.asarray(bundle[k], float)
        for k in (
            "mass",
            "positive_mass",
            "probabilities",
            "condition_reference",
            "reference",
            "support",
        )
    )
    if (
        m.shape != (2, 4)
        or pos.shape != (2, 4, 3)
        or p.shape != (2, 4, 3)
        or mu.shape != (2, 3)
        or ref.shape != (3,)
        or support.shape != (2, 4)
        or not all(np.isfinite(v).all() for v in (m, pos, ref, support))
        or np.any(m < 0)
        or np.any(pos < 0)
        or np.any(pos > m[:, :, None] + 1e-12)
        or np.any(support < 0)
        or not np.isclose(m.sum(), 1)
        or not np.array_equal(support, np.floor(support))
        or bundle["source_authors"] <= 0
        or not np.array_equal(m > 0, support > 0)
        or not np.array_equal(np.isfinite(p), np.repeat((m > 0)[:, :, None], 3, axis=2))
        or np.any((p < 0) | (p > 1))
        or not np.allclose(ref, pos.sum(axis=(0, 1)))
        or int(support.sum()) != bundle["source_events"]
    ):
        raise ValueError("Inconsistent source law")
    for c in range(2):
        cells = m[c] > 0
        if cells.any():
            if not np.allclose(p[c, cells] * m[c, cells, None], pos[c, cells]):
                raise ValueError("Probability/positive-mass disagreement")
            if not np.allclose(mu[c], pos[c].sum(0) / m[c].sum()):
                raise ValueError("Condition reference disagreement")
        elif np.isfinite(mu[c]).any():
            raise ValueError("Unobserved condition cannot have a reference")
    states(np.zeros((0, 15)), bundle["thresholds"])


def event_probabilities(bundle, coordinates, conditions):
    """Read supported probabilities without future outcomes or application fit."""
    verify_law(bundle)
    c = np.asarray(conditions)
    s = states(coordinates, bundle["thresholds"])
    if c.shape != s.shape or not np.isin(c, (0, 1)).all():
        raise ValueError("Unknown recorded-condition address")
    c = c.astype(int)
    return np.asarray(bundle["probabilities"], float)[c, s]


def cold_author(bundle, record):
    """Compose an author window, retaining choice and conditional composition."""
    verify_law(bundle)
    if set(record) != {"author_key", "events"}:
        raise ValueError("Cold input excludes outcomes, scales and latent truth")
    event_ids = [e.get("event_id") for e in record["events"]]
    if len(set(event_ids)) != len(event_ids) or any(
        not isinstance(v, str) for v in event_ids
    ):
        raise ValueError("Distinct string event identities required")
    for e in record["events"]:
        if set(e) != {"event_id", "condition", "coordinates"}:
            raise ValueError("Current event fields only")
    n = len(record["events"])
    result = {"author_key": record["author_key"], "events": n, "targets": list(TARGETS)}
    if not n:
        return result | {"status": "NO_OBSERVATIONS", "mean": None, "relative": None}
    x = np.asarray([e["coordinates"] for e in record["events"]], float)
    c = np.asarray([e["condition"] for e in record["events"]])
    s = states(x, bundle["thresholds"])
    p = event_probabilities(bundle, x, c)
    if not np.isfinite(p).all():
        missing = sorted(
            {
                (int(a), int(b))
                for a, b, row in zip(c, s, p, strict=True)
                if not np.isfinite(row).all()
            }
        )
        return result | {
            "status": "UNOBSERVED_SOURCE_CELLS",
            "missing_cells": missing,
            "mean": None,
            "relative": None,
        }
    c = c.astype(int)
    empirical = np.zeros((2, 4))
    np.add.at(empirical, (c, s), 1.0 / n)
    mu, mass, ref = (
        np.asarray(bundle[k], float)
        for k in ("condition_reference", "mass", "reference")
    )
    used = mass.sum(1) > 0
    choice = (empirical.sum(1)[used] - mass.sum(1)[used]) @ mu[used]
    within = np.mean(p - mu[c], axis=0)
    return result | {
        "status": "OBSERVED_SOURCE_FUNCTION",
        "empirical_mass": empirical.tolist(),
        "mean": p.mean(0).tolist(),
        "relative": (p.mean(0) - ref).tolist(),
        "choice": choice.tolist(),
        "within": within.tolist(),
        "reference": ref.tolist(),
        "condition_share": empirical.sum(1).tolist(),
    }
