#!/usr/bin/env python3
"""Retain author and within-author sampling in a finite target-reader update."""

import numpy as np

from grain_research.construction import math_profile_reader as M
from grain_research.construction import math_vector_lift as V

LAWS = ("original", "author_only", "evidence_only", "joint")
VERSION = "HM183_source_law_update_v1"


def disintegrate(joint):
    """Factor a finite positive author/evidence law without smoothing support."""
    p = np.asarray(joint, float)
    if p.ndim != 2 or not np.isfinite(p).all() or np.min(p) < 0:
        raise ValueError("Nonnegative finite author/evidence probabilities required")
    a = p.sum(1)
    if np.min(a) <= 0 or abs(a.sum() - 1) > 1e-12:
        raise ValueError("Positive author masses and a normalized law required")
    return a, p / a[:, None]


def four_laws(original, updated):
    """Return actual joint law and the two registered sampling-level controls."""
    a, b = disintegrate(original)
    aa, bb = disintegrate(updated)
    if b.shape != bb.shape:
        raise ValueError("Unchanged author/event support required")
    laws = np.stack((original, aa[:, None] * b, a[:, None] * bb, updated))
    interaction = (aa - a)[:, None] * (bb - b)
    return laws, interaction


def effective_features(phi, laws):
    """Convert probability-weighted evidence to fixed-profile mean contributions."""
    p, laws = np.asarray(phi, float), np.asarray(laws, float)
    if p.ndim != 3 or laws.ndim != 3 or laws.shape[1:] != p.shape[:2]:
        raise ValueError("Aligned laws and author/triple features required")
    return len(p) * np.einsum("jut,utk->juk", laws, p)


def contrast_change(moments, prevalence):
    """Recover marginal high-low change including its unconditional moment."""
    w, pi = np.asarray(moments, float), np.asarray(prevalence, float)
    if w.shape[-2] != 6 or pi.shape != (5,) or np.any((pi <= 0) | (pi >= 1)):
        raise ValueError("All five target moments and interior prevalences required")
    return (w[..., 1:, :] + (1 - 2 * pi)[..., None] * w[..., :1, :]) / (
        2 * pi * (1 - pi)
    )[..., None]


def cold_record(fields, record):
    """Read a source-law family from coordinates only,never application targets."""
    f = np.asarray(fields, float)
    if f.shape != (2, 6, 4, 5, 4, 4096) or not np.isfinite(f).all():
        raise ValueError("Full frozen target-reader family required")
    if set(record) != {"frame_key", "view", "events"} or record["view"] not in (0, 1):
        raise ValueError("Unexpected cold fields")
    ids, slots, x = [], [], []
    for e in record["events"]:
        if set(e) != {"event_id", "slot", "coordinates"} or type(e["slot"]) is not int:
            raise ValueError("Only addressed numeric evidence is permitted")
        if not isinstance(e["event_id"], str):
            raise TypeError("Opaque string event identity required")
        ids.append(e["event_id"])
        slots.append(e["slot"])
        x.append(e["coordinates"])
    if len(set(ids)) != len(ids):
        raise ValueError("Repeated evidence event")
    phi, n, _ = V.subset_feature(np.asarray(x).reshape(-1, 15), np.asarray(slots, int))
    value = np.einsum("vajqck,k->vajqc", f, phi)
    return value, n


def readers_from_features(features, labels):
    """Construct every marginal/matched/composition field at fixed source counts."""
    return np.stack([M.construct(x, labels)["fields"] for x in features])
