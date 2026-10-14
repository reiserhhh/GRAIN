#!/usr/bin/env python3
"""Source-author-linked continuous evidence fields with a fixed observation law."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

TARGETS = ("E", "N", "A", "C", "O")
SPACES = ("GRAIN15", "E5")
FAMILIES = ("window_mean", "comment_mixture")


def projection(directions):
    """Preserve a nonorthogonal observation's inherited Gaussian covariance."""
    d = np.asarray(directions, float)
    if d.ndim != 2 or not np.isfinite(d).all() or d.shape[1] > d.shape[0]:
        raise ValueError("A finite full-column-rank observation map is required")
    chol = np.linalg.cholesky(d.T @ d)
    return np.linalg.solve(chol, d.T).T


def validate(source, codes, anchors, variance):
    """Require complete author-linked numeric prototypes and source-only targets."""
    z, c, a = np.asarray(source, float), np.asarray(codes), np.asarray(anchors, float)
    if (
        z.ndim != 3
        or min(z.shape) < 1
        or c.shape != (len(z),)
        or c.dtype.kind not in "iu"
        or not np.isin(c, np.arange(32)).all()
        or a.shape != (len(z), 5)
        or not np.isfinite(z).all()
        or not np.isfinite(a).all()
        or np.any((a < 0) | (a > 1))
        or not np.isfinite(variance)
        or variance <= 0
    ):
        raise ValueError("Finite source authors/prototypes/five targets required")
    bits = (c[:, None] >> np.arange(5)) & 1
    if np.any((bits.mean(0) == 0) | (bits.mean(0) == 1)):
        raise ValueError("Both source sides of every target must be observed")
    return z, c, a, bits


def author_log_law(points, source, variance):
    """Average Gaussian observations within each author, not across target groups."""
    x, z = np.asarray(points, float), np.asarray(source, float)
    if (
        x.ndim != 2
        or z.ndim != 3
        or x.shape[1] != z.shape[2]
        or min(z.shape) < 1
        or not np.isfinite(x).all()
        or not np.isfinite(z).all()
        or not np.isfinite(variance)
        or variance <= 0
    ):
        raise ValueError(
            "Finite matched vector observations and positive variance required"
        )
    flat = z.reshape(-1, z.shape[2])
    distance = np.sum(x * x, 1)[:, None] + np.sum(flat * flat, 1)[None] - 2 * x @ flat.T
    log_kernel = (-distance / (2 * variance)).reshape(len(x), len(z), z.shape[1])
    return logsumexp(log_kernel, -1) - np.log(z.shape[1])


def readings(log_law, codes, anchors):
    """Keep joint profiles, probability, continuous anchor and prior-free evidence."""
    ell, c, a = (
        np.asarray(log_law, float),
        np.asarray(codes),
        np.asarray(anchors, float),
    )
    if (
        ell.ndim != 2
        or c.shape != (ell.shape[1],)
        or c.dtype.kind not in "iu"
        or not np.isin(c, np.arange(32)).all()
        or a.shape != (len(c), 5)
        or not np.isfinite(ell).all()
        or not np.isfinite(a).all()
    ):
        raise ValueError("Finite author likelihoods and source targets required")
    bits = (c[:, None] >> np.arange(5)) & 1
    prior = bits.mean(0)
    if np.any((prior == 0) | (prior == 1)):
        raise ValueError("Both target sides must have source support")
    weight = np.exp(ell - logsumexp(ell, -1, keepdims=True))
    profile = np.stack([weight[:, c == g].sum(1) for g in range(32)], -1)
    evidence = np.column_stack(
        [
            logsumexp(ell[:, bits[:, q] == 1], -1)
            - np.log(np.count_nonzero(bits[:, q]))
            - logsumexp(ell[:, bits[:, q] == 0], -1)
            + np.log(np.count_nonzero(1 - bits[:, q]))
            for q in range(5)
        ]
    )
    return {
        "probability": weight @ bits,
        "anchor_mean": weight @ a,
        "log_Bayes_factor": evidence,
        "profile_mass": profile,
        "effective_source_authors": 1 / np.sum(weight * weight, 1),
    }


def score_events(bundle, points):
    """Score fixed numerical events with both source families and observation spaces."""
    x = np.asarray(points, float)
    z, c, a, _ = validate(
        bundle["source_comments"],
        bundle["codes"],
        bundle["anchors"],
        bundle["variance"],
    )
    if x.ndim != 2 or x.shape[1] != z.shape[2]:
        raise ValueError("The frozen input vector width must match")
    q = projection(bundle["directions"])
    result = {}
    for name in SPACES:
        obs, source = (x @ q, z @ q) if name == "GRAIN15" else (x, z)
        m = source.shape[1]
        if m % 2:
            raise ValueError("Exactly two equal source windows are required")
        prototypes = {
            "window_mean": source.reshape(len(source), 2, m // 2, -1).mean(2),
            "comment_mixture": source,
        }
        for family in FAMILIES:
            law = author_log_law(obs, prototypes[family], bundle["variance"])
            result[f"{name}/{family}"] = readings(law, c, a)
    return result


def cold_record(bundle, record):
    """Exclude answers/labels and repeated evidence; aggregate under one source author."""
    if set(record) != {"frame_key", "events"} or not isinstance(
        record["frame_key"], str
    ):
        raise ValueError("Only frame identity and addressed vectors are allowed")
    seen, vectors = set(), []
    for event in record["events"]:
        if set(event) != {"event_id", "vector"}:
            raise ValueError("No application labels or model answers are allowed")
        identity = event["event_id"]
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("Each evidence event must have a distinct identity")
        seen.add(identity)
        vectors.append(event["vector"])
    if not vectors:
        raise ValueError("Missing text is not a neutral observation")
    x = np.asarray(vectors, float)
    z, c, a, _ = validate(
        bundle["source_comments"],
        bundle["codes"],
        bundle["anchors"],
        bundle["variance"],
    )
    if x.ndim != 2 or x.shape[1] != z.shape[2] or not np.isfinite(x).all():
        raise ValueError("Finite vectors in the frozen encoder coordinates required")
    q = projection(bundle["directions"])
    result = {}
    for name in SPACES:
        obs, source = (x @ q, z @ q) if name == "GRAIN15" else (x, z)
        if source.shape[1] % 2:
            raise ValueError("Exactly two equal source windows are required")
        for family in FAMILIES:
            prototypes = (
                source.reshape(len(source), 2, source.shape[1] // 2, -1).mean(2)
                if family == "window_mean"
                else source
            )
            # One shared source-author index across distinct events, not products of marginals.
            joint = author_log_law(obs, prototypes, bundle["variance"]).sum(
                0, keepdims=True
            )
            result[f"{name}/{family}"] = {
                k: v[0].tolist() for k, v in readings(joint, c, a).items()
            }
    return {
        "frame_key": record["frame_key"],
        "event_ids": sorted(seen),
        "readings": result,
    }
