#!/usr/bin/env python3
"""Signed item configurations retain the averaged key and its orthogonal remainder."""

from __future__ import annotations

import numpy as np

from grain_research.construction import math_output_observability as O


def item_frame(responses, bank, target):
    """Use the old balanced key to define fixed item coordinates,never fit a basis."""
    x = np.asarray(responses, float)
    if x.ndim < 2 or x.shape[-1] != 50 or not np.isfinite(x).all():
        raise ValueError("Every complete finite50-item response required")
    if type(target) is not int or target not in range(5):
        raise ValueError("A declared target address required")
    key = O.key_matrix(bank)[target]
    ids = np.flatnonzero(key)
    direction = np.sqrt(abs(key[ids]))
    frame = x[..., ids] * np.sign(key[ids]) * direction
    return frame, direction, ids


def split_frame(frame, direction):
    """Preserve full,key and shape; the complement is not text noise deletion."""
    z, v = np.asarray(frame, float), np.asarray(direction, float)
    if (
        z.ndim < 2
        or v.shape != (z.shape[-1],)
        or not np.isfinite(z).all()
        or not np.isfinite(v).all()
        or abs(v @ v - 1) > 1e-12
    ):
        raise ValueError("Finite item coordinates and unchanged unit keyed direction")
    keyed = (z @ v)[..., None] * v
    return np.stack((z, keyed, z - keyed), axis=0)


def kernels(left, right, direction):
    """Return three cross-window Gram matrices,retaining every sign."""
    a, b = np.asarray(left, float), np.asarray(right, float)
    if a.ndim != 2 or b.shape != a.shape or len(a) < 2:
        raise ValueError("Two complete aligned author response frames required")
    x, y = split_frame(a, direction), split_frame(b, direction)
    return np.einsum("pid,pjd->pij", x, y)


def signed_patterns(scores, response_parts):
    """Return source-oriented cross-moments,not fitted model predictions."""
    s, z = np.asarray(scores, float), np.asarray(response_parts, float)
    if s.ndim != 1 or z.ndim != 3 or z.shape[1] != len(s) or not np.isfinite(z).all():
        raise ValueError("SOURCE coefficients and three author-by-item frames")
    a = O.centered_scores(s)
    return np.einsum("i,pid->pd", a, z) / len(a)


def cold(record):
    """Recompute identified pattern accounts without labels,model or fitted probes."""
    expected = {
        "version",
        "author_keys",
        "left_event_ids",
        "right_event_ids",
        "source_readings",
        "item_joint_responses",
        "item_bank",
    }
    if set(record) != expected or record["version"] != "HM286-v1":
        raise ValueError("Only registered numeric observations,no application labels")
    authors = record["author_keys"]
    n = len(authors)
    ids = record["left_event_ids"] + record["right_event_ids"]
    if (
        n < 2
        or len(set(authors)) != n
        or any(not isinstance(a, str) or not a for a in authors)
        or len(record["left_event_ids"]) != n
        or len(record["right_event_ids"]) != n
        or len(ids) != 2 * n
        or any(len(row) != 2 for row in ids)
    ):
        raise ValueError("Unique authors and two identified pairs per author")
    flat = [i for row in ids for i in row]
    if len(set(flat)) != len(flat) or any(
        not isinstance(i, str) or not i for i in flat
    ):
        raise ValueError("No repeated or unidentified observed comment event")
    scores = np.asarray(record["source_readings"], float)
    response = np.asarray(record["item_joint_responses"], float)
    if scores.shape != (2, n, 5) or response.shape != (2, n, 50):
        raise ValueError("Two SOURCE views and complete50-item responses required")
    bank = record["item_bank"]
    patterns = np.empty((2, 5, 5, 3, 10))
    contact = np.empty((5, 5, 3))
    for q in range(5):
        a, b = O.centered_scores(scores[0, :, q]), O.centered_scores(scores[1, :, q])
        for r in range(5):
            z, v, _ = item_frame(response, bank, r)
            k = kernels(z[0], z[1], v)
            for view in range(2):
                patterns[view, q, r] = signed_patterns(
                    scores[view, :, q], split_frame(z[view], v)
                )
            contact[q, r] = [
                O.cross_author_accounts(a, b, part)["distinct_authors"] for part in k
            ]
    return {
        "signed_source_query_item_patterns": patterns.tolist(),
        "distinct_author_source_query_parts": contact.tolist(),
        "source_sign_reversal_reverses_patterns_not_quadratic_witness": True,
        "not_individual_trait_predictions": True,
    }
