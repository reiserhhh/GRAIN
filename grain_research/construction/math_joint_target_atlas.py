#!/usr/bin/env python3
"""Finite target-profile observation atlas with crossed author/text families."""

import hashlib

import numpy as np

VERSION = "HM185_joint_target_atlas_v1"
TARGETS = ("E", "N", "A", "C", "O")


def profile_codes(labels):
    """Retain the complete inherited five-bit profile, without recutting labels."""
    y = np.asarray(labels)
    if y.ndim != 2 or y.shape[1] != 5 or not np.isin(y, (0, 1)).all():
        raise ValueError("Five binary inherited source labels required")
    return (y @ (2 ** np.arange(5))).astype(int)


def characters():
    """Return all Z2^5 characters under the declared uniform profile reference."""
    signs = 2 * ((np.arange(32)[:, None] >> np.arange(5)) & 1) - 1
    h = np.ones((32, 32))
    for mask in range(1, 32):
        h[:, mask] = signs[:, [q for q in range(5) if mask & (1 << q)]].prod(1)
    return h


def author_folds(codes, indices, salt):
    """Split once within profiles; keep every observed author and profile side."""
    c, ids = np.asarray(codes), np.asarray(indices)
    if c.shape != ids.shape or len(set(ids.tolist())) != len(ids):
        raise ValueError("Unique source indices required")
    folds = np.empty(len(c), int)
    for g in range(32):
        positions = np.flatnonzero(c == g)
        if len(positions) < 2:
            raise ValueError("Both author folds need every source profile")
        ordered = sorted(
            positions,
            key=lambda i: hashlib.sha256(f"{salt}:{int(ids[i])}".encode()).digest(),
        )
        folds[ordered] = np.arange(len(ordered)) % 2
    return folds


def profile_weights(codes, folds=None):
    """Build exact author-balanced cell means, not a regression on model answers."""
    c = np.asarray(codes, int)
    if c.ndim != 1 or not np.isin(c, np.arange(32)).all():
        raise ValueError("Valid profile codes required")
    groups = (np.ones(len(c), bool),) if folds is None else (folds == 0, folds == 1)
    weights, counts = [], []
    for group in groups:
        n = np.bincount(c[group], minlength=32)
        if np.min(n) == 0:
            raise ValueError("Missing profile; no smoothing or neutral imputation")
        w = np.zeros((32, len(c)))
        for g in range(32):
            w[g, group & (c == g)] = 1 / n[g]
        weights.append(w)
        counts.append(n)
    return np.asarray(weights), np.asarray(counts)


def construct(means, codes, folds=None):
    """Compile complete vector-valued harmonics from observed source cell means."""
    x = np.asarray(means, float)
    if x.ndim != 3 or len(x) != len(codes) or not np.isfinite(x).all():
        raise ValueError("Finite source-author by text-view by vector means required")
    w, n = profile_weights(codes, folds)
    cells = np.einsum("bgu,uvd->bvgd", w, x)
    fields = np.einsum("gm,bvgd->bvmd", characters() / 32, cells)
    return {"fields": fields, "cells": cells, "counts": n, "weights": w}


def edge_matrix():
    """Read all five families of conditional edges from the SAME coefficient bank."""
    h = characters()
    e = np.empty((5, 16, 32))
    for q in range(5):
        others = [i for i in range(5) if i != q]
        for b in range(16):
            lo = sum(((b >> j) & 1) << k for j, k in enumerate(others))
            e[q, b] = h[lo | (1 << q)] - h[lo]
    return e


def orders():
    """Return order tags, without assigning psychological construct names."""
    return np.array([m.bit_count() for m in range(32)])


def cross_author_statistics(fields):
    """Keep signed cross-family agreements; no positivity clipping or axis alignment."""
    f = np.asarray(fields, float)
    if f.ndim != 5 or f.shape[:3] != (2, 2, 32) or f.shape[3] != 3:
        raise ValueError("Two author folds/two views/32modes/three spaces required")
    dots = np.einsum("vmsd,wmsd->ms", f[0], f[1]) / 4
    rows = []
    groups = [(str(k), orders() == k) for k in range(1, 6)]
    groups += [("joint", orders() >= 2), ("all", orders() >= 1)]
    for name, keep in groups:
        for space in range(3):
            a, b = f[0, :, keep, space], f[1, :, keep, space]
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            same = (
                sum(np.sum(f[0, v, keep, space] * f[1, v, keep, space]) for v in (0, 1))
                / 2
            )
            opposite = (
                sum(
                    np.sum(f[0, v, keep, space] * f[1, 1 - v, keep, space])
                    for v in (0, 1)
                )
                / 2
            )
            within = (
                sum(
                    np.sum(f[bk, 0, keep, space] * f[bk, 1, keep, space])
                    for bk in (0, 1)
                )
                / 2
            )
            rows.append(
                {
                    "order": name,
                    "space": space,
                    "signed_inner_product": float(dots[keep, space].sum()),
                    "cross_family_coherence": float(
                        2 * dots[keep, space].sum() / denominator
                    )
                    if denominator
                    else None,
                    "same_view_cross_author": float(same),
                    "different_view_cross_author": float(opposite),
                    "within_author_fold_cross_view": float(within),
                }
            )
    return rows


def permute_profiles(codes, folds, rng):
    """Exchange full profile assignments within each fixed author family."""
    out = np.asarray(codes).copy()
    for fold in (0, 1):
        ids = np.flatnonzero(folds == fold)
        out[ids] = rng.permutation(out[ids])
    return out


def cold_record(fields, semantic, record):
    """Read joint coordinates and conditional functions without new author labels."""
    f, s = np.asarray(fields, float), np.asarray(semantic, float)
    if f.shape != (2, 32, 3, 1024) or s.shape != (2, 3, 1024):
        raise ValueError("Complete fixed source/semantic reader family required")
    if not np.isfinite(f).all() or not np.isfinite(s).all():
        raise ValueError("Finite cold reader coefficients required")
    if set(record) != {"frame_key", "events"} or not isinstance(
        record["frame_key"], str
    ):
        raise ValueError("Only numeric addressed events permitted")
    ids, slots, x = [], [], []
    for event in record["events"]:
        if set(event) != {"event_id", "slot", "vector"}:
            raise ValueError("Unexpected evidence fields")
        if (
            not isinstance(event["event_id"], str)
            or type(event["slot"]) is not int
            or event["slot"] not in range(12)
        ):
            raise ValueError("Original event identity and slot required")
        ids.append(event["event_id"])
        slots.append(event["slot"])
        x.append(event["vector"])
    if len(set(ids)) != len(ids) or len(set(slots)) != len(slots):
        raise ValueError("Repeated evidence event")
    if not ids:
        return {
            "frame_key": record["frame_key"],
            "count": 0,
            "modes": None,
            "edges": None,
            "semantic": None,
        }
    x = np.asarray(x, float)
    if x.shape != (len(ids), 1024) or not np.isfinite(x).all():
        raise ValueError("Finite1024-dimensional observations required")
    mean = x.mean(0)
    values = np.einsum("vmsd,d->vms", f, mean)
    return {
        "frame_key": record["frame_key"],
        "count": len(ids),
        "modes": values.tolist(),
        "edges": np.einsum("qbm,vms->vqbs", edge_matrix(), values).tolist(),
        "semantic": np.einsum("qsd,d->qs", s, mean).tolist(),
    }
