#!/usr/bin/env python3
"""Source-supported signed vector observations, without output/precision fitting."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

VERSION = "HM181_actual_vector_observer_v1"
TARGETS = ("E", "N", "A", "C", "O")
OFFSETS = np.indices((4, 4, 4)).reshape(3, -1).T + np.array([0, 4, 8])


def code_hash():
    """Bind the fixed numeric reader independently of source results."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sectors():
    """Index every augmented tensor coefficient by its number of nonconstants."""
    return np.sum(np.indices((16, 16, 16)).reshape(3, -1) > 0, axis=0)


def tensor_features(x):
    """Return all4096 explicit polynomial features on three real GRAIN15 vectors."""
    x = np.asarray(x, float)
    if x.ndim != 3 or x.shape[1:] != (3, 15) or not np.isfinite(x).all():
        raise ValueError("Finite addressed three-by15 coordinates required")
    a = np.concatenate((np.ones(x.shape[:2] + (1,)), x), axis=2)
    return np.einsum("ni,nj,nk->nijk", a[:, 0], a[:, 1], a[:, 2]).reshape(-1, 4096)


def lift_weights(cells, targets, profiles, full, changes):
    """Lift frozen coarse signed measures to actual author/event triple support."""
    cells, targets = np.asarray(cells), np.asarray(targets)
    profiles, full, changes = map(np.asarray, (profiles, full, changes))
    if (
        cells.ndim != 2
        or cells.shape[1] != 12
        or not np.isin(cells, np.arange(8)).all()
        or targets.shape != (len(cells), 5)
        or not np.isin(targets, (0, 1)).all()
        or profiles.ndim != 2
        or profiles.shape[1] != 5
        or full.shape != (len(profiles), 8, 8, 8)
        or changes.shape != (5,) + full.shape
        or not np.isfinite(full).all()
        or not np.isfinite(changes).all()
        or np.any(full < 0)
    ):
        raise ValueError("Invalid source law or addressed profile observations")
    lookup = {tuple(p): j for j, p in enumerate(profiles.tolist())}
    if len(lookup) != len(profiles):
        raise ValueError("Duplicate source profiles")
    groups = np.array([lookup[tuple(y)] for y in targets.tolist()])
    triples = cells[:, OFFSETS]
    flat = ((groups[:, None] * 8 + triples[:, :, 0]) * 8 + triples[:, :, 1]) * 8
    flat += triples[:, :, 2]
    f = full.ravel()[flat]
    if np.any(f <= 0):
        raise ValueError("Actual source triple absent from frozen support")
    ratios = (changes.reshape(5, -1)[:, flat] / f).transpose(1, 2, 0)
    weights = ratios / (len(cells) * 64)
    lifted = (1 - ratios.sum(2)) / (len(cells) * 64)
    if np.min(lifted) < 0:
        raise ValueError("Frozen action has no nonnegative support-preserving lift")
    return weights, lifted, flat, triples


def source_moments(coordinates, embeddings, cells, targets, profiles, full, changes):
    """Compile complete signed target/vector moments, retaining all tensor sectors."""
    x, h = np.asarray(coordinates, float), np.asarray(embeddings, float)
    if (
        x.shape != (len(cells), 12, 15)
        or h.shape != (len(cells), 12, 1024)
        or not np.isfinite(x).all()
        or not np.isfinite(h).all()
    ):
        raise ValueError("Actual cached GRAIN15/E5 coordinates required")
    weights, lifted, flat, triples = lift_weights(
        cells, targets, profiles, full, changes
    )
    moments, e5, coarse = (
        np.zeros((5, 6, 4096)),
        np.zeros((5, 6, 3, 1024)),
        np.zeros((5, 6, 512)),
    )
    signs = np.column_stack((np.ones(len(targets)), 2 * np.asarray(targets, float) - 1))
    for u in range(len(cells)):
        phi = tensor_features(x[u, OFFSETS])
        wphi = weights[u].T @ phi
        moments += wphi[:, None, :] * signs[u][None, :, None]
        role_h = h[u, OFFSETS].reshape(64, -1)
        wh = (weights[u].T @ role_h).reshape(5, 3, 1024)
        e5 += wh[:, None, :, :] * signs[u][None, :, None, None]
        abc = (triples[u, :, 0] * 8 + triples[u, :, 1]) * 8 + triples[u, :, 2]
        for q in range(5):
            for r in range(6):
                np.add.at(coarse[q, r], abc, weights[u, :, q] * signs[u, r])
    return {
        "weights": weights,
        "lifted": lifted,
        "flat": flat,
        "triples": triples,
        "moments": moments,
        "e5": e5,
        "coarse": coarse,
    }


def subset_feature(coordinates, slots):
    """Keep original-frame opportunity; empty roles do not become neutral traits."""
    x, slots = np.asarray(coordinates, float), np.asarray(slots)
    if (
        x.shape != (len(slots), 15)
        or not np.isfinite(x).all()
        or not np.issubdtype(slots.dtype, np.integer)
        or len(np.unique(slots)) != len(slots)
        or not np.isin(slots, np.arange(12)).all()
    ):
        raise ValueError("Distinct original slots and finite15 coordinates required")
    averages, counts = [], []
    for role in range(3):
        z = x[slots // 4 == role]
        counts.append(len(z))
        averages.append(np.concatenate(([len(z)], z.sum(0))) / 4)
    phi = np.einsum("i,j,k->ijk", *averages).ravel()
    return phi, int(np.prod(counts)), counts


def validate_bundle(bundle):
    """Reject outcome fields and arbitrary renamed scoring programs."""
    if set(bundle) != {
        "version",
        "code_hash",
        "targets",
        "dimensions",
        "witnesses",
        "interpretation",
    }:
        raise ValueError("Unexpected cold bundle fields")
    if (
        bundle["version"] != VERSION
        or bundle["code_hash"] != code_hash()
        or tuple(bundle["targets"]) != TARGETS
        or bundle["dimensions"] != 15
        or bundle["interpretation"]
        != "source_signed_tensor_reading;not_probability_or_text_intervention"
    ):
        raise ValueError("Cold reader contract changed")
    w = np.asarray(bundle["witnesses"], float)
    if w.shape != (2, 5, 4096) or not np.isfinite(w).all():
        raise ValueError("Invalid source-only witnesses")
    return w


def cold_record(bundle, record):
    """Score numeric event evidence without labels,author features or model queries."""
    w = validate_bundle(bundle)
    if set(record) != {"frame_key", "view", "events"} or record["view"] not in (0, 1):
        raise ValueError("Unexpected cold record fields")
    ids, slots, coordinates = [], [], []
    for event in record["events"]:
        if set(event) != {"event_id", "slot", "coordinates"}:
            raise ValueError("Unexpected event or target field")
        if type(event["slot"]) is not int or not isinstance(event["event_id"], str):
            raise ValueError("Integer slots and string event IDs required")
        ids.append(event["event_id"])
        slots.append(event["slot"])
        coordinates.append(event["coordinates"])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate evidence event")
    phi, count, role_counts = subset_feature(
        np.asarray(coordinates, float).reshape(-1, 15), np.asarray(slots, dtype=int)
    )
    mass = np.einsum("vqk,k->vq", w, phi)
    return {
        "frame_key": record["frame_key"],
        "view": record["view"],
        "role_counts": role_counts,
        "triples": count,
        "opportunity_mass": count / 64,
        "retained_readings": mass.tolist(),
        "conditional_readings": (mass * 64 / count).tolist() if count else None,
        "status": "OBSERVED" if count else "NO_TRIPLE_OPPORTUNITY",
    }
