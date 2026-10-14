#!/usr/bin/env python3
"""Criterion-qualified source functions at the same paired neural observation."""

from __future__ import annotations

import numpy as np

from grain_research.construction import math_author_pair_criterion as A
from grain_research.construction import math_criterion_components as C
from grain_research.construction import math_same_regime_reader as O
from grain_research.construction import math_source_event_coupling as P


def pair_components(weights, labels, means):
    """Contract label-blind F-J weights with all three retained criterion parts."""
    w, y, m = (np.asarray(a, float) for a in (weights, labels, means))
    if (
        w.ndim != 3
        or w.shape[1:] != (2, len(y))
        or y.shape != m.shape
        or y.shape[1:] != (5,)
        or not all(np.isfinite(a).all() for a in (w, y, m))
        or np.max(abs(w.sum(-1))) > 1e-12
    ):
        raise ValueError("Aligned two-order signed weights and five criteria required")
    return np.stack([w @ a for a in (y, m, y - m)], axis=2)


def contacts(values, reader):
    """Retain all source/readout crossings and joint/order/singleton accounts."""
    x = np.asarray(values, float)
    if x.ndim != 4 or x.shape[1:] != (2, 3, 5):
        raise ValueError("Pairs/orders/three components/five criteria required")
    return np.stack(
        [np.stack(list(O.contacts(x[:, :, c], reader).values())) for c in range(3)]
    )


def observer_pullback(weights, reader):
    """Return SOURCE coefficients of the fixed joint-reading contact functional."""
    w, r = np.asarray(weights, float), np.asarray(reader, float)
    if w.ndim != 3 or w.shape[1] != 2 or len(w) != len(r) or len(w) == 0:
        raise ValueError("Nonempty aligned two-order SOURCE weights required")
    return w.mean(1).T @ O.reader_parts(r)["join"] / len(w)


def conditional_reference(alpha, labels, means, groups, assignments):
    """Reassign only SOURCE criteria, preserving each bank/configuration stratum."""
    a, y, m = (np.asarray(x, float) for x in (alpha, labels, means))
    g, ix = np.asarray(groups), np.asarray(assignments)
    if (
        a.ndim != 3
        or a.shape[1:] != y.shape
        or y.shape != m.shape
        or g.shape != y.shape
        or ix.ndim != 3
        or ix.shape[0] != 5
        or ix.shape[2] != len(y)
        or ix.dtype.kind not in "iu"
        or not np.isfinite(a).all()
    ):
        raise ValueError("Aligned fixed observer coefficients and assignments required")
    e = y - m
    observed = np.einsum("jq,sjr->sqr", e, a)
    null = np.empty((ix.shape[1], len(a), 5, 5))
    variance = np.empty_like(observed)
    for q in range(5):
        if not np.array_equal(
            np.sort(ix[q], axis=1), np.tile(np.arange(len(y)), (len(ix[q]), 1))
        ) or not np.array_equal(g[ix[q], q], np.tile(g[:, q], (len(ix[q]), 1))):
            raise ValueError("Target assignment left its SOURCE stratum")
        null[:, :, q] = np.einsum("kj,sjr->ksr", e[ix[q], q], a)
        v = C.orbit_covariance(y[:, q], g[:, q])
        variance[:, q] = np.einsum("sjr,jk,skr->sr", a, v, a)
    nominal = (1 + np.sum(abs(null) >= abs(observed), axis=0)) / (len(null) + 1)
    return observed, null, variance, nominal


def cold_pair(bundle, record, space):
    """Read one qualified two-event vector record without application labels/LLMs."""
    P.cold(bundle, record)
    if space not in ("E5", "GRAIN15"):
        raise ValueError("The frozen E5 or GRAIN15 space is required")
    x = np.array([e["vector"] for e in record["events"]], float)
    z = np.asarray(bundle["prototypes"], float)
    y = P.BITS[bundle["codes"]]
    m = C.criterion_design(y, bundle["folds"])[0]
    if space == "GRAIN15":
        x, z = x @ bundle["projection"], z @ bundle["projection"]
    result = {}
    for bank, mask in (
        ("bank0", bundle["folds"] == 0),
        ("bank1", bundle["folds"] == 1),
        ("pooled", np.ones(len(y), bool)),
    ):
        logs = P.event_logs(np.stack([x, x[::-1]]), z[mask], float(bundle["variance"]))
        w = A.source_author_weights(logs, bundle["relink"][mask])
        result[bank] = pair_components((w[:, 0] - w[:, 1])[None], y[mask], m[mask])[
            0
        ].tolist()
    return {
        "frame_key": record["frame_key"],
        "space": space,
        "channels": C.CHANNELS,
        "source_readings": result,
    }
