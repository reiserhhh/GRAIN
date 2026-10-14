#!/usr/bin/env python3
"""Whole SOURCE-target actions and valuation-class contacts on addressed evidence."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from . import math_event_anchor_field as A
from . import math_ordered_valuation as O
from . import math_source_target_law as S


def event_weights(bank, points):
    """Reuse the exact HM267 author law; no labels enter kernel construction."""
    z, _, _, _ = A.validate(
        bank["source_comments"], bank["codes"], bank["anchors"], bank["variance"]
    )
    x = np.asarray(points, float)
    if x.ndim != 2 or x.shape[1] != z.shape[2] or not np.isfinite(x).all():
        raise ValueError(
            "Finite vectors in the frozen observation coordinates required"
        )
    if z.shape[1] % 2:
        raise ValueError("Exactly two equal source windows required")
    projection = A.projection(bank["directions"])
    result = {}
    for space in A.SPACES:
        obs, source = (x @ projection, z @ projection) if space == "GRAIN15" else (x, z)
        for family in A.FAMILIES:
            prototypes = (
                source.reshape(len(source), 2, source.shape[1] // 2, -1).mean(2)
                if family == "window_mean"
                else source
            )
            ell = A.author_log_law(obs, prototypes, bank["variance"])
            result[f"{space}/{family}"] = np.exp(
                ell - logsumexp(ell, -1, keepdims=True)
            )
    return result


def cut_contract(codes, anchors, cuts):
    """Verify an inherited strict cut, preserving ties rather than refitting it."""
    c, a, t = np.asarray(codes), np.asarray(anchors, float), np.asarray(cuts, float)
    if (
        c.ndim != 1
        or c.dtype.kind not in "iu"
        or not np.isin(c, np.arange(32)).all()
        or a.shape != (len(c), 5)
        or t.shape != (5,)
        or not np.isfinite(a).all()
        or not np.isfinite(t).all()
    ):
        raise ValueError("Frozen source codes/anchors/five cuts required")
    bits = (c[:, None] >> np.arange(5)) & 1
    mismatch = (bits != (a > t)).sum(0)
    if mismatch.any():
        raise ValueError("Wrong source cut contract; no automatic recutting")
    return {
        "cuts": t.tolist(),
        "mismatches": mismatch.tolist(),
        "ties": (a == t).sum(0).tolist(),
    }


def atom_laws(weights, anchors, target):
    """Push complete source laws onto all distinct ordered values, merging ties."""
    w, a = S.validate_laws(weights, anchors)
    if target not in range(5):
        raise ValueError("One of the five existing source targets required")
    support, inverse = np.unique(a[:, target], return_inverse=True)
    if len(support) < 2:
        raise ValueError("Constant source target has no ordered comparison")
    indicator = (inverse[:, None] == np.arange(len(support))).astype(float)
    return support, w @ indicator


def ordered_account(support, measure):
    """Retain the exact whole action and both areas; never optimize a scalar."""
    s, m = np.asarray(support, float), O.validate_measure(measure)
    if (
        s.ndim != 1
        or len(s) != m.shape[-1]
        or not np.isfinite(s).all()
        or np.any(np.diff(s) <= 0)
    ):
        raise ValueError("Strictly ordered finite support matched to zero-mass action")
    tail = O.tail_actions(m)
    gaps = np.diff(s)
    return {
        "tail": tail,
        "mean_change": m @ s,
        "integral": tail @ gaps,
        "positive_area": np.maximum(tail, 0) @ gaps,
        "negative_area": np.maximum(-tail, 0) @ gaps,
        "certificate": O.certificate(m) if m.ndim == 1 else None,
    }


def behavior_contact(weights, behavior):
    """Compile a signed contact and genuine behavior-oriented comparison laws."""
    w, b = np.asarray(weights, float), np.asarray(behavior, float)
    if (
        w.ndim != 3
        or w.shape[1] != 2
        or not len(w)
        or b.shape != (len(w),)
        or not np.isfinite(w).all()
        or not np.isfinite(b).all()
        or np.any(w < 0)
        or np.max(abs(w.sum(-1) - 1)) > 1e-12
    ):
        raise ValueError(
            "Paired normalized source laws and SAME behavior observations required"
        )
    measure = np.mean((w[:, 0] - w[:, 1]) * b[:, None], 0)
    scale = float(np.mean(abs(b)))
    if scale == 0:
        return {
            "measure": measure,
            "scale": scale,
            "status": "BEHAVIOR_ZERO_UNOBSERVED",
            "plus": None,
            "minus": None,
        }
    orientation = b < 0
    plus = np.sum(
        abs(b[:, None]) * w[np.arange(len(w)), orientation.astype(int)], 0
    ) / np.sum(abs(b))
    minus = np.sum(
        abs(b[:, None]) * w[np.arange(len(w)), (~orientation).astype(int)], 0
    ) / np.sum(abs(b))
    return {
        "measure": measure,
        "scale": scale,
        "status": "OBSERVED",
        "plus": plus,
        "minus": minus,
    }


def cold_record(bank, record):
    """Read a numeric edit's whole target action with no labels or model answers."""
    if (
        set(record) != {"frame_key", "before", "after"}
        or not isinstance(record["frame_key"], str)
        or not record["frame_key"]
    ):
        raise ValueError(
            "Only frame identity and two addressed vector observations allowed"
        )
    points, ids = [], []
    for name in ("after", "before"):
        item = record[name]
        if (
            set(item) != {"event_id", "vector"}
            or not isinstance(item["event_id"], str)
            or not item["event_id"]
        ):
            raise ValueError("Only opaque event identity and numeric vector allowed")
        points.append(item["vector"])
        ids.append(item["event_id"])
    if ids[0] == ids[1]:
        raise ValueError("Repeated event is not new evidence")
    result = {}
    for key, weights in event_weights(bank, points).items():
        targets = {}
        for q, target in enumerate(A.TARGETS):
            support, mass = atom_laws(weights, bank["anchors"], q)
            measure = mass[0] - mass[1]
            account = ordered_account(support, measure)
            targets[target] = {
                "support": support.tolist(),
                "after_mass": mass[0].tolist(),
                "before_mass": mass[1].tolist(),
                "signed_mass": measure.tolist(),
                "tail_after_each_support_value": account["tail"].tolist(),
                "mean_change": float(account["mean_change"]),
                "positive_area": float(account["positive_area"]),
                "negative_area": float(account["negative_area"]),
                "all_increasing_valuations": account["certificate"],
            }
        result[key] = targets
    return {
        "frame_key": record["frame_key"],
        "before_event": ids[1],
        "after_event": ids[0],
        "readings": result,
        "meaning": "SOURCE-supported target distribution action, not a human-trait change",
    }
