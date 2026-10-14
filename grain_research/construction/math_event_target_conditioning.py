#!/usr/bin/env python3
"""Source-standardized same-event target changes, retaining co-target routing."""

from __future__ import annotations

import numpy as np

from . import math_event_anchor_field as A
from . import math_profile_reader as P
from . import math_target_evidence_routing as T

PARTS = ("marginal", "matched", "reference_shift", "profile_composition")


def reference(codes):
    """Reuse the SOURCE-only common reference, never current-author outcomes."""
    c = np.asarray(codes)
    if (
        c.ndim != 1
        or c.dtype.kind not in "iu"
        or not np.isin(c, np.arange(32)).all()
        or np.bincount(c, minlength=32).min() == 0
    ):
        raise ValueError("Every SOURCE joint profile must be observed")
    bits = (c[:, None] >> np.arange(5)) & 1
    compiled = P.construct(np.zeros((len(c), 1)), bits)
    return {
        "omega": compiled["omega"],
        "counts": compiled["counts"],
        "profile_counts": np.bincount(c, minlength=32),
    }


def decompose(profile, omega):
    """Observe same-event changes within fixed profiles and their rerouting."""
    p, o = np.asarray(profile, float), np.asarray(omega, float)
    if (
        p.ndim != 3
        or p.shape[1:] != (2, 32)
        or not len(p)
        or not np.isfinite(p).all()
        or np.any(p <= 0)
        or np.max(abs(p.sum(-1) - 1)) > 1e-12
        or o.shape != (5, 16)
        or not np.isfinite(o).all()
        or np.any(o <= 0)
        or np.max(abs(o.sum(-1) - 1)) > 1e-12
    ):
        raise ValueError(
            "Two positive normalized profile laws/common reference required"
        )
    lp = np.log(p)
    old = T.analyze(lp, lp[:, 0] - lp[:, 1])
    v, r = old["profile_weights"], old["conditional_high"]
    dr = r[:, 0] - r[:, 1]
    mean_v, mean_r = v.mean(1), r.mean(1)
    matched = np.sum(o * dr, -1)
    shifted = np.sum((mean_v - o) * dr, -1)
    composition_cells = (v[:, 0] - v[:, 1]) * mean_r
    composition = composition_cells.sum(-1)
    total = old["corners"][..., 3] - old["corners"][..., 0]
    return {
        "values": np.stack((total, matched, shifted, composition), -1),
        "conditional_delta": dr,
        "matched_cells": o * dr,
        "composition_cells": composition_cells,
        "profile_weights": v,
        "conditional_high": r,
        "one_sided_accounts": old["probability_accounts"],
    }


def cold_record(bank, record):
    """Return all conditional observations for one explicitly addressed vector edit."""
    if (
        set(record) != {"frame_key", "before", "after"}
        or not isinstance(record["frame_key"], str)
        or not record["frame_key"]
    ):
        raise ValueError("Only frame identity and two addressed observations allowed")
    points, ids = [], []
    for key in ("after", "before"):
        event = record[key]
        if (
            set(event) != {"event_id", "vector"}
            or not isinstance(event["event_id"], str)
            or not event["event_id"]
        ):
            raise ValueError("Each observation needs an opaque ID and numeric vector")
        ids.append(event["event_id"])
        points.append(event["vector"])
    if ids[0] == ids[1]:
        raise ValueError("Repeated event evidence")
    omega = reference(bank["codes"])["omega"]
    fields = {}
    for name, readings in A.score_events(bank, np.asarray(points, float)).items():
        d = decompose(readings["profile_mass"][None], omega)
        fields[name] = {k: value[0].tolist() for k, value in d.items()}
    return {
        "frame_key": record["frame_key"],
        "targets": list(A.TARGETS),
        "parts": list(PARTS),
        "before_event": ids[1],
        "after_event": ids[0],
        "readings": fields,
        "meaning": "SOURCE conditional target observation, not causal trait isolation",
    }
