#!/usr/bin/env python3
"""Exact SOURCE evidence valuation under background and author-mixture changes."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from grain_research.construction import math_relational_diagnostic_update as R


def validate_corners(corners, focal_view):
    """Require a fixed focal edit and fixed companions, not four arbitrary inputs."""
    x = np.asarray(corners, float)
    if (
        type(focal_view) is not int
        or focal_view not in (0, 1)
        or x.ndim != 4
        or x.shape[:3] != (2, 2, 2)
        or x.shape[-1] == 0
        or not np.isfinite(x).all()
    ):
        raise ValueError("Finite own/foreign x selected/control x oriented windows")
    if not np.array_equal(x[0, :, focal_view], x[1, :, focal_view]):
        raise ValueError("Background switch changed the focal evidence")
    if not np.array_equal(x[:, 0, 1 - focal_view], x[:, 1, 1 - focal_view]):
        raise ValueError("Focal edit changed the companion evidence")
    return x


def valuation(log_weights, log_response):
    """Value a profile edit under target-state conditional weights, all five bits."""
    w, r = np.asarray(log_weights, float), np.asarray(log_response, float)
    if w.shape != (32,) or r.shape != (32,):
        raise ValueError("All32 profiles required; no target/profile selection")
    if not np.isfinite(w).all() or not np.isfinite(r).all():
        raise ValueError("Finite log weights and responses required")
    return np.array(
        [
            logsumexp((w + r)[R.BITS[:, q] == 1])
            - logsumexp(w[R.BITS[:, q] == 1])
            - logsumexp((w + r)[R.BITS[:, q] == 0])
            + logsumexp(w[R.BITS[:, q] == 0])
            for q in range(5)
        ]
    )


def split_accounts(crossed):
    """Keep the symmetric composition/response accounts and their signed closure."""
    p = np.asarray(crossed, float)
    if p.shape != (2, 2, 5) or not np.isfinite(p).all():
        raise ValueError("Both background weights x both responses x five targets")
    composition = (p[0, 0] - p[1, 0] + p[0, 1] - p[1, 1]) / 2
    within = (p[0, 0] - p[0, 1] + p[1, 0] - p[1, 1]) / 2
    total = p[0, 0] - p[1, 1]
    return {
        "mixed": total,
        "composition": composition,
        "within_profile": within,
        "closure": total - composition - within,
    }


def read(bank, corners, focal_view):
    """Return exact conditional action accounts; never fit native answers."""
    x = validate_corners(corners, focal_view)
    old = R.read(bank, x.reshape(4, 2, x.shape[-1]))
    lp = old["log_profile"].reshape(2, 2, 2, 32)
    odds = old["log_odds"].reshape(2, 2, 2, 5)
    control, response = lp[:, 1], lp[:, 0] - lp[:, 1]
    cross = np.array(
        [
            [
                [valuation(control[c, law], response[d, law]) for d in (0, 1)]
                for c in (0, 1)
            ]
            for law in (0, 1)
        ]
    )
    split = [split_accounts(p) for p in cross]
    actions = (odds[:, 0] - odds[:, 1]).transpose(1, 0, 2)
    shifted = response[0] - response[1]
    return {
        "log_profile": lp,
        "log_response": response,
        "crossed": cross,
        "actions": actions,
        **{k: np.array([s[k] for s in split]) for k in split[0]},
        "direct_action_error": actions - cross[:, (0, 1), (0, 1)],
        "profile_response_shift": shifted - shifted.mean(-1, keepdims=True),
        "common_response_shift": shifted.mean(-1),
    }


def addressed_corners(record, dimensions):
    """Bind window-mean evidence IDs across corners without accepting truth labels."""
    if set(record) != {"frame_key", "focal_view", "corners"}:
        raise ValueError("Only frame key, focal view and addressed corners accepted")
    if not isinstance(record["frame_key"], str) or not record["frame_key"]:
        raise ValueError("Nonempty frame key required")
    if not isinstance(record["corners"], list) or len(record["corners"]) != 2:
        raise ValueError("Both companions required")
    points = np.empty((2, 2, 2, dimensions))
    identities = []
    for c in (0, 1):
        if not isinstance(record["corners"][c], list) or len(record["corners"][c]) != 2:
            raise ValueError("Both focal edit arms required")
        groups = []
        for a in (0, 1):
            windows = record["corners"][c][a]
            if not isinstance(windows, list) or len(windows) != 2:
                raise ValueError("Two nonempty addressed windows required")
            ids, seen = [], set()
            for v, window in enumerate(windows):
                if set(window) != {"event_ids", "mean"}:
                    raise ValueError("Window means and event IDs only")
                names = window["event_ids"]
                if (
                    not isinstance(names, list)
                    or not names
                    or any(not isinstance(n, str) or not n for n in names)
                    or len(set(names)) != len(names)
                    or seen.intersection(names)
                ):
                    raise ValueError("Missing/repeated evidence is not another window")
                seen.update(names)
                ids.append(tuple(names))
                vec = np.asarray(window["mean"], float)
                if vec.shape != (dimensions,) or not np.isfinite(vec).all():
                    raise ValueError(
                        "Finite registered window-mean coordinates required"
                    )
                points[c, a, v] = vec
            groups.append(ids)
        identities.append(groups)
    v = record["focal_view"]
    validate_corners(points, v)
    if any(identities[0][a][v] != identities[1][a][v] for a in (0, 1)):
        raise ValueError("Different events cannot be equated by identical means")
    if any(identities[c][0][1 - v] != identities[c][1][1 - v] for c in (0, 1)):
        raise ValueError("Companion event identity changed during the focal edit")
    return points


def cold(bank, record):
    """Emit an addressed five-target explanation using only a frozen SOURCE bank."""
    p = addressed_corners(record, bank["prototypes"].shape[-1])
    r = read(bank, p, record["focal_view"])
    return {
        "frame_key": record["frame_key"],
        "targets": list(R.TARGETS),
        "laws": list(R.LAWS),
        **{
            k: r[k].tolist()
            for k in ("actions", "mixed", "composition", "within_profile", "crossed")
        },
        "source_conditional_measurement_not_psychological_truth": True,
    }
