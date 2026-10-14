#!/usr/bin/env python3
"""Marginal-matched source linkage on actual joined evidence, without fitting."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from grain_research.construction import math_event_anchor_field as A
from grain_research.construction import math_relational_diagnostic_update as D

VIEWS = ("pooled_comments", "registered_windows")
COMPONENTS = ("linked", "profile_product", "linkage_correction")


def profile_laws(event_logs, codes):
    """Keep singleton marginals, empirical profile prior and two coupling laws."""
    ell, g = np.asarray(event_logs, float), np.asarray(codes)
    if (
        ell.ndim != 3
        or ell.shape[1] != 2
        or g.shape != (ell.shape[2],)
        or g.dtype.kind not in "iu"
        or not np.isin(g, np.arange(32)).all()
        or not np.isfinite(ell).all()
    ):
        raise ValueError("Finite paired event logs and aligned full profiles required")
    counts = np.bincount(g, minlength=32)
    if counts.min() == 0:
        raise ValueError("Every declared profile must have source support")
    joint, product = np.empty((len(ell), 32)), np.empty((len(ell), 32))
    for code in range(32):
        a, b = ell[:, :, g == code].transpose(1, 0, 2)
        n = np.log(counts[code])
        joint[:, code] = logsumexp(a + b, -1) - n
        product[:, code] = logsumexp(a, -1) + logsumexp(b, -1) - 2 * n
    prior = counts / counts.sum()
    result = D.update(product + np.log(prior), joint - product)
    result["profile_prior"] = prior
    return result


def pair_readings(bundle, points, space, view):
    """Read ordered pairs through a frozen comment law, not a fitted relation."""
    x = np.asarray(points, float)
    source, codes, _, _ = A.validate(
        bundle["source_comments"],
        bundle["codes"],
        bundle["anchors"],
        bundle["variance"],
    )
    if (
        x.ndim != 3
        or x.shape[1:] != (2, source.shape[-1])
        or not np.isfinite(x).all()
        or space not in A.SPACES
        or view not in VIEWS
        or source.shape[1] % 2
    ):
        raise ValueError(
            "Complete ordered pairs in a declared source contract required"
        )
    if space == "GRAIN15":
        q = A.projection(bundle["directions"])
        x, source = x @ q, source @ q
    banks = [source, source] if view == VIEWS[0] else list(np.split(source, 2, axis=1))
    logs = np.stack(
        [A.author_log_law(x[:, v], banks[v], bundle["variance"]) for v in range(2)],
        1,
    )
    return profile_laws(logs, codes)


def corners(endpoints):
    """Enumerate BOTH native orders and all registered real endpoint joins."""
    x = np.asarray(endpoints, float)
    if x.ndim != 4 or x.shape[1:3] != (2, 2) or not np.isfinite(x).all():
        raise ValueError("Author x source-window x own/foreign x vector required")
    points = np.stack(
        [
            np.stack((x[:, 0, a], x[:, 1, b]), 1)
            for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))
        ],
        1,
    )
    return np.stack((points, points[:, :, ::-1]), 1)


def mixed_actions(targets):
    """Keep normalized linked, profile-only and their signed mixed-action difference."""
    t = np.asarray(targets, float)
    if t.ndim != 5 or t.shape[1:] != (2, 4, 2, 5) or not np.isfinite(t).all():
        raise ValueError("Both orders/four corners/two laws/five targets required")
    mixed = t[:, :, 0] - t[:, :, 1] - t[:, :, 2] + t[:, :, 3]
    return {
        COMPONENTS[0]: mixed[:, :, 0],
        COMPONENTS[1]: mixed[:, :, 1],
        COMPONENTS[2]: mixed[:, :, 0] - mixed[:, :, 1],
    }


def contact(source_action, native_action):
    """Keep every source/native target contact; graph edges are not iid people."""
    s, n = np.asarray(source_action, float), np.asarray(native_action, float)
    if s.ndim != 2 or s.shape[1] != 5 or n.shape != s.shape:
        raise ValueError("Aligned complete five-target actions required")
    if not np.isfinite(s).all() or not np.isfinite(n).all():
        raise ValueError("Missing observations cannot be neutral scores")
    return s.T @ n / len(s)


def cold(bundle, record):
    """Compile a four-endpoint evidence edit without answers, text or target labels."""
    if (
        set(record) != {"frame_key", "endpoints"}
        or not isinstance(record["frame_key"], str)
        or not record["frame_key"]
        or not isinstance(record["endpoints"], list)
        or len(record["endpoints"]) != 2
    ):
        raise ValueError("Frame identity and two own/foreign endpoint pairs required")
    seen, points = set(), []
    for window in record["endpoints"]:
        if not isinstance(window, list) or len(window) != 2:
            raise ValueError("Both real own/foreign endpoints required")
        row = []
        for event in window:
            if set(event) != {"event_id", "vector"}:
                raise ValueError("No application labels or model answers permitted")
            uid = event["event_id"]
            if not isinstance(uid, str) or not uid or uid in seen:
                raise ValueError("Four distinct evidence event identities required")
            seen.add(uid)
            row.append(event["vector"])
        points.append(row)
    c = corners(np.asarray(points, float)[None])
    result = {}
    for space in A.SPACES:
        for view in VIEWS:
            laws = pair_readings(bundle, c.reshape(8, 2, -1), space, view)
            targets = laws["targets"].reshape(1, 2, 4, 2, 5)
            result[f"{space}/{view}"] = {
                "corner_targets": targets[0].tolist(),
                "corner_profiles": laws["profile"].reshape(2, 4, 2, 32).tolist(),
                "mixed_actions": {
                    k: v[0].tolist() for k, v in mixed_actions(targets).items()
                },
            }
    return {
        "frame_key": record["frame_key"],
        "event_ids": sorted(seen),
        "readings": result,
    }
