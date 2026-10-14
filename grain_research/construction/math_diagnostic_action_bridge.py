#!/usr/bin/env python3
"""Source-defined finite evidence edits, separate from native model answers."""

import numpy as np
from scipy.special import logsumexp

from grain_research.construction import math_relational_diagnostic_update as R

LAWS = ("single_window", "paired_author", "paired_profile_product")


def potentials(bank, points):
    """Return single and paired profile laws using the frozen SOURCE reference."""
    b = R.compile_bank(bank["prototypes"], bank["codes"], bank["variance"])
    z, g, s2 = b["prototypes"], b["codes"], float(b["variance"])
    x = np.asarray(points, float)
    if x.ndim != 3 or x.shape[1:] != z.shape[1:] or not np.isfinite(x).all():
        raise ValueError("Registered paired vector coordinates required")
    k = np.stack(
        [x[:, v] @ z[:, v].T / s2 - (z[:, v] ** 2).sum(1) / (2 * s2) for v in (0, 1)],
        axis=1,
    )
    single = np.empty((len(x), 2, 32))
    joint = np.empty((len(x), 32))
    for code in range(32):
        ix = g == code
        single[:, :, code] = logsumexp(k[:, :, ix], -1) - np.log(ix.sum())
        joint[:, code] = logsumexp(k[:, :, ix].sum(1), -1) - np.log(ix.sum())
    return single, joint, single.sum(1)


def odds(logs):
    """Read all five odds from one complete positive profile law."""
    a = np.asarray(logs, float)
    if a.shape[-1] != 32 or not np.isfinite(a).all():
        raise ValueError("All32 finite profile potentials required")
    return np.stack(
        [
            logsumexp(a[..., R.BITS[:, q] == 1], -1)
            - logsumexp(a[..., R.BITS[:, q] == 0], -1)
            for q in range(5)
        ],
        -1,
    )


def readings(bank, points, view):
    """Keep the single available window and both declared paired protocols."""
    if type(view) is not int or view not in (0, 1):
        raise ValueError("Declared edited view required")
    single, joint, product = potentials(bank, points)
    return np.stack((odds(single[:, view]), odds(joint), odds(product)), 1)


def single_field(bank, points, view):
    """Compute the analytic single-window odds gradient without model gradients."""
    if type(view) is not int or view not in (0, 1):
        raise ValueError("Declared edited view required")
    b = R.compile_bank(bank["prototypes"], bank["codes"], bank["variance"])
    z, g, s2 = b["prototypes"][:, view], b["codes"], float(b["variance"])
    x = np.asarray(points, float)
    if x.ndim != 2 or x.shape[1] != z.shape[1] or not np.isfinite(x).all():
        raise ValueError("Finite single-window coordinates required")
    counts = np.bincount(g, minlength=32)
    logk = x @ z.T / s2 - (z**2).sum(1) / (2 * s2) - np.log(counts[g])
    fields = []
    for q in range(5):
        high = ((g >> q) & 1) == 1
        side_means = []
        for mask in (high, ~high):
            a = logk[:, mask]
            w = np.exp(a - logsumexp(a, -1)[:, None])
            side_means.append(w @ z[mask])
        fields.append((side_means[0] - side_means[1]) / s2)
    return np.stack(fields, 1)


def edit_points(record, dimensions):
    """Bind removals to observed IDs; unrelated IDs and duplicate evidence fail."""
    if set(record) != {"frame_key", "windows", "edit"}:
        raise ValueError("Only addressed evidence and a declared edit permitted")
    base = {k: record[k] for k in ("frame_key", "windows")}
    full = R.cold_point(base, dimensions)
    edit = record["edit"]
    if not isinstance(edit, dict) or set(edit) != {"view", "selected", "control"}:
        raise ValueError("One view and two explicit removal sets required")
    view = edit["view"]
    if type(view) is not int or view not in (0, 1):
        raise ValueError("Unknown edit view")
    removed = [edit[k] for k in ("selected", "control")]
    ids = {e["event_id"] for e in base["windows"][view]}
    if any(
        not isinstance(group, list)
        or len(group) != 4
        or any(not isinstance(i, str) for i in group)
        or len(set(group)) != 4
        or not set(group) < ids
        for group in removed
    ) or set(removed[0]) & set(removed[1]):
        raise ValueError(
            "Disjoint four-event removals inside the actual window required"
        )
    points = []
    for group in removed:
        pair = full.copy()
        pair[view] = np.mean(
            [e["vector"] for e in base["windows"][view] if e["event_id"] not in group],
            axis=0,
        )
        points.append(pair)
    return np.asarray(points), view


def cold(bank, record):
    """Produce frozen five-target score actions without native outputs or labels."""
    points, view = edit_points(record, bank["prototypes"].shape[-1])
    values = readings(bank, points, view)
    dx = points[0, view] - points[1, view]
    midpoint = points[:, view].mean(0, keepdims=True)
    first = single_field(bank, midpoint, view)[0] @ dx
    change = values[0] - values[1]
    return {
        "frame_key": record["frame_key"],
        "view": view,
        "laws": list(LAWS),
        "finite_actions": change.tolist(),
        "single_midpoint_action": first.tolist(),
        "single_finite_remainder": (change[0] - first).tolist(),
        "paired_dependence_action": (change[1] - change[2]).tolist(),
        "interpretation": "frozen_source_channel_log_odds_actions_not_native_answers_or_human_probabilities",
    }


def literal(bank, points, view):
    """Independently evaluate explicit distance kernels and profile products."""
    z, g, s2 = bank["prototypes"], bank["codes"], float(bank["variance"])
    rows = []
    for pair in points:
        kernels = np.array(
            [
                np.exp(-np.sum((pair[v] - z[:, v]) ** 2, axis=1) / (2 * s2))
                for v in (0, 1)
            ]
        )
        laws = np.empty((3, 32))
        for code in range(32):
            a, b = kernels[:, g == code]
            laws[0, code] = (a if view == 0 else b).mean()
            laws[1, code] = sum(
                float(aa * bb) for aa, bb in zip(a, b, strict=True)
            ) / len(a)
            laws[2, code] = (sum(float(aa * bb) for aa in a for bb in b)) / len(a) ** 2
        if np.any(laws <= 0):
            raise ValueError("Literal audit outside finite exponential range")
        rows.append(
            [
                [
                    np.log(law[R.BITS[:, q] == 1].sum() / law[R.BITS[:, q] == 0].sum())
                    for q in range(5)
                ]
                for law in laws
            ]
        )
    return np.array(rows)
