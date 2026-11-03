#!/usr/bin/env python3
"""Source-referenced joint/conditional publication fields with criterion covectors."""

import numpy as np

PARTS = ("joint", "opportunity", "response")
BEHAVIORS = ("next_log_span", "log_cached_gap", "same_exact_venue")


def pair_observations(lengths, times, venues):
    """Observe actual within-window metadata; gaps are not human reply latencies."""
    length, time, venue = map(np.asarray, (lengths, times, venues))
    if (
        length.ndim != 1
        or len(length) < 2
        or time.shape != length.shape
        or venue.shape != length.shape
        or not np.isfinite(length).all()
        or not np.isfinite(time).all()
        or np.any(length < 0)
        or np.any(np.diff(time) <= 0)
    ):
        raise ValueError("Aligned strictly chronological recorded metadata required")
    return np.column_stack(
        [
            np.log1p(length[1:]),
            np.log1p(np.diff(time)),
            venue[:-1] == venue[1:],
        ]
    )


def validate_arrays(states, outputs):
    """Preserve complete author/view/event axes and finite three-channel outcomes."""
    a, b = np.asarray(states), np.asarray(outputs, float)
    if (
        a.ndim != 3
        or not len(a)
        or not a.shape[-1]
        or a.dtype.kind not in "iu"
        or np.any((a < 0) | (a >= 8))
        or b.shape != (*a.shape, 3)
        or not np.isfinite(b).all()
    ):
        raise ValueError(
            "Aligned author/view/pair fields in the frozen alphabet required"
        )
    return a, b


def reference(states, outputs):
    """Compile equal-author/view finite conditional means; no application fits."""
    a, b = validate_arrays(states, outputs)
    mu, sigma = b.mean((0, 1, 2)), b.std((0, 1, 2))
    if np.any(sigma <= 0):
        raise ValueError("Unobserved output variance cannot be smoothed into a score")
    z = (b - mu) / sigma
    r = np.bincount(a.ravel(), minlength=8) / a.size
    if np.any(r <= 0):
        raise ValueError("Unobserved SOURCE address cannot be imputed as neutral")
    total = np.zeros((8, 3))
    np.add.at(total, a.ravel(), z.reshape(-1, 3))
    psi = total / (a.size * r[:, None])
    return {"mean": mu, "scale": sigma, "address_mass": r, "conditional_mean": psi}


def fields(frame, states, outputs):
    """Return all joint and support-weighted deviations,not fabricated conditional means."""
    a, b = validate_arrays(states, outputs)
    mu, sigma = frame["mean"], frame["scale"]
    r, psi = frame["address_mass"], frame["conditional_mean"]
    if r.shape != (8,) or np.any(r <= 0) or psi.shape != (8, 3):
        raise ValueError("Complete immutable SOURCE observation frame required")
    n, views, pairs = a.shape
    counts, joint = np.zeros((n, views, 8), int), np.zeros((n, views, 8, 3))
    z = (b - mu) / sigma
    for u in range(n):
        for w in range(views):
            np.add.at(counts[u, w], a[u, w], 1)
            np.add.at(joint[u, w], a[u, w], z[u, w])
    probability, joint = counts / pairs, joint / pairs
    opportunity = probability[..., None] * psi
    residual = joint - opportunity
    center, units = r[:, None] * psi, np.sqrt(r)[:, None]
    values = {
        "joint": ((joint - center) / units).reshape(n, views, 24),
        "opportunity": ((opportunity - center) / units).reshape(n, views, 24),
        "response": (residual / units).reshape(n, views, 24),
        "support": counts > 0,
        "address_probability": probability,
    }
    return values


def criterion_residuals(labels):
    """Keep all co-target profiles; one-sided groups yield zero observed variation."""
    y = np.asarray(labels)
    if y.ndim != 2 or y.shape[1] != 5 or not np.isin(y, (0, 1)).all():
        raise ValueError("Five complete binary SOURCE criteria required")
    e = np.zeros_like(y, float)
    for q in range(5):
        code = np.delete(y, q, axis=1).astype(int) @ (1 << np.arange(4))
        for g in np.unique(code):
            ix = code == g
            e[ix, q] = y[ix, q] - y[ix, q].mean()
    return e


def compile_rule(states, outputs, assignments):
    """Orient one measuring covector per criterion using SOURCE early joint fields."""
    frame = reference(states, outputs)
    x = fields(frame, states, outputs)["joint"][:, 0]
    ys = np.asarray(assignments)
    if ys.shape != (2, len(x), 5):
        raise ValueError("Both old source assignments required")
    raw = np.array([criterion_residuals(y).T @ x / len(x) for y in ys])
    norms = np.linalg.norm(raw, axis=-1)
    if np.any(norms <= 0):
        raise ValueError(
            "No SOURCE orientation;do not create a neutral personality ruler"
        )
    return frame | {"covectors": raw / norms[..., None], "source_contact_norm": norms}


def score(rule, measured):
    """Use the SAME immutable source covectors for all retained components."""
    return {
        p: np.einsum("uwf,aqf->uwaq", measured[p], rule["covectors"]) for p in PARTS
    }


def correlation(x, y):
    """Return Pearson r or explicit undefined,never a fabricated zero."""
    a, b = np.asarray(x, float), np.asarray(y, float)
    if (
        a.shape != b.shape
        or a.ndim != 1
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Finite aligned author fields required")
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return None if denominator == 0 else float(a @ b / denominator)


def cold(rule, record):
    """Score only an identified observed window,without criterion labels/future windows."""
    if set(record) != {"author_key", "pairs"} or not record["pairs"]:
        raise ValueError(
            "Only opaque author key and recorded publication pairs accepted"
        )
    cells, outputs, previous, seen = [], [], None, set()
    keys = {
        "origin_id",
        "destination_id",
        "origin_time",
        "destination_time",
        "origin_cell",
        "next_span",
        "same_venue",
    }
    for pair in record["pairs"]:
        if set(pair) != keys:
            raise ValueError(
                "Pair fields cannot contain labels or unqualified outcomes"
            )
        origin, destination = pair["origin_id"], pair["destination_id"]
        if (
            origin == destination
            or (origin, destination) in seen
            or pair["destination_time"] <= pair["origin_time"]
        ):
            raise ValueError("Duplicate/nonchronological event pair")
        if previous is not None and (origin, pair["origin_time"]) != previous:
            raise ValueError("Pairs must form one qualified cached-event window")
        if (
            type(pair["origin_cell"]) is not int
            or not 0 <= pair["origin_cell"] < 8
            or pair["next_span"] < 0
            or type(pair["same_venue"]) is not bool
        ):
            raise ValueError("Invalid observed-address/production units")
        seen.add((origin, destination))
        previous = destination, pair["destination_time"]
        cells.append(pair["origin_cell"])
        outputs.append(
            [
                np.log1p(pair["next_span"]),
                np.log1p(pair["destination_time"] - pair["origin_time"]),
                pair["same_venue"],
            ]
        )
    measured = fields(rule, np.array(cells)[None, None], np.array(outputs)[None, None])
    return {"fields": measured, "scores": score(rule, measured)}
