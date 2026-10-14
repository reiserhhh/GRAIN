#!/usr/bin/env python3
"""Separate SOURCE target assignment from retained profile association."""

import numpy as np
from scipy.special import logsumexp

from grain_research.construction import math_diagnostic_action_bridge as D
from grain_research.construction import math_relational_diagnostic_update as R


def validate(kernels, codes):
    """Require addressed positive paired kernels and every observed profile."""
    k, g = np.asarray(kernels, float), np.asarray(codes)
    if (
        g.ndim != 1
        or not np.issubdtype(g.dtype, np.integer)
        or not np.isin(g, np.arange(32)).all()
        or np.bincount(g, minlength=32).min() == 0
        or k.ndim != 5
        or k.shape[1:4] != (2, 2, 2)
        or k.shape[-1] != len(g)
        or not np.isfinite(k).all()
        or np.any(k <= 0)
    ):
        raise ValueError("Positive SAME-event kernels/all32 SOURCE profiles required")
    return k, g


def densities(kernels, codes):
    """Compile the full common catalogue; no arbitrary source-bank choice."""
    k, g = validate(kernels, codes)
    out = np.empty((len(k), 2, 3, 2, 32))
    for code in range(32):
        a = k[..., g == code]
        mu = a.mean(-1)
        out[..., code] = np.stack(
            (
                np.stack((mu[:, 0, :, 0], mu[:, 1, :, 1]), 1),
                a.prod(-2).mean(-1),
                mu.prod(-1),
            ),
            2,
        )
    return out


def orbit_density(kernels, codes, target):
    """Exact assignment-orbit mean density, NOT the mean of nonlinear scores."""
    k, g = validate(kernels, codes)
    if type(target) is not int or target not in range(5):
        raise ValueError("One declared target required")
    other = g & ~(1 << target)
    out = np.empty((len(k), 2, 3, 2, 32))
    for c in np.unique(other):
        a = k[..., other == c]
        size, mu = a.shape[-1], a.mean(-1)
        residual = a - mu[..., None]
        cov = residual.prod(-2).sum(-1) / (size - 1)
        single = np.stack((mu[:, 0, :, 0], mu[:, 1, :, 1]), 1)
        joint = a.prod(-2).mean(-1)
        for code in (int(c), int(c | (1 << target))):
            n = int(np.sum(g == code))
            product = mu.prod(-1) + (1 / n - 1 / size) * cov
            out[..., code] = np.stack((single, joint, product), 2)
    if np.any(out <= 0) or not np.isfinite(out).all():
        raise ValueError("Orbit expectation must retain positive observation laws")
    return out


def priors(codes):
    """Keep uniform profiles and source frequencies as distinct fixed references."""
    n = np.bincount(np.asarray(codes, int), minlength=32)
    if n.min() == 0:
        raise ValueError("No missing-profile smoothing")
    return np.stack((np.full(32, 1 / 32), n / n.sum()))


def action(density, reference, target):
    """Finite edit log-odds action, retaining all hypothetical other profiles."""
    d, p = np.asarray(density, float), np.asarray(reference, float)
    if (
        d.shape[-2:] != (2, 32)
        or p.shape != (32,)
        or not np.isfinite(d).all()
        or not np.isfinite(p).all()
        or np.any(d <= 0)
        or np.any(p <= 0)
        or abs(p.sum() - 1) > 1e-12
        or type(target) is not int
        or target not in range(5)
    ):
        raise ValueError("Positive full profile law, reference and target required")
    logs = np.log(d) + np.log(p)
    high = R.BITS[:, target] == 1
    odds = logsumexp(logs[..., high], -1) - logsumexp(logs[..., ~high], -1)
    return odds[..., 0] - odds[..., 1]


def accounts(kernels, codes):
    """Return full/association/departure for every reference/law/target."""
    full = densities(kernels, codes)
    pp = priors(codes)
    out = np.empty((2, len(kernels), 2, 3, 3, 5))
    for q in range(5):
        baseline = orbit_density(kernels, codes, q)
        for p in range(2):
            a, b = action(full, pp[p], q), action(baseline, pp[p], q)
            out[p, ..., q] = np.stack((a, b, a - b), -1)
    return out


def rank(observed, references):
    """Doubled finite rank tails; no assumption of a zero or symmetric null."""
    x = np.asarray(references, float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Fixed finite reference values required")
    upper = (1 + np.sum(x >= observed)) / (len(x) + 1)
    lower = (1 + np.sum(x <= observed)) / (len(x) + 1)
    return {
        "upper_rank": float(upper),
        "lower_rank": float(lower),
        "two_sided_rank": float(min(1, 2 * min(upper, lower))),
        "reference_mean": float(x.mean()),
        "reference_median": float(np.median(x)),
    }


def cold(bank, record):
    """Score addressed vector edits with no application labels/model responses."""
    from grain_research.construction import math_source_catalogue_response as C

    points, _ = D.edit_points(record, bank["source_means"].shape[-1])
    # Both edited-view slots are needed by the common compiler; the caller's
    # one actual edit is copied, not treated as two independent observations.
    view = record["edit"]["view"]
    x = np.stack((points, points), 0)[None]
    k = C.kernels(x, bank["source_means"], bank["variance"])
    a = accounts(k, bank["codes"])[:, 0, view]
    return {
        "frame_key": record["frame_key"],
        "view": view,
        "actions": a.tolist(),
        "order": "reference,law,part,target",
        "interpretation": "source_assignment_diagnostic_log_odds_not_human_traits",
    }
