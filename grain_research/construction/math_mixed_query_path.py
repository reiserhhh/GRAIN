#!/usr/bin/env python3
"""HM35 addressed mixed derivatives and fixed three-node path quadrature."""

from __future__ import annotations

import itertools

import numpy as np

NODES = (0.0, 0.5, 1.0)


def design(rows, queries):
    """Select the predeclared pairs/wordings, retaining all eight H/B/Q fibers."""
    result = []
    for pair, role in ((0, "source"), (3, "application")):
        for query in queries:
            for wording in (0, 2):
                selected = {
                    (
                        r["donor_side"],
                        r["receiver_side"],
                        r["early_query_side"],
                        r["query_side"],
                    ): r
                    for r in rows
                    if (r["pair"], r["wording"], r["query"]) == (pair, wording, query)
                }
                if len(selected) != 16:
                    raise ValueError("complete sixteen-corner group required")
                for h, b, q in itertools.product(range(2), repeat=3):
                    left, right = selected[h, b, 0, q], selected[h, b, 1, q]
                    if (
                        left["role"] != role
                        or right["role"] != role
                        or left["input_ids"] != right["input_ids"]
                        or left["early_query_token_ids"]
                        != right["early_query_token_ids"]
                    ):
                        raise ValueError(
                            "fixed receiver, roles and ordered query tokens required"
                        )
                    result.append(
                        {
                            "path": len(result),
                            "pair": pair,
                            "wording": wording,
                            "query": query,
                            "role": role,
                            "H": h,
                            "B": b,
                            "Q": q,
                            "left": left,
                            "right": right,
                        }
                    )
    return result


def interpolate(p0, p1, t):
    """Keep exact float32 endpoints and interpolate only aligned P coordinates."""
    a, b = np.asarray(p0), np.asarray(p1)
    if a.shape != b.shape or a.dtype != np.float32 or b.dtype != np.float32:
        raise ValueError("matching original float32 P states required")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or t not in NODES:
        raise ValueError("finite states and declared node required")
    if t == 0:
        return b.copy()
    if t == 1:
        return a.copy()
    return (b.astype(float) + t * (a.astype(float) - b.astype(float))).astype(
        np.float32
    )


def project(model, delta, token_ids):
    """Use the frozen source projector, never refit with an application state."""
    x = np.asarray(delta, float)
    if list(token_ids) != model["query_token_ids"] or list(x.shape) != model["p_shape"]:
        raise ValueError("source and application P coordinates must match")
    d, inverse = model["D"], model["gram_inverse"]
    supported = (d @ (inverse @ (d.T @ x.ravel()))).reshape(x.shape)
    return supported, x - supported


def mixed_actions(covectors, delta, supported, unsupported):
    """Pair actual second-derivative covectors with three signed directions."""
    v = np.asarray(covectors, float)
    ds = [np.asarray(d, float) for d in (delta, supported, unsupported)]
    if v.shape[1:] != ds[0].shape or any(d.shape != ds[0].shape for d in ds):
        raise ValueError("aligned P covectors and displacement required")
    if not all(np.isfinite(x).all() for x in [v, *ds]):
        raise ValueError("finite mixed derivatives required")
    if not np.allclose(ds[0], ds[1] + ds[2], atol=1e-12, rtol=1e-12):
        raise ValueError("support decomposition must retain full displacement")
    return np.array([[np.sum(c * d) for d in ds] for c in v])


def quadrature(values):
    """Return both fixed half-integrals; never choose a rule after scoring."""
    v = np.asarray(values, float)
    if v.shape[0] != 3 or not np.isfinite(v).all():
        raise ValueError("three finite ordered node values required")
    return {
        "trapezoid": (v[0] + 2 * v[1] + v[2]) / 8,
        "simpson": (v[0] + 4 * v[1] + v[2]) / 12,
    }


def new_status():
    """Initialize separate attempted/completed counts, not artifact-only counts."""
    return {
        "status": "PREPARED",
        "model_seconds": 0.0,
        "in_flight": None,
        **{
            f"{kind}_{phase}": 0
            for kind in ("forward", "first", "second")
            for phase in ("attempted", "completed")
        },
        "committed_nodes": 0,
        "new_fits": 0,
    }


def reserve(status, kind, maximum):
    """Reserve an actual call before executing; uncertain calls cannot replay."""
    s = dict(status)
    if kind not in ("forward", "first", "second"):
        raise ValueError("unknown derivative stage")
    if s[f"{kind}_attempted"] >= maximum:
        raise RuntimeError("fixed call budget exhausted")
    s[f"{kind}_attempted"] += 1
    return s
