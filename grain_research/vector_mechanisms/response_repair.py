#!/usr/bin/env python3
"""Exact state/tangent crossing and signed error-reduction accounting."""

from __future__ import annotations

import numpy as np

CELLS = ("VV", "VJ", "JV", "JJ")


def allocate(cells, actual):
    """Allocate a fixed 2x2 response change; retain the interaction separately."""
    f, y = np.asarray(cells, float), np.asarray(actual, float)
    if f.shape[:2] != (2, 2) or f.shape[2:] != y.shape:
        raise ValueError(
            "Two state choices x two tangent choices and aligned targets required"
        )
    if not np.isfinite(f).all() or not np.isfinite(y).all():
        raise ValueError("Finite cells and targets required")
    vv, vj, jv, jj = f[0, 0], f[0, 1], f[1, 0], f[1, 1]
    s, t = ((jv - vv) + (jj - vj)) / 2, ((vj - vv) + (jj - jv)) / 2
    change, error = jj - vv, vv - y
    return {
        "actual": y,
        "value": vv,
        "joint": jj,
        "state": s,
        "tangent": t,
        "interaction": jj - jv - vj + vv,
        "change": change,
        "base_squared_error": error**2,
        "joint_squared_error": (jj - y) ** 2,
        "state_linear_error": 2 * error * s,
        "tangent_linear_error": 2 * error * t,
        "state_squared": s**2,
        "tangent_squared": t**2,
        "cross_error": 2 * s * t,
        "error_change": (jj - y) ** 2 - error**2,
    }


def summarize(columns):
    """Describe response geometry and error gain, never treating directions as people."""
    a = {k: np.asarray(v, float) for k, v in columns.items()}
    if any(v.ndim != 1 or not np.isfinite(v).all() for v in a.values()):
        raise ValueError("Finite one-dimensional response columns required")
    s, t, d = a["state"], a["tangent"], a["change"]
    mse = float(np.mean(a["base_squared_error"]))
    energy = float(np.mean(d * d))
    means = {k: float(np.mean(v)) for k, v in a.items()}
    state_cost = (
        means["state_linear_error"] + means["state_squared"] + means["cross_error"] / 2
    )
    tangent_cost = (
        means["tangent_linear_error"]
        + means["tangent_squared"]
        + means["cross_error"] / 2
    )
    total = state_cost + tangent_cost
    return means | {
        "actions": len(d),
        "state_rms": float(np.sqrt(np.mean(s * s))),
        "tangent_rms": float(np.sqrt(np.mean(t * t))),
        "change_rms": float(np.sqrt(energy)),
        "interaction_rms": float(np.sqrt(np.mean(a["interaction"] ** 2))),
        "state_tangent_covariance": float(np.mean((s - s.mean()) * (t - t.mean()))),
        "state_tangent_opposite_signs": int(np.sum(s * t < 0)),
        "state_projection_on_change": float(np.mean(s * d) / energy)
        if energy
        else float("nan"),
        "tangent_projection_on_change": float(np.mean(t * d) / energy)
        if energy
        else float("nan"),
        "state_attributed_error_gain": -state_cost / mse if mse else float("nan"),
        "tangent_attributed_error_gain": -tangent_cost / mse if mse else float("nan"),
        "total_relative_error_gain": -total / mse if mse else float("nan"),
        "allocation_max_error": float(np.max(abs(s + t - d))),
        "mse_accounting_error": abs(total - means["error_change"]),
    }
