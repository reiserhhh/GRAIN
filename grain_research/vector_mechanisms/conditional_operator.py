"""Source-fitted finite evidence-action operators, without model dependencies."""

from __future__ import annotations

import numpy as np


def split_name(family, situation):
    """Separate entire task/wording groups, never individual paired texts."""
    known_f = family in {"named", "described"}
    known_c = situation in {"diary", "opinion", "advice"}
    if known_f and known_c:
        return "source"
    if known_f:
        return "new_task"
    return "new_wording" if known_c else "new_both"


def make_pairs(rows):
    """Require unique high/low records for each axis/family/task/seed."""
    groups = {}
    for r in rows:
        if r["axis"] == "none":
            continue
        key = (r["axis"], r["family"], r["situation"], r["seed"])
        g = groups.setdefault(key, {})
        if r["pole"] in g:
            raise ValueError(f"Duplicate pole: {key}")
        g[r["pole"]] = r
    out = []
    for (axis, family, situation, seed), g in sorted(groups.items()):
        if set(g) != {-1, 1}:
            raise ValueError("Incomplete pair")
        out.append(
            {
                "pair": f"{axis}|{family}|{situation}|s{seed}",
                "axis": axis,
                "family": family,
                "situation": situation,
                "seed": seed,
                "split": split_name(family, situation),
                "high": g[1],
                "low": g[-1],
            }
        )
    return out


def fit_action(x, y, rank=None, alpha=1.0):
    """Fit zero-origin Ridge on supplied source pairs only; optionally truncate."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("Expected aligned matrices")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or alpha <= 0:
        raise ValueError("Invalid finite fit data or penalty")
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = x / scale
    w = np.linalg.solve(z.T @ z + alpha * np.eye(x.shape[1]), z.T @ y)
    if rank is not None:
        u, s, vt = np.linalg.svd(w, full_matrices=False)
        w = (u[:, :rank] * s[:rank]) @ vt[:rank]
    return {"scale": scale, "weight": w}


def predict_action(model, x):
    """Compute an action from text-coordinate changes without a neural call."""
    return np.asarray(x, float) / model["scale"] @ model["weight"]


def metrics(predicted, target):
    """Report finite contrast error and direction; undefined correlation is null."""
    p, y = np.asarray(predicted, float), np.asarray(target, float)
    corr = None
    if len(y) > 2 and np.std(p) > 1e-12 and np.std(y) > 1e-12:
        corr = float(np.corrcoef(p, y)[0, 1])
    return {
        "n": len(y),
        "r": corr,
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "mean_prediction": float(np.mean(p)),
        "mean_target": float(np.mean(y)),
        "direction": float(np.mean(np.sign(p) == np.sign(y))),
    }
