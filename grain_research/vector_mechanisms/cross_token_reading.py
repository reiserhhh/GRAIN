#!/usr/bin/env python3
"""Finite cross-region responses at a fixed hidden-state boundary."""

from __future__ import annotations

import numpy as np


def intervention(state, positions, directions):
    """Change only the first BODY half along the frozen equal-factor direction."""
    x, pos, ds = np.asarray(state), np.asarray(positions), np.asarray(directions, float)
    if (
        x.ndim != 2
        or x.dtype.kind != "f"
        or pos.ndim != 1
        or pos.dtype.kind not in "iu"
        or len(pos) < 2
        or np.any(np.diff(pos) <= 0)
        or pos[0] < 0
        or pos[-1] >= len(x) - 1
        or ds.shape != (4, len(pos), x.shape[1])
        or not np.isfinite(x).all()
        or not np.isfinite(ds).all()
    ):
        raise ValueError(
            "Finite state, ordered nonfinal BODY and four fixed directions required"
        )
    cut = len(pos) // 2
    norms = np.linalg.norm(ds[:, :cut], axis=(1, 2))
    if np.any(norms <= 1e-12):
        raise ValueError("Unsupported: a restricted factor direction vanishes")
    unit = (ds[:, :cut] / norms[:, None, None]).mean(0)
    norm = float(np.linalg.norm(unit))
    if norm <= 1e-12:
        raise ValueError("Unsupported: equal-factor direction cancels")
    unit /= norm
    epsilon = 0.01 * float(np.linalg.norm(x[pos[:cut]].astype(float)))
    if epsilon <= 1e-12:
        raise ValueError("Unsupported: zero front-region state")
    keep = np.ones(len(x), dtype=bool)
    keep[pos[:cut]] = False
    plus, minus = x.copy(), x.copy()
    for changed, sign in ((plus, 1), (minus, -1)):
        changed[pos[:cut]] = x[pos[:cut]].astype(float) + sign * epsilon * unit
        if changed[keep].tobytes() != x[keep].tobytes():
            raise ValueError("Untouched rows changed")
        if not np.isfinite(changed).all():
            raise ValueError("Nonfinite changed state")
    return (
        plus,
        minus,
        {
            "cut": np.array(cut),
            "epsilon": np.array(epsilon),
            "unit_direction": unit,
            "factor_norms": norms,
            "unchanged_mask": keep,
            "positions": pos.copy(),
            "realized_plus_norm": np.array(np.linalg.norm((plus - x).astype(float))),
            "realized_minus_norm": np.array(np.linalg.norm((minus - x).astype(float))),
            "midpoint_error": np.array(
                np.linalg.norm((plus.astype(float) + minus.astype(float)) / 2 - x)
            ),
        },
    )


def region_account(original, plus, minus, directions, cut):
    """Contract central finite responses with original directions; no token-as-subject tests."""
    g, gp, gm, ds = (np.asarray(v, float) for v in (original, plus, minus, directions))
    if (
        g.ndim != 2
        or gp.shape != g.shape
        or gm.shape != g.shape
        or ds.shape != (4, *g.shape)
        or not 0 < cut < len(g)
        or any(not np.isfinite(v).all() for v in (g, gp, gm, ds))
    ):
        raise ValueError(
            "Aligned fields/directions and nonempty disjoint regions required"
        )
    rows, acts = {}, {}
    for region, selected in (("A", slice(None, cut)), ("B", slice(cut, None))):
        fields = [v[selected] for v in (g, gp, gm)]
        delta = (fields[1] - fields[2]) / 2
        base_norm = float(np.linalg.norm(fields[0]))
        if base_norm <= 0:
            raise ValueError("Nonzero baseline region gradient required")
        rows[region] = {
            "baseline_norm": base_norm,
            "delta_norm": float(np.linalg.norm(delta)),
            "relative_change": float(np.linalg.norm(delta) / base_norm),
        }
        act = [np.einsum("th,fth->f", f, ds[:, selected]) for f in fields]
        change = np.einsum("th,fth->f", delta, ds[:, selected])
        np.testing.assert_allclose(
            change, (act[1] - act[2]) / 2, atol=1e-10, rtol=1e-10
        )
        acts[region] = dict(
            zip(("original", "plus", "minus", "delta"), [*act, change], strict=True)
        )
    return rows, acts
