#!/usr/bin/env python3
"""Bound possible source mixtures from multi-field liquid observations on a shared clock."""

import argparse

import numpy as np
from scipy.optimize import linprog

from . import liquid_instrument as li


def bank(score, episode):
    """Each condition is a separately measured complete response, not an additive edit."""
    e = score["episodes"][episode]
    f = e["nine_fields"]
    y = np.array(f["frozen_reader"], float)
    dy = np.array(f["source_deletion_delta"], float)
    if dy.shape != (len(y), *y.shape) or len(f["names"]) != y.shape[1]:
        raise ValueError("missing source-by-time-by-field records")
    return (
        ["as_observed", *[f"omit_slice_{k}" for k in range(len(y))]],
        np.concatenate([y[None], y - dy]),
        f["names"],
    )


def bounds(a, b, tolerance):
    """Return coordinate bounds and independent endpoint witnesses, never a best-fit point."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if (
        a.ndim != 2
        or b.shape != (len(a),)
        or not a.size
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or not np.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("invalid declared measurement constraints")
    n = a.shape[1]
    # Conditioning removes common readout level, preserving the ORIGINAL absolute tolerance.
    offset = a[:, 0].copy()
    z = a - offset[:, None]
    scale = np.maximum(np.abs(z).max(1), 1e-9)
    z = z / scale[:, None]
    t = (b - offset) / scale
    tol = tolerance / scale
    kwargs = {
        "A_eq": np.ones((1, n)),
        "b_eq": [1.0],
        "A_ub": np.vstack([z, -z]),
        "b_ub": np.r_[t + tol, -t + tol],
        "bounds": (0, None),
        "method": "highs",
        "options": {
            "dual_feasibility_tolerance": 1e-9,
            "primal_feasibility_tolerance": 1e-9,
        },
    }
    feasible = linprog(np.zeros(n), **kwargs)
    if feasible.status == 2:
        return {"status": "outside_declared_source_bank", "bounds": None}
    if not feasible.success:
        raise RuntimeError(feasible.message)
    intervals = []
    witnesses = []
    for c in np.eye(n):
        ends = []
        for sign in (1, -1):
            sol = linprog(sign * c, **kwargs)
            if not sol.success:
                raise RuntimeError(sol.message)
            if max(abs(a @ sol.x - b)) > tolerance + 1e-8 * max(1.0, np.abs(a).max()):
                raise ArithmeticError("endpoint fails in original observation units")
            ends.append(sol.x)
        intervals.append([float(c @ x) for x in ends])
        witnesses.append(np.array(ends).tolist())
    return {
        "status": "compatible",
        "bounds": intervals,
        "endpoint_witnesses": witnesses,
    }


def estimate(score, q):
    if q["reference_digest"] != score["reference_digest"]:
        raise ValueError("different reader reference")
    source_names, profiles, names = bank(score, q["episode"])
    if q["unit_digests"] != score["episodes"][q["episode"]]["unit_digests"]:
        raise ValueError("different source sequence")
    cells = q["observations"]
    rows = []
    ys = []
    groups = {}
    used = set()
    for c in cells:
        k = c["time"]
        field = c["field"]
        if (
            not isinstance(k, int)
            or not 0 <= k < profiles.shape[1]
            or (k, field) in used
        ):
            raise ValueError("unknown time or duplicate time/field lookup")
        used.add((k, field))
        groups.setdefault(field, []).append(len(rows))
        rows.append(profiles[:, k, names.index(field)])
        ys.append(c["value"])
    a, b = np.array(rows), np.array(ys)
    tol = q["absolute_tolerance"]
    joint = bounds(a, b, tol)
    return {
        "version": "liquid-field-inverse-v1",
        "source_bank": source_names,
        "field_reference": q["reference_digest"],
        "query_digest": li.digest(q),
        "absolute_tolerance": tol,
        "single_fields": {f: bounds(a[ix], b[ix], tol) for f, ix in groups.items()},
        "joint": joint,
        "interpretation": "alternative source interventions averaged across draws; not simultaneous edits or unique psychological sources",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("score", "query", "out"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    li.save(a.out, estimate(li.load(a.score), li.load(a.query)))


if __name__ == "__main__":
    main()
