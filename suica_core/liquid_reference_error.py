#!/usr/bin/env python3
"""Frozen reference and resampling operators for matched original/liquid readers."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .foundation_reference_frame import fit_reference_frame


def calibration_member(person, namespace):
    digest = hashlib.sha256(f"{namespace}:{person}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 2 == 0


def fit_chart(x, axes, *, population, basis, ridge=1e-4, counts=None, event_var=None):
    """One vector per independent reference unit, not one vote per text."""
    x = np.asarray(x, float)
    axes = list(axes)
    if (
        x.ndim != 2
        or len(x) <= x.shape[1]
        or x.shape[1] != len(axes)
        or not axes
        or len(set(axes)) != len(axes)
        or not population
        or not basis
        or not np.isfinite(x).all()
        or not np.isfinite(ridge)
        or ridge <= 0
    ):
        raise ValueError("invalid reference units, coordinate names, or provenance")
    n = len(x)
    noise = None
    if counts is not None or event_var is not None:
        from .stream_precision import eb_fit, eb_apply
        noise = eb_fit(x,counts,event_var)
        x = eb_apply(x,counts,noise)
    ref = fit_reference_frame(
        {
            "means": x[:, None, None, :],
            "counts": np.ones((n, 1, 1)),
            "groups": np.zeros(n, int),
            "facet_provenance": True,
        },
        lambda_facet=np.ones(1),
        target_group_weights=np.ones(1),
        ridge=ridge,
    )
    return {
        "schema": "liquid-population-1",
        **({"precision_chart": noise} if noise is not None else {}),
        "axes": axes,
        "n": n,
        "population": population,
        "basis": basis,
        "ridge": ridge,
        "center": ref["center"].tolist(),
        "whitener": ref["inverse_sqrt"].tolist(),
        "sorted_axes": np.sort(x, axis=0).T.tolist(),
    }


def locate(chart, x, *, axes, basis, counts=None):
    """Return per-axis midrank percentiles and joint whitened distance."""
    x = np.asarray(x, float)
    if (
        chart["schema"] != "liquid-population-1"
        or basis != chart["basis"]
        or list(axes) != chart["axes"]
        or x.ndim != 2
        or x.shape[1] != len(axes)
        or not np.isfinite(x).all()
    ):
        raise ValueError("incompatible chart or observations")
    if "precision_chart" in chart:
        from .stream_precision import eb_apply
        x = eb_apply(x,counts,chart["precision_chart"])
    elif counts is not None:
        raise ValueError("counts supplied for a chart without calibrated event variance")
    pct = np.stack(
        [
            (np.searchsorted(a, x[:, j], "left") + np.searchsorted(a, x[:, j], "right"))
            / (2 * chart["n"])
            for j, a in enumerate(chart["sorted_axes"])
        ],
        axis=1,
    )
    coords = (x - chart["center"]) @ chart["whitener"]
    return pct, np.linalg.norm(coords, axis=1)


def resample_runs(runs, rng):
    """Resample whole episodes, returning boundaries even for repeated draws."""
    if not runs or any(len(r) == 0 for r in runs):
        raise ValueError("empty run")
    return [runs[j] for j in rng.integers(len(runs), size=len(runs))]


def matched_endpoints(values, retention):
    """Per-cell transports; each comparison preserves its own first-block survival."""
    v, a = np.asarray(values, float), np.asarray(retention, float)
    if a.ndim == 2:
        a = a[:, :, None]
    if (
        v.ndim != 3
        or v.shape[1] < 2
        or a.ndim != 3
        or not a.shape[2]
        or a.shape[:2] != v.shape[:2]
        or not np.isfinite(v).all()
        or not np.isfinite(a).all()
        or np.any(a <= 0)
        or np.any(a >= 1)
    ):
        raise ValueError("require finite complete episodes with open-interval carry")
    k = np.exp(np.log(a[:, 1:]).mean(axis=1))[:, :, None]
    liquid = np.repeat(v[:, 0, None, :], a.shape[2], axis=1)
    fixed = liquid.copy()
    for t in range(1, v.shape[1]):
        r = a[:, t, :, None]
        liquid = r * liquid + (1 - r) * v[:, t, None, :]
        fixed = k * fixed + (1 - k) * v[:, t, None, :]
    return liquid.mean(axis=1), fixed.mean(axis=1)


def endpoint_weights(retention):
    """Exact input weights conditional on an observed cell-carry path, not a gradient."""
    a = np.asarray(retention, float)
    if (
        a.ndim != 3
        or min(a.shape) < 1
        or not np.isfinite(a).all()
        or (a < 0).any()
        or (a > 1).any()
    ):
        raise ValueError("expected batch by event by cell carries in [0,1]")
    remaining = np.ones((a.shape[0], a.shape[2]))
    w = np.empty(a.shape[:2])
    for i in range(a.shape[1] - 1, 0, -1):
        w[:, i] = ((1 - a[:, i]) * remaining).mean(1)
        remaining *= a[:, i]
    w[:, 0] = remaining.mean(1)
    return w


def population_io(mode, document, *, chart=None):
    """Fit or place independently aggregated units in a declared identity namespace."""
    namespace = document["namespace"]
    rows = document["units"]
    if not isinstance(namespace, str) or not namespace or not rows:
        raise ValueError("empty namespace or population")
    ids = [r["id"] for r in rows]
    if not all(isinstance(i, str) and i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("expected one nonempty string ID per independent unit")
    tokens = [
        hashlib.sha256(
            json.dumps([namespace, i], ensure_ascii=False).encode()
        ).hexdigest()
        for i in ids
    ]
    values = [r["values"] for r in rows]
    args = {"axes": document["axes"], "basis": document["basis"]}
    if any("count" in row for row in rows):
        args["counts"] = [row.get("count") for row in rows]
    if mode == "fit":
        c = fit_chart(values, population=document["population"], event_var=document.get("event_variance"), **args)
        return {**c, "namespace": namespace, "reference_units": sorted(tokens)}
    if mode != "locate" or chart is None or chart.get("namespace") != namespace:
        raise ValueError("wrong operation or undeclared identity namespace")
    if set(tokens) & set(chart["reference_units"]):
        raise ValueError("a target was a calibration unit")
    p, d = locate(chart, values, **args)
    return {
        "basis": chart["basis"],
        "axes": chart["axes"],
        "population": chart["population"],
        "reference_n": chart["n"],
        "units": [
            {"id": i, "percentiles_0_1": v.tolist(), "joint_distance": float(z)}
            for i, v, z in zip(ids, p, d, strict=True)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "locate"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--chart", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    a = parser.parse_args()
    if a.out.exists() or bool(a.chart) != (a.mode == "locate"):
        parser.error("fresh output required; supply --chart only for locate")
    result = population_io(
        a.mode,
        json.loads(a.input.read_text()),
        chart=json.loads(a.chart.read_text()) if a.chart else None,
    )
    with a.out.open("x") as f:
        json.dump(result, f, allow_nan=False)


if __name__ == "__main__":
    main()
