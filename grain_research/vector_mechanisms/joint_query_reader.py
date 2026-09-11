#!/usr/bin/env python3
"""Paired absolute functional coordinates and coherent independent-query lifts."""

import numpy as np

from . import evidence_coordinate_reader as coord
from . import teaching_composition as accounting

base = coord.base
VARIANT = "original12_joint_absolute_coordinates"


def energies(decomposition):
    """Retain the existing squared and signed-cross error definitions."""
    return accounting.energies(decomposition)


def origin(baseline):
    """Use the unchanged source common reader and difference origin."""
    return np.stack(
        (baseline["m"] + baseline["d0"] / 2, baseline["m"] - baseline["d0"] / 2)
    )


def frames(primal, vectors, units):
    """Return each query's scalar/four-action rows for the fixed all4 arm."""
    p, v, u = [np.asarray(a, float) for a in (primal, vectors, units)]
    if p.ndim != 3 or p.shape[:2] != (2, 4) or v.shape != (2, 4, 3, 4, p.shape[-1]):
        raise ValueError("Two queries and four compiled compositions required")
    if u.shape != (5,) or not np.isfinite(u).all() or np.any(u <= 0):
        raise ValueError("Five fixed positive units required")
    r = np.concatenate((p[:, 3, None], v[:, 3, 2]), axis=1) / u[None, :, None]
    for row in r:
        coord.frame_geometry(row)
    return r


def teaching(current, native, readers, baseline, units):
    """Construct both teacher definitions without reducing queries to contrast."""
    cur, nat, rr, u = [np.asarray(x, float) for x in (current, native, readers, units)]
    if (
        cur.ndim != 4
        or cur.shape[1:3] != (2, 5)
        or nat.shape != cur.shape
        or rr.shape != (len(cur), 2, cur.shape[-1])
    ):
        raise ValueError("Aligned source paired five-row observations required")
    if (
        u.shape != (5,)
        or np.any(u <= 0)
        or not all(np.isfinite(a).all() for a in (cur, nat, rr, u))
    ):
        raise ValueError("Finite data and positive fixed units required")
    r = cur / u[None, None, :, None]
    for pair in r:
        for row in pair:
            coord.frame_geometry(row)
    z = origin(baseline)
    target = np.einsum("nqkh,nqh->nqk", nat, rr)
    offset = np.einsum("nqkh,qh->nqk", cur, z)
    return {
        "frame": r,
        "target_values": target,
        "offset": offset,
        "compatible_coordinates": (target - offset) / u,
        "native_coordinates": np.einsum("nqkh,nqh->nqk", r, rr - z),
    }


def fit(source, points, target):
    """One source12 solve for ten coordinates; keep existing units/weights/lambda."""
    x, y = np.asarray(points, float), np.asarray(target, float)
    if x.shape != source["x"].shape or len(x) != 12 or y.shape != (12, 2, 5):
        raise ValueError("Twelve fixed source points and ten query observations")
    np.testing.assert_array_equal(x, source["x"])
    if not np.isfinite(y).all() or np.any(source["kinds"]) or np.any(source["v"]):
        raise ValueError("Finite source value observations only")
    k = base.gram(x, source["v"], source["kinds"], x, source["v"], source["kinds"])
    lam = float(source["regularizer"])
    a = k + np.diag(lam / source["weights"] ** 2)
    flat = y.reshape(12, 10)
    beta = np.linalg.solve(a, flat)
    residual = np.linalg.norm(a @ beta - flat) / max(
        np.linalg.norm(flat), np.finfo(float).tiny
    )
    if not np.isfinite(beta).all() or residual > 1e-9:
        raise ValueError("Joint source solve failed")
    return {
        k: source[k] for k in ("x", "v", "kinds", "weights", "regularizer", "units")
    } | {
        "target": flat,
        "fitted": k @ beta,
        "beta": beta,
        "d0": np.zeros(10),
        "coefficients": np.asarray(beta.size),
        "regularized_condition": np.asarray(np.linalg.cond(a)),
        "normal_residual_relative": np.asarray(residual),
    }


def readers(baseline, model, blocks, tangents, r, dr):
    """Differentiate baseline, coordinates and each query's evidence frame."""
    x, dx = [
        base.flatten(a, baseline["scale"], "grain_field") for a in (blocks, tangents)
    ]
    db, ddb = base.field(baseline, x, dx)
    c, dc = base.field(model, x, dx)
    c, dc = c.reshape(2, 5), dc.reshape(-1, 2, 5).transpose(1, 0, 2)
    rb = np.stack((baseline["m"] + db / 2, baseline["m"] - db / 2))
    drb = np.stack((ddb / 2, -ddb / 2))
    z = origin(baseline)
    pairs = [
        coord.lift(rb[q], drb[q], r[q], dr[q], c[q], dc[q], z[q]) for q in range(2)
    ]
    return (
        np.stack([a[0] for a in pairs]),
        np.stack([a[1] for a in pairs]),
        c,
        dc,
        np.stack([a[2] for a in pairs]),
    )


def value_readers(baseline, model, blocks, r):
    """Independent value-only path for whole-function finite differences."""
    x = base.flatten(blocks, baseline["scale"], "grain_field")
    db, c = base.field(baseline, x), base.field(model, x).reshape(2, 5)
    rb = np.stack((baseline["m"] + db / 2, baseline["m"] - db / 2))
    z = origin(baseline)
    return np.stack([coord.lift_value(rb[q], r[q], c[q], z[q]) for q in range(2)])


def modes(r):
    """Query-specific and material-dependent shared reader, including tangents."""
    return np.stack((r, np.broadcast_to(r.mean(axis=0), r.shape)), axis=1)


def projections(primal, vectors, r, dr):
    """Keep common-reader derivatives: they are not zero under joint teaching."""
    rr, dd = modes(r), modes(dr)
    scalar = np.einsum("qah,qmh->qam", primal, rr)
    actions = np.einsum("qacfh,qmh->qacmf", vectors, rr)
    correction = np.einsum("qah,qmfh->qamf", primal, dd)
    return {
        "scalar": scalar,
        "actions": actions,
        "correction": correction,
        "diagonal": actions[:, :, 2] + correction,
    }


def decompose(current, native, reader, native_reader):
    """Contrast error with the current material-dependent common reader."""
    cur, nat, r, nr = [
        np.asarray(a, float) for a in (current, native, reader, native_reader)
    ]
    if (
        cur.ndim != 4
        or cur.shape[1] != 2
        or nat.shape != cur.shape
        or r.shape != (len(cur), 2, cur.shape[-1])
        or nr.shape != r.shape
    ):
        raise ValueError("Paired finite evidence and readers required")
    if not all(np.isfinite(a).all() for a in (cur, nat, r, nr)):
        raise ValueError("Nonfinite decomposition inputs")
    y = np.einsum("nqkh,nqh->nqk", nat, nr)
    p = np.einsum("nqkh,nqh->nqk", cur, r)
    moved = np.einsum("nqkh,nqh->nqk", cur - nat, nr)
    t = moved[:, 0] - moved[:, 1]
    c = np.einsum("nkh,nh->nk", cur[:, 0] - cur[:, 1], r.mean(axis=1) - nr.mean(axis=1))
    d = np.einsum(
        "nkh,nh->nk", cur.mean(axis=1), (r[:, 0] - r[:, 1]) - (nr[:, 0] - nr[:, 1])
    )
    pieces = np.stack((t, c, d), axis=1)
    target, pred = y[:, 0] - y[:, 1], p[:, 0] - p[:, 1]
    error = float(abs(pieces.sum(axis=1) - (pred - target)).max())
    np.testing.assert_allclose(pieces.sum(axis=1), pred - target, atol=1e-10)
    return {
        "target": target,
        "prediction": pred,
        "pieces": pieces,
        "closure_max": error,
    }
