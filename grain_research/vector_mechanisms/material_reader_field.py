#!/usr/bin/env python3
"""Source-weighted Gaussian value/derivative regression for a joint-input reader."""

import numpy as np

CONTRACTS = ("context_field", "body_mean_field", "grain_field")
BLOCKS = ("context", "body_mean", "G01", "G02", "G03", "G04")
GROUPS = ("value", "body", "context")


def features(data, directions=None):
    """Compute shared BODY moments and optional tangents with frozen GRAIN values."""
    if len(data) != 2:
        raise ValueError("Two fixed query states required")
    a, b = data
    pos = a["positions"]
    np.testing.assert_array_equal(pos, b["positions"])
    np.testing.assert_array_equal(a["state"][pos], b["state"][pos])
    np.testing.assert_array_equal(a["grain"], b["grain"])
    h, g = a["state"][pos].astype(float), a["grain"].astype(float)
    gc = g - g.mean(axis=0)
    mean = h.mean(axis=0)
    cov = gc.T @ (h - mean) / len(h)
    result = [
        np.concatenate([x["state"][-1] for x in data]).astype(float) / np.sqrt(2),
        mean,
        *cov,
    ]
    if directions is None:
        return result
    da, db = directions
    if (
        da.ndim != 3
        or da.shape[1:] != a["state"].shape
        or db.shape != (len(da), *b["state"].shape)
    ):
        raise ValueError("Aligned multiple full-state tangents required")
    np.testing.assert_array_equal(da[:, pos], db[:, pos])
    dc = np.concatenate((da[:, -1], db[:, -1]), axis=1) / np.sqrt(2)
    dm = da[:, pos].mean(axis=1)
    dg = np.einsum("tf,nth->fnh", gc, da[:, pos]) / len(pos)
    return result, [dc, dm, *dg]


def scales(source_blocks):
    """Only the four original source values set each median-distance unit."""
    result = []
    for block in range(6):
        x = np.stack([s[block] for s in source_blocks])
        distances = np.linalg.norm(x[:, None] - x[None], axis=-1)
        result.append(float(np.median(distances[np.triu_indices(len(x), 1)])))
    return np.asarray(result)


def flatten(blocks, scale, contract):
    """Apply a fixed nested input contract to values or batches of tangents."""
    limit = dict(zip(CONTRACTS, (1, 2, 6), strict=True))[contract]
    active = [i for i in range(limit) if scale[i] > 0]
    if not active or not np.isfinite(scale).all():
        raise ValueError("Nondegenerate finite source feature units required")
    return np.concatenate([blocks[i] / scale[i] for i in active], axis=-1) / np.sqrt(
        len(active)
    )


def gram(x, v, derivative, z, w, other_derivative):
    """Evaluate Gaussian value/directional-derivative covariance without whitening."""
    x, v, z, w = [np.asarray(a, float) for a in (x, v, z, w)]
    left, right = np.asarray(derivative, bool), np.asarray(other_derivative, bool)
    if (
        x.ndim != 2
        or z.ndim != 2
        or v.shape != x.shape
        or w.shape != z.shape
        or x.shape[1] != z.shape[1]
    ):
        raise ValueError("Aligned observation positions and tangents required")
    if (
        left.shape != (len(x),)
        or right.shape != (len(z),)
        or not all(np.isfinite(a).all() for a in (x, v, z, w))
    ):
        raise ValueError("Finite observation arrays and kinds required")
    delta = x[:, None] - z[None]
    k = np.exp(-0.5 * np.sum(delta * delta, axis=-1))
    lv = np.einsum("ijp,ip->ij", delta, v)
    rw = np.einsum("ijp,jp->ij", delta, w)
    multiplier = np.where(left[:, None], -lv, 1.0) * np.where(right[None], rw, 1.0)
    multiplier += (left[:, None] & right[None]) * (v @ w.T)
    return k * multiplier


def observation_weights(target, groups):
    """Give the three fixed source target groups equal total squared-energy units."""
    weights = np.ones(len(target))
    zero = []
    for gi in range(3):
        use = groups == gi
        energy = float(np.sum(target[use] ** 2))
        weights[use] = 1 / np.sqrt(energy) if energy else 1.0
        zero.append(not bool(energy))
    return weights, np.asarray(zero)


def fit(x, v, kinds, target, groups, d0, mean_reader):
    """One fixed regularized vector solve using source values and derivative rows."""
    kernel = gram(x, v, kinds, x, v, kinds)
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-10)
    weights, zero = observation_weights(target, groups)
    kw = weights[:, None] * kernel * weights[None]
    eig = np.linalg.eigvalsh(kw)
    if eig[0] < -1e-9 * max(1.0, eig[-1]):
        raise ValueError("Derivative kernel lost positive semidefiniteness")
    lam = 0.001 * float(np.trace(kw)) / len(kw)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("Finite positive fixed regularizer required")
    beta = weights[:, None] * np.linalg.solve(
        kw + lam * np.eye(len(kw)), weights[:, None] * target
    )
    fitted = kernel @ beta
    losses = []
    for gi in range(3):
        use = groups == gi
        energy = float(np.sum(target[use] ** 2))
        losses.append(
            float(np.sum((fitted[use] - target[use]) ** 2) / energy) if energy else 0.0
        )
    return {
        "x": x,
        "v": v,
        "kinds": kinds,
        "beta": beta,
        "weights": weights,
        "d0": d0,
        "m": mean_reader,
        "group_loss": np.asarray(losses),
        "zero_target_groups": zero,
        "regularizer": np.asarray(lam),
        "kernel_min_eigenvalue": np.asarray(eig[0]),
        "kernel_rank": np.asarray(
            np.sum(eig > max(eig[-1], 1.0) * len(eig) * np.finfo(float).eps)
        ),
        "regularized_condition": np.asarray((eig[-1] + lam) / (eig[0] + lam)),
        "zero_operator_rows": np.asarray(np.sum(np.diag(kernel) == 0)),
    }


def field(model, x, dx=None):
    """Evaluate d and optional material tangents without current neural targets."""
    x = np.asarray(x, float)
    row = gram(
        x[None], np.zeros_like(x)[None], [False], model["x"], model["v"], model["kinds"]
    )
    d = model["d0"] + (row @ model["beta"])[0]
    if dx is None:
        return d
    dx = np.atleast_2d(dx)
    rows = gram(
        np.broadcast_to(x, dx.shape),
        dx,
        np.ones(len(dx), bool),
        model["x"],
        model["v"],
        model["kinds"],
    )
    return d, rows @ model["beta"]


def readers(model, blocks, tangents, scale, contract):
    """Return paired readers and their BODY derivatives for the same feature map."""
    x, dx = [flatten(z, scale, contract) for z in (blocks, tangents)]
    d, dd = field(model, x, dx)
    r = np.stack((model["m"] + d / 2, model["m"] - d / 2))
    dr = np.stack((dd / 2, -dd / 2))
    return r, dr
