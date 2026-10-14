#!/usr/bin/env python3
"""Whole-author source exchange, separate from changing a natural population."""

import math

import numpy as np

from grain_research.construction import math_source_reference_action as S


def strata(codes, counts=None):
    """Partition author addresses; exact conditions restrict, not erase, support."""
    g = np.asarray(codes)
    if g.ndim != 1 or not np.isin(g, np.arange(32)).all():
        raise ValueError("Valid inherited profile codes required")
    keys = g[:, None]
    if counts is not None:
        c = np.asarray(counts)
        if c.shape != (len(g), 2) or not np.isin(c, np.arange(13)).all():
            raise ValueError("Exact counts in both12-event windows required")
        keys = np.column_stack((keys, c))
    return [
        np.flatnonzero(np.all(keys == key, axis=1)) for key in np.unique(keys, axis=0)
    ]


def support(codes, folds, counts=None):
    """Describe the allocation orbit and frozen addresses, not psychological power."""
    g, f = np.asarray(codes), np.asarray(folds)
    if f.shape != g.shape or not np.isin(f, (0, 1)).all():
        raise ValueError("Matched two-bank membership required")
    ns = [np.bincount(g[f == b], minlength=32) for b in (0, 1)]
    if min(int(n.min()) for n in ns) == 0:
        raise ValueError("Missing profile cannot be neutralized")
    groups = strata(g, counts)
    movable = [ix for ix in groups if len(np.unique(f[ix])) == 2]
    orbit = math.prod(math.comb(len(ix), int(np.sum(f[ix] == 0))) for ix in groups)
    return {
        "strata": len(groups),
        "movable_strata": len(movable),
        "movable_authors": sum(map(len, movable)),
        "frozen_authors": len(g) - sum(map(len, movable)),
        "distinct_allocation_orbit_size": str(orbit),
        "movable_authors_by_profile": [
            sum(int(np.sum(g[ix] == k)) for ix in movable) for k in range(32)
        ],
        "profile_bank_counts": [n.tolist() for n in ns],
    }


def exchange(folds, groups, rng):
    """Uniform full symmetric-group action on whole-author bank memberships."""
    f = np.asarray(folds, int).copy()
    for ix in groups:
        f[ix] = rng.permutation(f[ix])
    return f


def kernels(points, prototypes, variance):
    """Precompute fixed positive Gaussian components, never fit a text decoder."""
    x, z, s = np.asarray(points, float), np.asarray(prototypes, float), float(variance)
    if (
        x.ndim != 5
        or x.shape[1:4] != (2, 2, 2)
        or z.ndim != 3
        or z.shape[1:] != x.shape[3:]
        or not all(np.isfinite(a).all() for a in (x, z))
        or not np.isfinite(s)
        or s <= 0
    ):
        raise ValueError(
            "Paired SAME-edit points and registered finite prototypes required"
        )
    k = np.empty((*x.shape[:-1], len(z)))
    for v in (0, 1):
        xx = x[..., v, :]
        distance = (
            np.sum(xx**2, -1)[..., None]
            + np.sum(z[:, v] ** 2, -1)
            - 2 * (xx @ z[:, v].T)
        )
        k[..., v, :] = np.exp(-distance / (2 * s))
    if not np.isfinite(k).all() or np.any(k <= 0):
        raise ValueError("Gaussian audit outside finite positive support")
    return k


def laws(k, codes, folds):
    """Read all three inherited laws for one allocation; keep all profile cells."""
    g, f = np.asarray(codes, int), np.asarray(folds, int)
    support(g, f)
    if k.ndim != 5 or k.shape[1:4] != (2, 2, 2) or k.shape[-1] != len(g):
        raise ValueError("Kernel addresses do not match the whole-author catalogue")
    out = np.empty((2, len(k), 2, 3, 2, 32))
    for b, code in np.ndindex(2, 32):
        a = k[..., (g == code) & (f == b)]
        mean = a.mean(-1)
        out[b, ..., code] = np.stack(
            (
                np.stack((mean[:, 0, :, 0], mean[:, 1, :, 1]), 1),
                a.prod(-2).mean(-1),
                mean.prod(-1),
            ),
            2,
        )
    return out


def orbit_mean(k, codes, folds, groups):
    """Exact density expectation, retaining finite-catalogue product covariance."""
    g, f = np.asarray(codes, int), np.asarray(folds, int)
    support(g, f)
    out = np.zeros((2, len(k), 2, 3, 2, 32))
    for b, code in np.ndindex(2, 32):
        n = int(np.sum((g == code) & (f == b)))
        mean = np.zeros(k.shape[:-1])
        joint = np.zeros(k.shape[:-2])
        covariance = np.zeros_like(joint)
        for ix in groups:
            if g[ix[0]] != code:
                continue
            a, size = k[..., ix], len(ix)
            nb = int(np.sum(f[ix] == b))
            mu = a.mean(-1)
            mean += nb * mu / n
            joint += nb * a.prod(-2).mean(-1) / n
            if size > 1 and 0 < nb < size:
                residual = a - mu[..., None]
                cov = residual.prod(-2).sum(-1) / (size - 1)
                covariance += nb * (1 - nb / size) * cov / n**2
        out[b, ..., code] = np.stack(
            (
                np.stack((mean[:, 0, :, 0], mean[:, 1, :, 1]), 1),
                joint,
                mean.prod(-1) + covariance,
            ),
            2,
        )
    return out


def response(densities):
    """Build the old response certificate on a new source-allocation observation."""
    if np.any(densities <= 0) or not np.isfinite(densities).all():
        raise ValueError("Unobserved or negative density is not a zero response")
    return S.certificate(np.log(densities[0]), np.log(densities[1]))


def statistics(certificate):
    """Return all-law/target RMS with one declared all-target primary."""
    kappa = certificate["conditional_cross_ratio"]
    defect = certificate["response_defect"].mean(1)
    return {
        "cross_ratio_RMS": np.sqrt(np.mean(kappa**2, axis=(0, 1, 4))),
        "response_defect_RMS": np.sqrt(np.mean(defect**2, axis=0)),
        "all5_cross_ratio_RMS": np.sqrt(np.mean(kappa**2, axis=(0, 1, 3, 4))),
    }
