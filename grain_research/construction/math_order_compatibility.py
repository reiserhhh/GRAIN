#!/usr/bin/env python3
"""Witness bounds for ordered SOURCE-to-reader maps, never a fitted scorer."""

import numpy as np


def _values(source, reader):
    """Validate finite observations without silently imputing missing accounts."""
    s, y = np.asarray(source, float), np.asarray(reader, float)
    if s.ndim == 1:
        s = s[:, None]
    if (
        s.ndim != 2
        or y.shape != (len(s),)
        or len(s) < 2
        or s.shape[1] < 1
        or not np.isfinite(s).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("At least two finite addressed SOURCE/reader observations")
    return s, y


def order_bound(source, reader, *, source_error=0.0, reader_error=0.0):
    """Bound every isotonic map on a scalar or componentwise partial order.

    The exact finite order includes equal coordinates. The guarded order uses
    strict separation in every coordinate, keeping near ties incomparable.
    Its adjusted half-gap is only a computational-error lower bound. No map,
    direction, parameters or predictions are fitted.
    """
    s, y = _values(source, reader)
    if (
        not np.isfinite([source_error, reader_error]).all()
        or source_error < 0
        or reader_error < 0
    ):
        raise ValueError("Finite nonnegative computational discrepancies required")
    differences = s[None, :, :] - s[:, None, :]
    exact = np.all(differences >= 0, -1)
    guarded = np.all(differences > 2 * source_error, -1)
    np.fill_diagonal(exact, False)
    np.fill_diagonal(guarded, False)
    gaps = y[:, None] - y[None, :]

    def account(mask, adjustment):
        """Keep missing order separate from a zero violation of observed order."""
        n = int(mask.sum())
        if n == 0:
            return {"constraints": 0, "violations": 0, "bound": None, "witness": None}
        constrained = np.where(mask, gaps, -np.inf)
        i, j = np.unravel_index(np.argmax(constrained), constrained.shape)
        value = max(0.0, float(gaps[i, j]) / 2 - adjustment)
        return {
            "constraints": n,
            "violations": int(np.count_nonzero(mask & (gaps > 2 * adjustment))),
            "bound": value,
            "witness": [int(i), int(j)],
        }

    return {
        "exact": account(exact, 0),
        "guarded": account(guarded, reader_error),
        "source_gap": differences,
        "reader_gap": gaps,
        "exact_order": exact,
        "guarded_order": guarded,
    }


def scalar_orientations(source, reader, **errors):
    """Retain increasing/decreasing and the lower bound for their union."""
    s = np.asarray(source, float)
    if s.ndim != 1:
        raise ValueError("Scalar target coordinate required")
    plus, minus = order_bound(s, reader, **errors), order_bound(-s, reader, **errors)
    result = {"increasing": plus, "decreasing": minus}
    for arm in ("exact", "guarded"):
        p, m = plus[arm]["bound"], minus[arm]["bound"]
        result[arm + "_either_bound"] = None if p is None or m is None else min(p, m)
    return result


def unique_observations(keys, source, reader, indices):
    """Collapse exact token aliases within author, rejecting changed observations."""
    s, y, ix = np.asarray(source, float), np.asarray(reader, float), np.asarray(indices)
    if len(keys) != len(s) or len(s) != len(y) or ix.shape != (len(s),):
        raise ValueError("Addressed source, reader and removal observations required")
    seen, keep, aliases = {}, [], []
    for j, key in enumerate(keys):
        if not isinstance(key, str) or len(key) != 64:
            raise ValueError("Exact five-prompt token hash required")
        if key not in seen:
            seen[key] = j
            keep.append(j)
        else:
            i = seen[key]
            if (
                ix[i] != ix[j]
                or not np.array_equal(s[i], s[j])
                or not np.array_equal(y[i], y[j])
            ):
                raise ValueError("Token alias has changed SOURCE/reader/removal values")
            aliases.append([i, j])
    if len(keep) < 2:
        raise ValueError("At least two unique inputs required")
    return np.asarray(keep, int), aliases
