#!/usr/bin/env python3
"""Same-event evidence alone, background interaction and retained common action."""

import numpy as np
from scipy.special import logsumexp


def partition(support, counter):
    """Require two disjoint four-event drivers and a four-event background."""
    s, t = tuple(support), tuple(counter)
    if (
        len(s) != 4
        or len(t) != 4
        or len(set(s)) != 4
        or len(set(t)) != 4
        or set(s) & set(t)
        or any(not isinstance(i, (int, np.integer)) or not 0 <= i < 12 for i in s + t)
    ):
        raise ValueError("Addressed disjoint four-event drivers required")
    return tuple(i for i in range(12) if i not in set(s + t))


def readings(logits):
    """Return five fixed high/low digit log-odds for both driver blocks."""
    a = np.asarray(logits, float)
    if a.ndim != 4 or a.shape[1:] != (2, 5, 10) or not np.isfinite(a).all():
        raise ValueError("Author x2drivers x5targets x10digits required")
    return logsumexp(a[..., 5:], -1) - logsumexp(a[..., :5], -1)


def account(with_background, driver_readings, direction):
    """Retain raw, specific and oriented-common actions without fitted axes."""
    w, y, s = (
        np.asarray(with_background, float),
        np.asarray(driver_readings, float),
        np.asarray(direction, float),
    )
    if (
        w.ndim != 2
        or w.shape[1] != 5
        or y.shape != (len(w), 2, 5)
        or s.shape != (5,)
        or not np.isin(s, [-1, 1]).all()
        or not np.isfinite(w).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Same-author five-target action family required")
    alone = y[:, 0] - y[:, 1]
    raw = np.stack((alone, w, w - alone), 1)
    common = raw.mean(-1)
    oriented = raw @ s / 5
    return {
        "raw": raw,
        "specific": raw - common[..., None],
        "common": common,
        "oriented_common": oriented,
        "oriented_remainder": raw - oriented[..., None] * s,
    }
