#!/usr/bin/env python3
"""Separate a reusable propensity from a consumable finite evidence inventory."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

VERSION = "HM196_evidence_sampling_v1"
REGIMES = ("propensity_replacement", "finite_inventory")


def validate(counts, targets):
    """Require aligned integer inventories, equal exposure and five SOURCE targets."""
    n, y = np.asarray(counts), np.asarray(targets, float)
    if (
        n.ndim != 2
        or n.shape[1] != 8
        or len(n) < 2
        or not np.issubdtype(n.dtype, np.integer)
        or np.any(n < 0)
        or np.any(n.sum(1) <= 0)
        or not np.all(n.sum(1) == n.sum(1)[0])
        or y.shape != (len(n), 5)
        or not np.isin(y, (0, 1)).all()
    ):
        raise ValueError(
            "Equal finite eight-cell inventories and five targets required"
        )
    return n, y


def empirical_counts(probabilities, exposure, tolerance=1e-10):
    """Restore exact cached event counts without fitting a new observation law."""
    p = np.asarray(probabilities, float)
    if (
        p.ndim != 2
        or p.shape[1] != 8
        or type(exposure) is not int
        or exposure <= 0
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or tolerance <= 0
    ):
        raise ValueError("Finite probabilities and positive declared exposure required")
    n = np.rint(exposure * p).astype(int)
    if np.any(n.sum(1) != exposure) or np.max(np.abs(n / exposure - p)) > tolerance:
        raise ValueError("SOURCE probabilities are not the declared finite inventory")
    return n


def state(counts, targets, used, log_weights, regime):
    """Read target means and joint next-event action with its declared sampling law."""
    n, y = validate(counts, targets)
    k, lw = np.asarray(used), np.asarray(log_weights, float)
    if (
        regime not in REGIMES
        or k.shape != (8,)
        or not np.issubdtype(k.dtype, np.integer)
        or np.any(k < 0)
        or lw.shape != (len(n),)
        or np.isnan(lw).any()
        or np.isposinf(lw).any()
    ):
        raise ValueError(
            "Declared sampling regime and consistent evidence state required"
        )
    normalizer = logsumexp(lw)
    if not np.isfinite(normalizer):
        return {"supported": False, "mu": None, "pi": None, "after": None}
    w = np.exp(lw - normalizer)
    total = int(n.sum(1)[0])
    mu = w @ y
    if regime == REGIMES[0]:
        p = n / total
    else:
        if np.any(n[w > 0] < k):
            raise ValueError("Positive weight assigned to an exhausted inventory")
        if k.sum() >= total:
            return {"supported": True, "mu": mu, "pi": None, "after": None}
        p = np.maximum(n - k, 0) / (total - k.sum())
    pi = w @ p
    joint = y.T @ (w[:, None] * p)
    after = np.divide(
        joint, pi[None, :], out=np.full_like(joint, np.nan), where=pi[None, :] > 0
    )
    return {"supported": True, "mu": mu, "pi": pi, "after": after}


def replay(counts, targets, events, regime):
    """Update on unique identified events; preserve unsupported evidence explicitly."""
    n, y = validate(counts, targets)
    if regime not in REGIMES:
        raise ValueError("Unknown observation law")
    total = int(n.sum(1)[0])
    used = np.zeros(8, int)
    lw = np.full(len(n), -np.log(len(n)))
    seen, rows = {}, []
    for event in events:
        if (
            set(event) != {"event_id", "cell"}
            or not isinstance(event["event_id"], str)
            or not event["event_id"]
            or type(event["cell"]) is not int
            or event["cell"] not in range(8)
        ):
            raise ValueError("Only identified numeric event observations allowed")
        identity, a = event["event_id"], event["cell"]
        if identity in seen:
            if seen[identity] != a:
                raise ValueError(
                    "The same event identity cannot change observation cell"
                )
            continue
        before = state(n, y, used, lw, regime)
        if regime == REGIMES[0]:
            likelihood = n[:, a] / total
        elif used.sum() < total:
            likelihood = np.maximum(n[:, a] - used[a], 0) / (total - used.sum())
        else:
            likelihood = np.zeros(len(n))
        with np.errstate(divide="ignore"):
            lw = lw + np.log(likelihood)
        used[a] += 1
        seen[identity] = a
        after = state(n, y, used, lw, regime)
        predicted = None
        if before["supported"] and before["after"] is not None:
            predicted = before["after"][:, a]
        rows.append(
            {
                "event_id": identity,
                "cell": a,
                "used": used.copy(),
                "before": before["mu"],
                "after": after["mu"],
                "predicted": predicted,
                "supported": after["supported"],
                "remaining_source_authors": int(np.isfinite(lw).sum()),
                "next_probabilities": after["pi"],
            }
        )
    return rows, state(n, y, used, lw, regime)


def direct_posterior(counts, targets, cells, regime):
    """Independent ordered-word likelihood via literal falling-factorial products."""
    n, y = validate(counts, targets)
    a = np.asarray(cells)
    if (
        a.ndim != 1
        or (a.size and not np.issubdtype(a.dtype, np.integer))
        or not np.isin(a, np.arange(8)).all()
        or regime not in REGIMES
    ):
        raise ValueError("Valid finite word and declared regime required")
    a = a.astype(int)
    k, total = np.bincount(a, minlength=8), int(n.sum(1)[0])
    likelihood = np.ones(len(n))
    if regime == REGIMES[0]:
        for cell in a:
            likelihood *= n[:, cell] / total
    elif len(a) > total:
        likelihood[:] = 0
    else:
        for cell in range(8):
            for used in range(k[cell]):
                likelihood *= np.maximum(n[:, cell] - used, 0)
        for used in range(len(a)):
            likelihood /= total - used
    if likelihood.sum() == 0:
        return None
    return likelihood @ y / likelihood.sum()
