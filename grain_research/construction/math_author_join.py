#!/usr/bin/env python3
"""Label-free author-pair joins and complete behavioral interaction accounts."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from grain_research.construction.math_behavior_predictive import key_accounts


def joint_matching(vectors, lengths, conditions, slots, absolute=8, relative=0.1):
    """Match every author to one stranger using two outcome-free event constraints."""
    x, n, c, s = map(np.asarray, (vectors, lengths, conditions, slots))
    if x.ndim != 4 or x.shape[1] != 2 or n.shape != x.shape[:3] or c.shape != n.shape:
        raise ValueError("Author x two windows x events x vectors required")
    if s.shape != (len(x), 2) or not np.isfinite(x).all():
        raise ValueError("Two addressed own events per author required")
    if np.any(s < 0) or np.any(s >= x.shape[2]) or np.any(n <= 0):
        raise ValueError("Nonempty in-range comments required")
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("Zero vectors cannot define matching")
    x = x / norm
    costs = np.full((len(x), len(x)), np.inf)
    candidates = np.full((len(x), len(x), 2), -1, int)
    for u in range(len(x)):
        for v in range(len(x)):
            if u == v:
                continue
            chosen, total = [], 0.0
            for w in range(2):
                own = s[u, w]
                eligible = np.flatnonzero(
                    (c[v, w] == c[u, w, own])
                    & (
                        abs(n[v, w] - n[u, w, own])
                        <= max(absolute, relative * n[u, w, own])
                    )
                )
                if not len(eligible):
                    break
                distance = 1 - x[v, w, eligible] @ x[u, w, own]
                i = int(np.argmin(distance))
                chosen.append(int(eligible[i]))
                total += float(distance[i])
            if len(chosen) == 2:
                costs[u, v], candidates[u, v] = total, chosen
    rows, donors = linear_sum_assignment(costs)
    if (
        not np.array_equal(rows, np.arange(len(x)))
        or not np.isfinite(costs[rows, donors]).all()
    ):
        raise ValueError("No full same-condition two-event stranger derangement")
    return donors, candidates[rows, donors], costs[rows, donors]


def mixed_difference(values):
    """Keep the four registered corners 00,01,10,11 in one exact mixed contrast."""
    x = np.asarray(values, float)
    if x.ndim < 2 or x.shape[-2] != 4 or not np.isfinite(x).all():
        raise ValueError("Finite complete four-corner accounts required")
    return x[..., 0, :] - x[..., 1, :] - x[..., 2, :] + x[..., 3, :]


def interaction_accounts(chosen, partition, lengths, bank):
    """Return signed candidate, balanced/common/keyed and even/odd join accounts."""
    z, a, n = (
        np.asarray(chosen, float),
        np.asarray(partition, float),
        np.asarray(lengths, int),
    )
    if (
        z.ndim != 4
        or z.shape[1:] != (2, 4, 50)
        or a.shape != z.shape
        or n.shape != (50,)
    ):
        raise ValueError(
            "Every author x two orders x four corners x50 candidates required"
        )
    if min(n) < 2 or not np.isfinite(z).all() or not np.isfinite(a).all():
        raise ValueError("No imputation or incomplete behavior event")
    result = {}
    for scale, divisor in (("sum", 1), ("per_token", n)):
        numer, normal = mixed_difference(z / divisor), mixed_difference(a / divisor)
        values = numer - normal
        accounts = key_accounts(values, bank)
        result[scale] = {
            "item_interaction": values,
            "chosen_interaction": numer,
            "normalizer_interaction": normal,
            **accounts,
            "order_even": (
                accounts["balanced_key"][:, 0] + accounts["balanced_key"][:, 1]
            )
            / 2,
            "order_odd": (
                accounts["balanced_key"][:, 0] - accounts["balanced_key"][:, 1]
            )
            / 2,
        }
    return result


def source_contacts(differences, effects):
    """Contact both same-event SOURCE secants with each joint behavior effect, without fit."""
    x, y = np.asarray(differences, float), np.asarray(effects, float)
    if x.ndim != 3 or x.shape[1:] != (2, 5) or y.shape != (len(x), 5):
        raise ValueError(
            "Addressed two-event SOURCE actions and five behavior effects required"
        )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite source contact")
    return np.einsum("up,ur,uq->prq", x[:, 0], x[:, 1], y) / len(x)
