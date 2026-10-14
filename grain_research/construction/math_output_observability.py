#!/usr/bin/env python3
"""SOURCE-qualified hidden responses measured in a decoder-visible Fisher geometry."""

from __future__ import annotations

import numpy as np

CHANNELS = ("full", "binary_answer", "alphabet_routing", "other_vocabulary")


def key_matrix(bank):
    """Use every public item with balanced poles; never learn a query direction."""
    if len(bank) != 50 or sorted(r["item"] for r in bank) != list(range(50)):
        raise ValueError("Fifty uniquely addressed complete items required")
    matrix = np.zeros((5, 50))
    for q in range(5):
        poles = [
            [r["item"] for r in bank if r["target"] == q and r["pole"] == p]
            for p in (1, -1)
        ]
        if not all(poles) or sum(map(len, poles)) != 10:
            raise ValueError("Both complete poles required for every query")
        for sign, ix in zip((1, -1), poles, strict=True):
            matrix[q, ix] = sign / (2 * len(ix))
    return matrix


def joint_fields(states, bank):
    """Form exchange-even pair minus singleton hidden responses at identical items."""
    x = np.asarray(states, float)
    if x.ndim != 5 or x.shape[:4] != (48, 2, 4, 50) or not np.isfinite(x).all():
        raise ValueError("All48 pairs/two operations/four contexts/50 hidden states")
    keyed = np.einsum("uocjd,qj->uocqd", x, key_matrix(bank))
    return (keyed[:, :, 0] + keyed[:, :, 1] - keyed[:, :, 2] - keyed[:, :, 3]) / 2


def fisher_from_sums(second, mean, binary_probabilities, binary_fields):
    """Exact Fisher chain rule for answer-conditioned, routing and remaining tokens."""
    k, mu, pa, za = (
        np.asarray(x, float)
        for x in (second, mean, binary_probabilities, binary_fields)
    )
    n = len(mu)
    if (
        k.shape != (n, n)
        or pa.shape != (2,)
        or za.shape != (n, 2)
        or not np.isfinite(k).all()
        or not np.isfinite(mu).all()
    ):
        raise ValueError("Aligned full output moments and both answer rows required")
    mass, rest = pa.sum(), 1 - pa.sum()
    if np.min(pa) <= 0 or not 0 < mass < 1:
        raise ValueError("Both answer tokens and outside vocabulary need support")
    a_mean = za @ pa / mass
    r_mean = (mu - za @ pa) / rest
    full = k - np.outer(mu, mu)
    binary = pa.prod() / mass * np.outer(za[:, 1] - za[:, 0], za[:, 1] - za[:, 0])
    routing = mass * rest * np.outer(a_mean - r_mean, a_mean - r_mean)
    outside = k - (za * pa) @ za.T - rest * np.outer(r_mean, r_mean)
    return np.array([full, binary, routing, outside])


def fisher_parts(logit_fields, probabilities, alphabet):
    """Dense finite fixture/API; production accumulation may stream vocabulary rows."""
    z, p, ids = (
        np.asarray(logit_fields, float),
        np.asarray(probabilities, float),
        np.asarray(alphabet),
    )
    if (
        z.ndim != 2
        or p.shape != (z.shape[1],)
        or ids.shape != (2,)
        or ids.dtype.kind not in "iu"
        or len(set(ids)) != 2
        or np.any((ids < 0) | (ids >= len(p)))
        or not np.isfinite(z).all()
        or not np.isfinite(p).all()
        or np.min(p) <= 0
        or abs(p.sum() - 1) > 1e-10
    ):
        raise ValueError(
            "A normalized fixed vocabulary law and unique supported alphabet"
        )
    return fisher_from_sums((z * p) @ z.T, z @ p, p[ids], z[:, ids])


def centered_scores(scores):
    """Center finite covariance coefficients, not source text or condition signals."""
    x = np.asarray(scores, float)
    if x.ndim < 1 or not np.isfinite(x).all():
        raise ValueError("Finite aligned source readings required")
    return x - x.mean(axis=-1, keepdims=True)


def cross_author_accounts(left, right, kernel):
    """Keep same-author diagonal and distinct-author products separately, with signs."""
    a, b, k = (
        np.asarray(left, float),
        np.asarray(right, float),
        np.asarray(kernel, float),
    )
    n = a.shape[-1]
    if (
        n < 2
        or a.shape != b.shape
        or k.shape != (n, n)
        or not all(np.isfinite(x).all() for x in (a, b, k))
    ):
        raise ValueError("Two disjoint views of at least two aligned authors required")
    product = np.einsum("...i,ij,...j->...", a, k, b)
    diagonal = np.sum(a * b * np.diag(k), axis=-1)
    return {
        "all_pairs": product / n**2,
        "same_author": diagonal / n,
        "distinct_authors": (product - diagonal) / (n * (n - 1)),
    }


def orbit_mean(left_weights, right_weights, covariance, kernel):
    """Shared SOURCE assignment has a NONZERO quadratic null mean in general."""
    a, b, v, k = (
        np.asarray(x, float) for x in (left_weights, right_weights, covariance, kernel)
    )
    n = len(a)
    if (
        a.shape != b.shape
        or v.shape != (a.shape[1], a.shape[1])
        or k.shape != (n, n)
        or n < 2
    ):
        raise ValueError(
            "Aligned source pullbacks and exact conditional orbit covariance"
        )
    a, b = a - a.mean(0), b - b.mean(0)
    cross = a @ v @ b.T
    return float(
        (np.sum(cross * k) - np.sum(np.diag(cross) * np.diag(k))) / (n * (n - 1))
    )


def cold(record):
    """Recompute qualified numeric response accounts, not a human personality score."""
    expected = {
        "version",
        "author_keys",
        "left_event_ids",
        "right_event_ids",
        "source_readings",
        "cross_output_kernels",
    }
    if set(record) != expected or record["version"] != "HM284-v1":
        raise ValueError(
            "Only registered numeric observation accounts, no labels/answers"
        )
    authors = record["author_keys"]
    n = len(authors)
    if (
        n < 2
        or len(set(authors)) != n
        or any(not isinstance(a, str) or not a for a in authors)
    ):
        raise ValueError("Unique opaque author identities required")
    events = record["left_event_ids"] + record["right_event_ids"]
    if len(events) != 2 * n or any(len(row) != 2 for row in events):
        raise ValueError("Two identified disjoint-comment pairs per author")
    all_ids = [value for row in events for value in row]
    if len(set(all_ids)) != len(all_ids) or any(
        not isinstance(a, str) or not a for a in all_ids
    ):
        raise ValueError("Duplicate or unqualified observed events")
    scores, kernels = (
        np.asarray(record["source_readings"], float),
        np.asarray(record["cross_output_kernels"], float),
    )
    if (
        scores.shape != (2, n, 5)
        or kernels.shape != (4, 5, n, n)
        or not np.isfinite(scores).all()
        or not np.isfinite(kernels).all()
        or np.max(abs(kernels[0] - kernels[1:].sum(0))) > 1e-8
    ):
        raise ValueError("Complete SOURCE readings and all retained output channels")
    rows = []
    for q in range(5):
        a, b = centered_scores(scores[0, :, q]), centered_scores(scores[1, :, q])
        for r in range(5):
            rows.append(
                [
                    float(cross_author_accounts(a, b, k)["distinct_authors"])
                    for k in kernels[:, r]
                ]
            )
    return {
        "distinct_author_source_query_channels": np.array(rows)
        .reshape(5, 5, 4)
        .tolist(),
        "not_individual_trait_predictions": True,
    }
