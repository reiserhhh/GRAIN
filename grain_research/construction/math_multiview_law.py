#!/usr/bin/env python3
"""Anonymous categorical observation-law identification, not trait naming."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


class IdentificationError(ValueError):
    """The fixed spectral constructor cannot identify a probability law."""


def joint_law(weights, emissions):
    """Construct the three-view law from column-stochastic emissions."""
    return np.einsum("h,ih,jh,kh->ijk", weights, *emissions)


def empirical_tensor(categories, sizes, event_views):
    """Count anonymous triples; distinct event names do not prove independence."""
    x = np.asarray(categories)
    if len(event_views) != 3 or len(set(event_views)) != 3:
        raise ValueError("Three distinct declared event views required")
    if x.ndim != 2 or x.shape[1] != 3 or not len(x):
        raise ValueError("Nonempty author-by-three categorical matrix required")
    if not np.issubdtype(x.dtype, np.integer):
        raise ValueError("Categories must be integer addresses, not scores or IDs")
    if any(np.any((x[:, v] < 0) | (x[:, v] >= n)) for v, n in enumerate(sizes)):
        raise ValueError("Category outside the frozen observation dictionary")
    counts = np.zeros(sizes, dtype=np.float64)
    np.add.at(counts, tuple(x.T), 1)
    return counts / len(x)


def _probabilities(a, axis):
    """Normalize only numerical roundoff; substantive negatives are failures."""
    a = np.asarray(a, dtype=float)
    if not np.all(np.isfinite(a)) or np.min(a) < -1e-10:
        raise IdentificationError("Spectral estimate is not a probability law")
    a = np.maximum(a, 0)
    s = a.sum(axis=axis, keepdims=True)
    if np.any(s <= 1e-12):
        raise IdentificationError("Zero probability normalizer")
    return a / s


def fit_law(tensor, k, eta, tolerance=1e-10):
    """Recover an anonymous law using a fixed observable-operator contraction.

    Inputs contain only observed frequencies, fixed component count and probe.
    No latent labels, teacher outputs, psychological labels or oracle emissions
    enter. Categorical sum-to-one fixes eigenvector scaling. Source eigenvalue
    order fixes anonymous component order. No contraction search or EM rescue.
    """
    t = np.asarray(tensor, dtype=float)
    eta = np.asarray(eta, dtype=float)
    if t.ndim != 3 or min(t.shape) < k or eta.shape != (t.shape[2],):
        raise ValueError("Incompatible tensor/component/probe dimensions")
    if np.min(t) < 0 or not np.isclose(t.sum(), 1, atol=1e-12):
        raise ValueError("Observed tensor must be a normalized probability law")
    p12 = t.sum(axis=2)
    u, s, vt = np.linalg.svd(p12, full_matrices=False)
    if s[k - 1] <= tolerance * s[0]:
        raise IdentificationError("Pair moment lacks the declared component rank")
    u, v = u[:, :k], vt[:k].T
    base = u.T @ p12 @ v

    def operator(z):
        contracted = np.einsum("ijk,k->ij", t, z)
        return np.linalg.solve(base.T, (u.T @ contracted @ v).T).T

    eigenvalues, r = np.linalg.eig(operator(eta))
    if max(np.max(abs(eigenvalues.imag)), np.max(abs(r.imag))) > 1e-9:
        raise IdentificationError("Fixed contraction has non-real eigenstructure")
    eigenvalues, r = eigenvalues.real, r.real
    order = np.argsort(eigenvalues)
    eigenvalues, r = eigenvalues[order], r[:, order]
    if np.min(np.diff(eigenvalues)) <= tolerance:
        raise IdentificationError("Fixed contraction does not separate components")
    m1 = u @ r
    sums = m1.sum(axis=0)
    if np.min(abs(sums)) <= tolerance:
        raise IdentificationError("Categorical eigenvector normalization is singular")
    m1 = _probabilities(m1 / sums, axis=0)
    weights = _probabilities(np.linalg.lstsq(m1, p12.sum(axis=1), rcond=None)[0], 0)
    m2 = _probabilities((np.linalg.pinv(m1) @ p12 / weights[:, None]).T, 0)
    # All coordinate contractions share the fixed eigenbasis; off-diagonal
    # residuals remain reported rather than being tuned away.
    matrices = [np.linalg.solve(r, operator(e)) @ r for e in np.eye(t.shape[2])]
    m3 = _probabilities(np.stack([np.diag(a) for a in matrices]), 0)
    off_diagonal = max(np.max(abs(a - np.diag(np.diag(a)))) for a in matrices)
    emissions = (m1, m2, m3)
    reconstructed = joint_law(weights, emissions)
    return {
        "weights": weights,
        "emissions": emissions,
        "eigenvalues": eigenvalues,
        "pair_singular_values": s,
        "off_diagonal_max": float(off_diagonal),
        "reconstruction_TV": float(abs(reconstructed - t).sum() / 2),
    }


def posterior(categories, weights, emissions):
    """Cold anonymous-class probabilities from distinct categorical evidence."""
    x = np.asarray(categories)
    if x.ndim != 2 or x.shape[1] != len(emissions):
        raise ValueError("A category per declared view is required")
    log_mass = np.broadcast_to(np.log(weights), (len(x), len(weights))).copy()
    for v, m in enumerate(emissions):
        if np.any((x[:, v] < 0) | (x[:, v] >= len(m))):
            raise ValueError("Unknown category")
        with np.errstate(divide="ignore"):
            log_mass += np.log(m[x[:, v]])
    denominator = logsumexp(log_mass, axis=1, keepdims=True)
    if not np.all(np.isfinite(denominator)):
        raise ValueError("Observed combination has zero modeled support")
    return np.exp(log_mass - denominator)


def qualified_posterior(records, weights, emissions):
    """Compose one author's declared events without multiplying repeated mentions.

    Event identity and self qualification are supplied, not inferred from text.
    Different events can still share hidden occasion causes. That assumption is
    outside this ledger operation and must not be manufactured by deduplication.
    """
    seen, selected, repeated = {}, [], 0
    for row in records:
        if set(row) != {"event", "view", "category", "qualification"}:
            raise ValueError("Unrecognized or potentially leaking record fields")
        if row["qualification"] != "self_actual":
            raise ValueError("Unresolved event qualification")
        event, view, category = row["event"], row["view"], row["category"]
        if not event or not isinstance(view, int) or not 0 <= view < len(emissions):
            raise ValueError("Invalid event/view")
        if not isinstance(category, int) or not 0 <= category < len(emissions[view]):
            raise ValueError("Invalid observation category")
        if event in seen:
            if seen[event] != (view, category):
                raise ValueError(
                    "Same event re-encoded or corrected without a history op"
                )
            repeated += 1
            continue
        seen[event] = (view, category)
        selected.append((view, category))
    log_mass = np.log(weights).copy()
    for view, category in selected:
        with np.errstate(divide="ignore"):
            log_mass += np.log(emissions[view][category])
    denominator = logsumexp(log_mass)
    if not np.isfinite(denominator):
        raise ValueError("Selected evidence has zero modeled support")
    return {
        "posterior": np.exp(log_mass - denominator),
        "unique_events": len(selected),
        "repeated_mentions": repeated,
    }


def pairwise_alias(weights, emissions, angle):
    """Rotate weighted component factors preserving every marginal and pair law.

    Positive interior factors at a sufficiently small angle give a second valid
    mixture. The third-order law generally changes; no fitting is involved.
    """
    q = np.sqrt(weights)
    axis = np.ones(3) / np.sqrt(3)
    cross = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    rotation = np.eye(3) * np.cos(angle) + np.outer(axis, axis) * (1 - np.cos(angle))
    rotation += cross * np.sin(angle)
    new_q = rotation.T @ q
    if np.min(new_q) <= 0:
        raise ValueError("Rotation leaves positive mixture interior")
    transformed = tuple((m * q) @ rotation / new_q for m in emissions)
    if any(np.min(m) <= 0 for m in transformed):
        raise ValueError("Rotation leaves positive observation-law interior")
    return new_q**2, transformed, rotation


def decode_vectors(vectors, alphabet):
    """Decode a finite frozen vector alphabet, not unrestricted natural language."""
    vectors, alphabet = np.asarray(vectors), np.asarray(alphabet)
    if vectors.ndim != 3 or alphabet.shape[:2] != (3, 3):
        raise ValueError("Three views/three source vectors per view required")
    x = np.empty(vectors.shape[:2], dtype=np.int8)
    for v in range(3):
        distance = ((vectors[:, v, None, :] - alphabet[v]) ** 2).sum(axis=-1)
        x[:, v] = np.argmin(distance, axis=1)
    return x
