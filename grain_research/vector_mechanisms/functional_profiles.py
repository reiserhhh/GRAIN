#!/usr/bin/env python3
"""Fixed query functions, matched-query projections and author coordinates."""

from __future__ import annotations

import numpy as np


def finite_matrix(values):
    """Require a finite nonempty observation-by-coordinate matrix."""
    x = np.asarray(values, float)
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError("Finite nonempty matrix required")
    return x


class FunctionalQuery:
    """The same frozen affine function scores levels and endpoint differences."""

    def __init__(self, p_ref, q_ref, g_matrix, g_final, anchor, rank=None):
        """Optionally compress only the token-centered coefficient matrix."""
        self.p_ref = finite_matrix(p_ref).copy()
        self.q_ref, self.g_final = np.asarray(q_ref, float), np.asarray(g_final, float)
        g = finite_matrix(g_matrix)
        self.anchor = float(anchor)
        if (
            g.shape != self.p_ref.shape
            or self.q_ref.shape != (g.shape[1],)
            or self.g_final.shape != self.q_ref.shape
            or not np.isfinite(self.q_ref).all()
            or not np.isfinite(self.g_final).all()
            or not np.isfinite(self.anchor)
        ):
            raise ValueError("Reference and coefficients disagree")
        self.mean_coef = g.sum(0)
        self.config_coef = g - g.mean(0)
        maximum = min(g.shape)
        self.rank = maximum if rank is None else int(rank)
        if self.rank < 1 or self.rank > maximum:
            raise ValueError("Invalid coefficient rank")
        if self.rank < maximum:
            u, s, v = np.linalg.svd(self.config_coef, full_matrices=False)
            self.config_coef = (u[:, : self.rank] * s[: self.rank]) @ v[: self.rank]
            self.config_coef -= self.config_coef.mean(0)

    def parts(self, p, q):
        """Return exact additive contributions in unclipped native model units."""
        p, q = finite_matrix(p), np.asarray(q, float)
        if (
            p.shape != self.p_ref.shape
            or q.shape != self.q_ref.shape
            or not np.isfinite(q).all()
        ):
            raise ValueError("Input state does not match query coordinates")
        delta = p - self.p_ref
        mean = delta.mean(0)
        result = {
            "mean": float(self.mean_coef @ mean),
            "configuration": float(np.sum(self.config_coef * (delta - mean))),
            "final": float(self.g_final @ (q - self.q_ref)),
        }
        result["score"] = self.anchor + sum(result.values())
        return result


class FixedAuthorCoordinates:
    """Reference-author norms never depend on the current scoring batch."""

    def fit(self, author_values):
        """Fit on one aggregate row per declared reference author, not text rows."""
        x = finite_matrix(author_values)
        if len(x) < 3:
            raise ValueError("At least three reference authors required")
        self.center, self.scale = x.mean(0), x.std(0)
        self.active = self.scale > 1e-12
        self.scale = np.where(self.active, self.scale, 1.0)
        return self

    def transform(self, values):
        """Keep constant-reference dimensions inactive rather than amplifying them."""
        x = finite_matrix(values)
        if x.shape[1] != len(self.center):
            raise ValueError("Author profile dimension mismatch")
        return (x - self.center) / self.scale * self.active


class MatchedQueryProjection:
    """Query-specific OLS on the other queries after fixed condition centering."""

    def fit(self, values, conditions):
        """Fit exclusively on matched reference observations with equal coverage."""
        x = finite_matrix(values)
        conditions = np.asarray(conditions, str)
        if len(conditions) != len(x) or x.shape[1] < 2:
            raise ValueError("Matched reference conditions and queries required")
        self.conditions = sorted(set(conditions))
        self.centers = {c: x[conditions == c].mean(0) for c in self.conditions}
        centered = x - np.array([self.centers[c] for c in conditions])
        self.scale = centered.std(0)
        self.active = self.scale > 1e-12
        self.scale = np.where(self.active, self.scale, 1.0)
        z = centered / self.scale * self.active
        self.coef = np.zeros((x.shape[1], x.shape[1]))
        for q in range(x.shape[1]):
            other = np.arange(x.shape[1]) != q
            self.coef[other, q] = np.linalg.pinv(z[:, other], rcond=1e-10) @ z[:, q]
        self.gram = z.T @ z / len(z)
        return self

    def transform(self, values, conditions):
        """Return observed, shared and exact remainder using only frozen moments."""
        x = finite_matrix(values)
        conditions = list(conditions)
        if len(x) != len(conditions) or x.shape[1] != len(self.scale):
            raise ValueError("Projection coordinate mismatch")
        if not set(conditions) <= set(self.conditions):
            raise ValueError("Uncalibrated material condition")
        z = (
            (x - np.array([self.centers[c] for c in conditions]))
            / self.scale
            * self.active
        )
        shared = z @ self.coef
        return z, shared, z - shared


def relative_rms_error(actual, predicted):
    """Numerical action error relative to the reference action RMS."""
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    if actual.shape != predicted.shape or not np.isfinite([actual, predicted]).all():
        raise ValueError("Matched finite actions required")
    error, rms = (
        np.sqrt(np.mean((actual - predicted) ** 2)),
        np.sqrt(np.mean(actual**2)),
    )
    return (
        float(error / rms) if rms > 1e-12 else (0.0 if error <= 1e-12 else float("inf"))
    )


def profile_pair_auc(left, right):
    """Paired author-view proximity vs all off-diagonal cross-author distances."""
    left, right = finite_matrix(left), finite_matrix(right)
    if left.shape != right.shape or len(left) < 3:
        raise ValueError("Matched author-view profiles required")
    distances = np.mean((left[:, None, :] - right[None, :, :]) ** 2, axis=-1)
    positive = np.diag(distances)
    negative = distances[~np.eye(len(left), dtype=bool)]
    auc = np.mean(positive[:, None] < negative) + 0.5 * np.mean(
        positive[:, None] == negative
    )
    return float(auc), distances
