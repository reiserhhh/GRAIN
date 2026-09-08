#!/usr/bin/env python3
"""Affine query-state localization with an exact mean/configuration accounting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nonlinear_readout import Geometry, fit_heads

FAMILIES = ("p_mean", "q", "p_mean_q", "p_config", "p_full_q", "p_config_q")


def validate_states(p, q):
    """Require paired author/token/hidden states from one frozen query template."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if (
        p.ndim != 3
        or q.ndim != 2
        or p.shape[0] != q.shape[0]
        or p.shape[2] != q.shape[1]
        or min(p.shape) < 1
        or not np.isfinite(p).all()
        or not np.isfinite(q).all()
    ):
        raise ValueError("Require finite paired P[n,t,d] and Q[n,d]")
    return p, q


def state_features(p, q, family):
    """Apply a fixed linear feature map; do not fit anything on supplied authors."""
    p, q = validate_states(p, q)
    m = p.mean(1)
    c = (p - m[:, None, :]).reshape(len(p), -1)
    if family == "p_mean":
        return m
    if family == "q":
        return q
    if family == "p_mean_q":
        return np.column_stack((m, q))
    if family == "p_config":
        return c
    if family == "p_full_q":
        return np.column_stack((p.reshape(len(p), -1), q))
    if family == "p_config_q":
        return np.column_stack((c, q))
    raise ValueError("Unknown state feature family")


def unroll_affine(geometry, linear):
    """Express standardized PCA/Ridge exactly in the input feature coordinates."""
    scaled_weight = geometry.pca.components_.T @ linear.coef_
    weight = scaled_weight / geometry.scaler.scale_
    intercept = float(
        linear.intercept_
        - geometry.pca.mean_ @ scaled_weight
        - geometry.scaler.mean_ @ weight
    )
    return weight, intercept


@dataclass
class CoordinateReader:
    """Compact frozen reader with author reference and query-shape contract."""

    family: str
    weight: np.ndarray
    intercept: float
    reference_p: np.ndarray
    reference_q: np.ndarray
    alpha: float
    rank: int
    unroll_error: float

    @classmethod
    def fit(cls, p, q, teacher, family, alpha, rank=64):
        """Fit geometry and coefficients on these training authors only."""
        p, q = validate_states(p, q)
        x = state_features(p, q, family)
        geometry = Geometry.fit(x, rank)
        z = geometry.transform(x)
        linear, _ = fit_heads(z, teacher, "ridge", {"alpha": alpha})
        weight, intercept = unroll_affine(geometry, linear)
        error = float(np.max(np.abs(x @ weight + intercept - linear.predict(z))))
        if error > 1e-7:
            raise ValueError("Affine unrolling differs from fitted predictor")
        return cls(
            family,
            weight,
            intercept,
            p.mean(0),
            q.mean(0),
            float(alpha),
            geometry.pca.n_components_,
            error,
        )

    def features(self, p, q):
        """Reject a different template shape before interpreting token coefficients."""
        p, q = validate_states(p, q)
        if p.shape[1:] != self.reference_p.shape:
            raise ValueError("Query token/hidden shape differs from reference")
        return state_features(p, q, self.family)

    def predict(self, p, q):
        """Read new states without changing the training reference or coefficients."""
        return self.features(p, q) @ self.weight + self.intercept

    def full_parts(self, p, q):
        """Account for full-P/Q predictions; correlated terms are not causal shares."""
        if self.family != "p_full_q":
            raise ValueError("Exact token accounting requires full-P/Q reader")
        self.features(p, q)
        p, q = validate_states(p, q)
        d = q.shape[1]
        w_p, w_q = self.weight[:-d].reshape(p.shape[1:]), self.weight[-d:]
        dp, dq = p - self.reference_p, q - self.reference_q
        dm = dp.mean(1)
        mean = dm @ w_p.sum(0)
        token_config = np.einsum("ntd,td->nt", dp - dm[:, None, :], w_p)
        token = np.einsum("ntd,td->nt", dp, w_p)
        reference = float(
            self.predict(self.reference_p[None], self.reference_q[None])[0]
        )
        parts = {
            "reference": np.full(len(p), reference),
            "p_mean": mean,
            "p_config": token_config.sum(1),
            "q": dq @ w_q,
        }
        if np.max(np.abs(sum(parts.values()) - self.predict(p, q))) > 1e-7:
            raise ValueError("Mean/configuration/Q accounting failed")
        return parts, token, token_config
