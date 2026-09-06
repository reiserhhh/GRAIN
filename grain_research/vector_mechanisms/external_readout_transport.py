"""Exact frozen readout transport, without fitting or psychological labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from grain_research.vector_mechanisms.external_geometry_coordinates import (
    geometry_basis,
)
from suica_core.v8_bridge import canonical_orbit_distance_signatures


@dataclass(frozen=True)
class AffineReadout:
    """Input-space coefficients, with the fitted output link left unchanged."""

    coefficient: np.ndarray
    intercept: np.ndarray
    link: str
    squeeze: bool

    def predict(self, x):
        """Evaluate finite input vectors without recalibration or refitting."""
        x = np.asarray(x, dtype=float)
        if (
            x.ndim != 2
            or x.shape[1] != len(self.coefficient)
            or not np.isfinite(x).all()
        ):
            raise ValueError("Invalid readout input")
        score = x @ self.coefficient + self.intercept
        if self.link == "logistic":
            score = expit(score)
        elif self.link != "identity":
            raise ValueError("Unsupported readout link")
        return score[:, 0] if self.squeeze else score


def collapse_dense_pipeline(model):
    """Compose a fitted scaler, optional PCA and linear head into input coordinates.

    This preserves the learned PCA subspace, unlike fitting a new PCA after
    changing coordinates. Unsupported transforms fail rather than approximate.
    """
    names = list(model.named_steps)
    if names not in (["scale", "head"], ["scale", "pca", "head"]):
        raise ValueError("Unsupported dense pipeline")
    scale = model.named_steps["scale"]
    if not isinstance(scale, StandardScaler):
        raise TypeError("Expected a fitted StandardScaler")
    dimensions = int(scale.n_features_in_)
    center = scale.mean_ if scale.with_mean else np.zeros(dimensions)
    divisor = scale.scale_ if scale.with_std else np.ones(dimensions)
    a, b = np.eye(dimensions) / divisor, -center / divisor
    if "pca" in model.named_steps:
        pca = model.named_steps["pca"]
        if not isinstance(pca, PCA):
            raise TypeError("Expected PCA")
        a, b = a @ pca.components_.T, (b - pca.mean_) @ pca.components_.T
        if pca.whiten:
            variance = pca.explained_variance_
            if np.any(variance <= np.finfo(float).eps):
                raise ValueError("Degenerate whitened PCA")
            a, b = a / np.sqrt(variance), b / np.sqrt(variance)
    head = model.named_steps["head"]
    if isinstance(head, LogisticRegression):
        if list(head.classes_) != [0, 1]:
            raise ValueError("Require binary classes [0, 1]")
        link, squeeze = "logistic", True
    elif isinstance(head, Ridge):
        link, squeeze = "identity", np.asarray(head.coef_).ndim == 1
    else:
        raise TypeError("Unsupported linear head")
    coefficient = np.atleast_2d(head.coef_).T
    result = AffineReadout(
        a @ coefficient,
        b @ coefficient + np.atleast_1d(head.intercept_),
        link,
        squeeze,
    )
    if not all(np.isfinite(v).all() for v in (result.coefficient, result.intercept)):
        raise ValueError("Nonfinite fitted readout")
    return result


def pullback_readout(readout, matrix, offset):
    """Compose a frozen readout with old_input = new_input @ matrix + offset."""
    matrix, offset = np.asarray(matrix, float), np.asarray(offset, float)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != len(readout.coefficient)
        or offset.shape != (matrix.shape[1],)
        or not np.isfinite(matrix).all()
        or not np.isfinite(offset).all()
    ):
        raise ValueError("Invalid affine coordinate map")
    return AffineReadout(
        matrix @ readout.coefficient,
        offset @ readout.coefficient + readout.intercept,
        readout.link,
        readout.squeeze,
    )


def transport_readout(readout, matrix, offset):
    """Preserve predictions when new_input = old_input @ matrix + offset.

    Solve the inverse map only for nonsingular, numerically supported matrices.
    No new empirical claim follows from an algebraic identity alone.
    """
    matrix, offset = np.asarray(matrix, float), np.asarray(offset, float)
    dimensions = len(readout.coefficient)
    if (
        matrix.shape != (dimensions, dimensions)
        or offset.shape != (dimensions,)
        or not np.isfinite(matrix).all()
        or not np.isfinite(offset).all()
        or np.linalg.cond(matrix) > 1e12
    ):
        raise ValueError("Require a finite, well-conditioned invertible map")
    inverse = np.linalg.solve(matrix, np.eye(dimensions))
    return pullback_readout(readout, inverse, -offset @ inverse)


def landmark_radial_lift(whitened, landmarks, *, relative_tolerance=1e-8):
    """Build (parallel coordinates, TOTAL squared radius) and its distance map.

    Squared landmark distances are affine in this lift, not in (parallel,
    perpendicular squared norm). The latter needs the additional ||parallel||^2
    term. Geometry comes solely from frozen landmarks, not labels or fitted heads.
    """
    q, l = np.asarray(whitened, float), np.asarray(landmarks, float)
    if (
        q.ndim != 2
        or l.ndim != 2
        or q.shape[1] != l.shape[1]
        or not np.isfinite(q).all()
    ):
        raise ValueError("Invalid query points")
    basis, _ = geometry_basis(l)
    reference, _, _ = canonical_orbit_distance_signatures(
        l, l, relative_tolerance=relative_tolerance
    )
    order = np.argmin(reference, axis=0)
    if len(set(order)) != len(l):
        raise ValueError("Ambiguous canonical landmark identities")
    centered = q - l[0]
    landmark_coordinates = (l[order] - l[0]) @ basis.T
    matrix = np.vstack([-2 * landmark_coordinates.T, np.ones(len(l))])
    offset = np.sum((l[order] - l[0]) ** 2, axis=1)
    lifted = np.column_stack([centered @ basis.T, np.sum(centered**2, axis=1)])
    return lifted, matrix, offset
