#!/usr/bin/env python3
"""Text-derived secant charts and jointly qualified finite evidence contrasts."""

from __future__ import annotations

import numpy as np

from grain_research.construction.math_response_connection import (
    horizontal,
    horizontal_derivative,
)


def support_frame(columns, rtol=1e-10):
    """Retain the whole empirical span, without target-directed column selection."""
    a = np.asarray(columns, float)
    if a.ndim != 2 or not a.size or not np.isfinite(a).all() or rtol <= 0:
        raise ValueError("Finite nonempty secant columns and positive rtol required")
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    rank = int(np.count_nonzero(s > rtol * s[0])) if s[0] else 0
    return u[:, :rank], s


def mixed_contrast(high_a, low_a, high_b, low_b):
    """Return a signed finite content-by-owner contrast, not a derivative claim."""
    values = [np.asarray(a, float) for a in (high_a, low_a, high_b, low_b)]
    if any(a.shape != values[0].shape or not np.isfinite(a).all() for a in values):
        raise ValueError("Aligned finite corner arrays required")
    return values[0] - values[1] - values[2] + values[3]


def innovation(vector, basis):
    """Split a finite interaction into marginal-span projection and new direction."""
    w, b = np.asarray(vector, float), np.asarray(basis, float)
    if w.ndim != 1 or b.ndim != 2 or b.shape[0] != len(w):
        raise ValueError("Aligned vector and orthonormal frame required")
    if not np.allclose(b.T @ b, np.eye(b.shape[1]), atol=1e-10, rtol=0):
        raise ValueError("Frame must preserve the original Euclidean metric")
    projected = b @ (b.T @ w)
    residual = w - projected
    norm = float(np.linalg.norm(w))
    return projected, residual, float(np.linalg.norm(residual) / norm) if norm else 0.0


class TextSpanResponse:
    """Pull back a fixed observer to an empirical affine secant chart.

    The numerical interior is not asserted to encode any natural text. An
    orthonormal frame preserves the ambient metric without hidden whitening.
    """

    def __init__(self, ambient, basis):
        """Bind the unchanged observer and an outcome-independent support frame."""
        self.ambient, self.basis = ambient, np.asarray(basis, float)
        if (
            self.basis.ndim != 2
            or self.basis.shape[0] != len(ambient.x)
            or not self.basis.shape[1]
            or not np.isfinite(self.basis).all()
            or not np.allclose(
                self.basis.T @ self.basis,
                np.eye(self.basis.shape[1]),
                rtol=0,
                atol=1e-10,
            )
        ):
            raise ValueError("Nonempty orthonormal ambient secant frame required")
        self.x = np.zeros(self.basis.shape[1])

    def lift(self, z):
        """Map numerical chart coordinates to ambient K/V, without decoding text."""
        z = np.asarray(z, float)
        if z.shape != self.x.shape or not np.isfinite(z).all():
            raise ValueError("Finite chart-sized coordinates required")
        return self.ambient.x + self.basis @ z

    def value_jacobian(self, z):
        """Apply the chain rule to the actual frozen attention observer."""
        f, j = self.ambient.value_jacobian(self.lift(z))
        return f, j @ self.basis

    def directional_jacobian(self, z, tangent):
        """Pull back its analytic Hessian action along a chart direction."""
        a = np.asarray(tangent, float)
        if a.shape != self.x.shape or not np.isfinite(a).all():
            raise ValueError("Finite chart-sized tangent required")
        return (
            self.ambient.directional_jacobian(self.lift(z), self.basis @ a) @ self.basis
        )

    def connection(self, z, rcond=1e-12):
        """Return the declared minimum-norm fields, preserving rank failures."""
        f, j = self.value_jacobian(z)
        u, inv, condition = horizontal(j, rcond)
        return f, j, u, inv, condition

    def brackets(self, z, rcond=1e-12):
        """Calculate all response commutators using the same pulled-back function."""
        f, j, u, inv, condition = self.connection(z, rcond)
        du = np.stack(
            [
                horizontal_derivative(j, self.directional_jacobian(z, a), u, inv)
                for a in u.T
            ]
        )
        pairs = [(i, k) for i in range(len(f)) for k in range(i + 1, len(f))]
        return (
            f,
            j,
            u,
            np.stack([du[i, :, k] - du[k, :, i] for i, k in pairs]),
            pairs,
            condition,
        )
