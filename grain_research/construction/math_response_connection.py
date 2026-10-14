#!/usr/bin/env python3
"""Exact attention response fields and their declared minimum-norm connection."""

from __future__ import annotations

import numpy as np


def horizontal(jacobian, rcond=1e-12):
    """Return a full-rank Euclidean right inverse; never regularize rank loss."""
    j = np.asarray(jacobian, float)
    if j.ndim != 2 or not np.isfinite(j).all():
        raise ValueError("Finite two-dimensional Jacobian required")
    gram = j @ j.T
    eig = np.linalg.eigvalsh(gram)
    if eig[-1] <= 0 or eig[0] <= rcond * eig[-1]:
        raise ValueError("Response rank loss: no declared horizontal controller")
    inv = np.linalg.solve(gram, np.eye(len(j)))
    return j.T @ inv, inv, float(np.sqrt(eig[-1] / eig[0]))


def horizontal_derivative(jacobian, tangent_jacobian, controller, gram_inverse):
    """Differentiate J^T(JJ^T)^-1 along an explicitly supplied state tangent."""
    j, h, u, inv = jacobian, tangent_jacobian, controller, gram_inverse
    dg = h @ j.T + j @ h.T
    return h.T @ inv - u @ dg @ inv


class AttentionResponse:
    """Six (or general m) fixed attention functions in joint K/V or V coordinates."""

    def __init__(self, keys, values, queries, directions, mode="joint"):
        """Bind original numeric units, GQA grouping and a nonempty carrier."""
        self.k, self.v, self.q, self.d = (
            np.asarray(a, float) for a in (keys, values, queries, directions)
        )
        if (
            self.k.ndim != 3
            or len(self.k) == 0
            or self.v.shape != self.k.shape
            or self.q.ndim != 3
            or self.d.shape != self.q.shape
            or self.q.shape[2] != self.k.shape[2]
            or self.q.shape[1] % self.k.shape[1]
            or mode not in ("joint", "value_only")
            or not all(np.isfinite(a).all() for a in (self.k, self.v, self.q, self.d))
        ):
            raise ValueError("Compatible finite K/V/query geometry required")
        self.mode, self.size = mode, self.k.size
        self.groups = np.array_split(np.arange(self.q.shape[1]), self.k.shape[1])
        self.kh = np.repeat(np.arange(self.k.shape[1]), len(self.groups[0]))
        self.x = (
            np.concatenate([self.k.ravel(), self.v.ravel()])
            if mode == "joint"
            else self.v.ravel().copy()
        )

    def unpack(self, x):
        """Separate state or tangent coordinates without changing units."""
        x = np.asarray(x, float)
        if x.shape != self.x.shape or not np.isfinite(x).all():
            raise ValueError("Finite carrier-sized state required")
        if self.mode == "joint":
            return x[: self.size].reshape(self.k.shape), x[self.size :].reshape(
                self.k.shape
            )
        return self.k, x.reshape(self.k.shape)

    def _parts(self, x):
        """Evaluate stable softmax probabilities and per-head value readings."""
        k, v = self.unpack(x)
        logits = np.einsum("ahd,thd->ath", k[:, self.kh], self.q)
        p = np.exp(logits - logits.max(0))
        p /= p.sum(0)
        c = np.einsum("ahd,thd->ath", v[:, self.kh], self.d)
        f = (p * c).sum(0)
        return p, c, f

    def value_jacobian(self, x):
        """Compute exact normalized observations and their analytic Jacobian."""
        p, c, f = self._parts(x)
        shape = (len(self.q),) + self.k.shape
        gk, gv = np.zeros(shape), np.zeros(shape)
        for kh, hs in enumerate(self.groups):
            gv[:, :, kh] = np.einsum("ath,thd->tad", p[:, :, hs], self.d[:, hs])
            gk[:, :, kh] = np.einsum(
                "ath,thd->tad", (p * (c - f))[:, :, hs], self.q[:, hs]
            )
        jv = gv.reshape(len(self.q), -1)
        j = np.concatenate([gk.reshape(len(self.q), -1), jv], axis=1)
        return f.sum(1), j if self.mode == "joint" else jv

    def directional_jacobian(self, x, tangent):
        """Compute the Hessian action dJ[tangent] with no finite differencing."""
        p, c, f = self._parts(x)
        if self.mode == "joint":
            dk, dv = self.unpack(tangent)
        else:
            dk, dv = np.zeros_like(self.k), np.asarray(tangent).reshape(self.k.shape)
        s = np.einsum("ahd,thd->ath", dk[:, self.kh], self.q)
        dc = np.einsum("ahd,thd->ath", dv[:, self.kh], self.d)
        dp = p * (s - (p * s).sum(0))
        df = (dp * c + p * dc).sum(0)
        shape = (len(self.q),) + self.k.shape
        gk, gv = np.zeros(shape), np.zeros(shape)
        for kh, hs in enumerate(self.groups):
            gv[:, :, kh] = np.einsum("ath,thd->tad", dp[:, :, hs], self.d[:, hs])
            gk[:, :, kh] = np.einsum(
                "ath,thd->tad",
                (dp * (c - f) + p * (dc - df))[:, :, hs],
                self.q[:, hs],
            )
        jv = gv.reshape(len(self.q), -1)
        j = np.concatenate([gk.reshape(len(self.q), -1), jv], axis=1)
        return j if self.mode == "joint" else jv

    def connection(self, x, rcond=1e-12):
        """Return the selected dual-response field and its regularity diagnostics."""
        f, j = self.value_jacobian(x)
        u, inv, condition = horizontal(j, rcond)
        return f, j, u, inv, condition

    def brackets(self, x, rcond=1e-12):
        """Return all pairwise Lie brackets of horizontal response fields."""
        f, j, u, inv, condition = self.connection(x, rcond)
        du = np.stack(
            [
                horizontal_derivative(j, self.directional_jacobian(x, a), u, inv)
                for a in u.T
            ]
        )
        pairs = [(i, k) for i in range(len(f)) for k in range(i + 1, len(f))]
        c = np.stack([du[i, :, k] - du[k, :, i] for i, k in pairs])
        return f, j, u, c, pairs, condition


def rectangle(model, x, i, j, epsilon, steps=1, rcond=1e-12):
    """Lift one output-coordinate rectangle by fixed-step RK4, preserving failures."""
    if steps < 1 or not np.isfinite(epsilon) or epsilon == 0:
        raise ValueError("Nonzero finite dose and positive step count required")
    y = np.asarray(x, float).copy()
    # Negative epsilon reverses both coordinates, not loop orientation.
    # Reverse orientation instead exchanges the ordered i and j legs.
    for axis, sign in ((i, 1), (j, 1), (i, -1), (j, -1)):
        h = sign * epsilon / steps
        for _ in range(steps):
            a = model.connection(y, rcond)[2][:, axis]
            b = model.connection(y + 0.5 * h * a, rcond)[2][:, axis]
            c = model.connection(y + 0.5 * h * b, rcond)[2][:, axis]
            d = model.connection(y + h * c, rcond)[2][:, axis]
            y += h * (a + 2 * b + 2 * c + d) / 6
    return y
