#!/usr/bin/env python3
"""Source-fitted least squares for declared profile and contrast functionals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from grain_research.response_accounting import difference_operator
from grain_research.semantic_link import SemanticLink, author_weights

ARMS = {
    "point": ("point",),
    "level": ("point", "level"),
    "total": ("point", "total"),
    "within": ("point", "within"),
    "balanced": ("point", "level", "total", "within"),
}


class FunctionalCoordinates(SemanticLink):
    """Reuse the declared C14 feature maps without fitting a pointwise head."""

    def fit(self, g, z, subreddits, authors):
        """Fit source-only features; neither targets nor evaluation rows enter."""
        if self.mode not in {"joint_linear", "joint_quadratic"}:
            raise ValueError("Expected a declared joint coordinate map")
        if len({len(a) for a in (g, z, subreddits, authors)}) != 1:
            raise ValueError("Coordinate rows differ")
        if not np.isfinite(g).all() or not np.isfinite(z).all():
            raise ValueError("Finite coordinates required")
        w = author_weights(authors)
        subreddits, authors = np.asarray(subreddits), np.asarray(authors)
        self.categories_ = sorted(
            name
            for name in set(subreddits)
            if len(set(authors[subreddits == name])) >= self.min_subreddit_authors
        )
        self.g_scaler_ = StandardScaler().fit(g, sample_weight=w)
        self.poly_ = PolynomialFeatures(2, include_bias=False).fit(g)
        self.x_scaler_ = StandardScaler().fit(
            self._raw(g, z, subreddits), sample_weight=w
        )
        return self

    def design(self, g, z, subreddits):
        """Return fixed feature coordinates and an explicit intercept column."""
        x = self.x_scaler_.transform(self._raw(g, z, subreddits))
        return np.column_stack([np.ones(len(x)), x])


@dataclass
class Block:
    """One estimand's rows, with equal total weight per eligible author."""

    x: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    authors: np.ndarray
    energy: float


def build_blocks(x, y, conditions, halves, strict):
    """Build point/level/chronological contrasts, omitting unobserved cells.

    x and y are author x comment x coordinate/target panels. A comment may be
    entirely missing; partially missing rows are rejected. Support is intersected
    for a target-panel permutation, never imputed.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    c, h, strict = np.asarray(conditions), np.asarray(halves), np.asarray(strict)
    if x.ndim != 3 or y.ndim != 3 or x.shape[:2] != y.shape[:2]:
        raise ValueError("Aligned author/comment panels required")
    if c.shape != x.shape[:2] or h.shape != c.shape or strict.shape != (len(x),):
        raise ValueError("Panel metadata differs")
    if strict.dtype != bool:
        raise ValueError("Strict-time mask must be boolean")
    for a in (x, y):
        finite, missing = np.isfinite(a).all(2), np.isnan(a).all(2)
        if not (finite | missing).all():
            raise ValueError("Partially missing or infinite panel row")
    rows = {name: [] for name in ("point", "level", "total", "within")}
    for u in range(len(x)):
        ok = np.isfinite(x[u]).all(1) & np.isfinite(y[u]).all(1)
        if not ok.any():
            continue
        xx, yy = x[u, ok], y[u, ok]
        rows["point"].append((xx, yy, u))
        rows["level"].append((xx.mean(0, keepdims=True), yy.mean(0, keepdims=True), u))
        if strict[u] and all(np.any(ok & (h[u] == t)) for t in (0, 1)):
            a, meta = difference_operator(c[u], h[u], ok)
            for p, name in ((0, "total"), (1, "within")):
                if name == "within" and not meta["within_measured"]:
                    continue
                rows[name].append((a[p : p + 1, ok] @ xx, a[p : p + 1, ok] @ yy, u))
    blocks = {}
    for name, entries in rows.items():
        if not entries:
            raise ValueError("No supported authors for " + name)
        bx = np.concatenate([r[0] for r in entries])
        by = np.concatenate([r[1] for r in entries])
        ids = np.concatenate([np.repeat(r[2], len(r[0])) for r in entries])
        w = np.concatenate(
            [np.full(len(r[0]), 1 / (len(entries) * len(r[0]))) for r in entries]
        )
        energy = (
            1.0
            if name == "point"
            else max(0.01, float(np.average((by**2).mean(1), weights=w)))
        )
        blocks[name] = Block(bx, by, w, ids, energy)
    return blocks


def stacked_system(blocks, arm):
    """Encode the declared objective as weighted pseudo-observations."""
    if arm not in ARMS:
        raise ValueError("Unknown functional loss")
    xs, ys = [], []
    for name in ARMS[arm]:
        b = blocks[name]
        if not np.isfinite(b.energy) or b.energy <= 0:
            raise ValueError("Positive source-frozen energy required")
        w = np.sqrt(b.weights / (len(ARMS[arm]) * b.energy))[:, None]
        xs.append(b.x * w)
        ys.append(b.y * w)
    return np.concatenate(xs), np.concatenate(ys)


def fit_functional(blocks, arm, alpha):
    """Solve weighted ridge with an unpenalized intercept; return stationarity."""
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Finite positive ridge alpha required")
    x, y = stacked_system(blocks, arm)
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0
    gram, rhs = x.T @ x + penalty, x.T @ y
    beta = solve(gram, rhs, assume_a="pos")
    relative = float(np.max(abs(gram @ beta - rhs)) / max(1.0, np.max(abs(rhs))))
    if relative > 1e-9 or not np.isfinite(beta).all():
        raise ArithmeticError("Functional solution did not satisfy normal equations")
    return beta, {
        "stationarity_relative_max": relative,
        "pseudo_rows": len(x),
        "features": x.shape[1],
    }


def block_loss(block, beta, energy=None):
    """Evaluate one block, optionally in frozen fit-set measurement units."""
    error = block.x @ beta - block.y
    return float(
        np.average((error**2).mean(1), weights=block.weights)
        / (block.energy if energy is None else energy)
    )


def author_panel_permutation(y, seed=915):
    """Shuffle whole source-author panels, preserving each donor's missingness."""
    y = np.asarray(y, float)
    if y.ndim != 3:
        raise ValueError("Author panel required")
    order = np.random.default_rng(seed).permutation(len(y))
    return y[order].copy(), order


def operator_profile(scores, operators, common, strict):
    """Apply frozen accounting to each reader, with absent responses remaining NaN."""
    scores, operators = np.asarray(scores, float), np.asarray(operators, float)
    if scores.ndim != 3 or operators.shape != (len(scores), 4, scores.shape[1]):
        raise ValueError("Score/operator panel mismatch")
    out = np.full((len(scores), 5, scores.shape[-1]), np.nan)
    for u in range(len(scores)):
        valid = np.isfinite(scores[u]).all(1)
        if valid.any():
            out[u, 0] = scores[u, valid].mean(0)
        if strict[u] and np.isfinite(operators[u]).all():
            if np.any(operators[u, :, ~valid] != 0):
                raise ValueError("Operator uses missing score")
            out[u, 1:] = operators[u][:, valid] @ scores[u, valid]
            if not common[u]:
                out[u, 2:4] = np.nan
    return out
