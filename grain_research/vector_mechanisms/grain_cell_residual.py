#!/usr/bin/env python3
"""Nested GRAIN polynomial/cell projections of a fixed native query field.

The saturated projector is empirical conditional expectation given exact GRAIN
rows. It is not a trained predictor, a smoothness assumption, or a trait score.
"""

import itertools

import numpy as np
import pandas as pd

PARTS = ("linear", "quadratic_increment", "between_cell_remainder", "within_cell")
VIEWS = ("neuroticism", "confirmation_seeking", "query_mean", "query_difference")
TABLES = (
    "cell_geometry",
    "gradient_energy",
    "responses",
    "cross_terms",
    "query_geometry",
)
COUNTS = [2, 32, 160, 240, 8]


def ratio(a, b):
    """Keep undefined ratios missing; signed response projections are not clipped."""
    return float(a / b) if b > 0 else np.nan


def thin_basis(a):
    """Unit-normalize nonzero columns before the frozen machine-precision rank rule."""
    norms = np.linalg.norm(a, axis=0)
    normalized = a[:, norms > 0] / norms[norms > 0]
    u, singular, _ = np.linalg.svd(normalized, full_matrices=False)
    cutoff = max(normalized.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.sum(singular > cutoff))
    return u[:, :rank], float(singular[0] / singular[rank - 1])


class CellMaps:
    """Build linear, complete degree-two, and exact-cell spaces without target data."""

    def __init__(self, grain):
        g = np.asarray(grain, float)
        if g.ndim != 2 or g.shape[1] != 4 or len(g) < 2 or not np.isfinite(g).all():
            raise ValueError("Finite token-by-four frozen GRAIN matrix required")
        centered = g - g.mean(0)
        self.a1 = np.column_stack((np.ones(len(g)), centered))
        quadratic = [
            centered[:, i] * centered[:, j] for i in range(4) for j in range(i, 4)
        ]
        self.a2 = np.column_stack((self.a1, *quadratic))
        self.q1, condition1 = thin_basis(self.a1)
        self.q2, condition2 = thin_basis(self.a2)
        _, self.index, self.counts = np.unique(
            g, axis=0, return_inverse=True, return_counts=True
        )
        self.order = np.argsort(self.index, kind="stable")
        self.starts = np.r_[0, np.cumsum(self.counts)[:-1]]
        self.geometry = {
            "tokens": len(g),
            "linear_rank": self.q1.shape[1],
            "quadratic_rank": self.q2.shape[1],
            "cells": len(self.counts),
            "within_cell_df": len(g) - len(self.counts),
            "singleton_token_fraction": float(np.sum(self.counts == 1) / len(g)),
            "cell_size_min": int(self.counts.min()),
            "cell_size_median": float(np.median(self.counts)),
            "cell_size_max": int(self.counts.max()),
            "linear_condition": condition1,
            "quadratic_condition": condition2,
        }
        self.checks = {
            "linear_quadratic_nesting": float(
                np.linalg.norm(self.q1 - self.q2 @ (self.q2.T @ self.q1))
            ),
            "quadratic_cell_nesting": float(
                np.linalg.norm(self.q2 - self.cell_project(self.q2))
            ),
            "q1_orthogonality": float(
                np.linalg.norm(self.q1.T @ self.q1 - np.eye(self.q1.shape[1]))
            ),
            "q2_orthogonality": float(
                np.linalg.norm(self.q2.T @ self.q2 - np.eye(self.q2.shape[1]))
            ),
        }
        if max(self.checks.values()) > 1e-8:
            raise ValueError("Numerical nesting failed; do not silently re-bin GRAIN")

    def cell_means(self, x):
        """Group token rows with exact equality, using the population divisor n_cell."""
        x = np.asarray(x, float)
        if x.ndim != 2 or len(x) != len(self.index) or not np.isfinite(x).all():
            raise ValueError("Aligned finite token matrix required")
        return (
            np.add.reduceat(x[self.order], self.starts, axis=0) / self.counts[:, None]
        )

    def cell_project(self, x):
        """Lift exact within-cell means back to the original token order."""
        return self.cell_means(x)[self.index]

    def decompose(self, x):
        """Four mutually orthogonal vector components when the spaces are nested."""
        p1 = self.q1 @ (self.q1.T @ x)
        p2 = self.q2 @ (self.q2.T @ x)
        pc = self.cell_project(x)
        return (p1, p2 - p1, pc - p2, x - pc)

    def covariance_action(self, gradient, direction):
        """Compute sum n_cell trace Cov_cell(J,D) via centered within-cell products."""
        j = gradient - self.cell_project(gradient)
        d = direction - self.cell_project(direction)
        covariance = self.cell_means(j * d)
        return float(np.sum(self.counts[:, None] * covariance))


def analyze(grain, directions, gradients):
    """Account for two native queries and all four original perturbation directions."""
    maps = CellMaps(grain)
    ds, jj = np.asarray(directions, float), np.asarray(gradients, float)
    if (
        jj.ndim != 3
        or jj.shape[:2] != (2, len(grain))
        or ds.shape != (4, *jj.shape[1:])
    ):
        raise ValueError("Two paired gradients and four aligned directions required")
    if not np.isfinite(ds).all() or not np.isfinite(jj).all():
        raise ValueError("Nonfinite gradient or direction")
    checks = dict(maps.checks)
    rows = {name: [] for name in TABLES}
    rows["cell_geometry"].append(maps.geometry)
    fields = (jj[0], jj[1], jj.mean(0), jj[0] - jj[1])
    base_parts = []
    for qi, (query, field) in enumerate(zip(VIEWS, fields, strict=True)):
        pieces = maps.decompose(field)
        if qi < 2:
            base_parts.append(pieces)
        total = float(np.sum(field**2))
        scale = max(total, np.finfo(float).tiny)
        checks["field_sum"] = max(
            checks.get("field_sum", 0),
            float(np.linalg.norm(sum(pieces) - field) / max(np.linalg.norm(field), 1)),
        )
        orth = max(
            abs(float(np.sum(pieces[i] * pieces[j])))
            for i, j in itertools.combinations(range(4), 2)
        )
        checks["orthogonal_components"] = max(
            checks.get("orthogonal_components", 0), orth / scale
        )
        energies = [float(np.sum(x**2)) for x in pieces]
        checks["energy_sum"] = max(
            checks.get("energy_sum", 0), abs(sum(energies) - total) / scale
        )
        native = np.einsum("th,fth->f", field, ds)
        actions = np.stack([np.einsum("th,fth->f", x, ds) for x in pieces])
        checks["action_sum"] = max(
            checks.get("action_sum", 0),
            float(np.max(abs(actions.sum(0) - native)) / max(1, np.max(abs(native)))),
        )
        for part, energy in zip(PARTS, energies, strict=True):
            rows["gradient_energy"].append(
                {"query": query, "part": part, "energy": energy, "native_energy": total}
            )
        for fi, direction in enumerate(ds):
            factor = f"F{fi + 1:02d}"
            cov = maps.covariance_action(field, direction)
            checks["conditional_covariance_identity"] = max(
                checks.get("conditional_covariance_identity", 0),
                abs(cov - actions[3, fi]) / max(1, abs(native[fi])),
            )
            for pi, part in enumerate(PARTS):
                rows["responses"].append(
                    {
                        "query": query,
                        "factor": factor,
                        "part": part,
                        "native": float(native[fi]),
                        "action": float(actions[pi, fi]),
                        "covariance_action": cov if pi == 3 else np.nan,
                    }
                )
            for pi, pj in itertools.combinations(range(4), 2):
                rows["cross_terms"].append(
                    {
                        "query": query,
                        "factor": factor,
                        "left": PARTS[pi],
                        "right": PARTS[pj],
                        "native_squared": float(native[fi] ** 2),
                        "cross": float(2 * actions[pi, fi] * actions[pj, fi]),
                    }
                )
    for pi, part in enumerate(PARTS):
        n, c = base_parts[0][pi], base_parts[1][pi]
        ne, ce, nc = float(np.sum(n * n)), float(np.sum(c * c)), float(np.sum(n * c))
        rows["query_geometry"].append(
            {
                "part": part,
                "pair_energy": ne + ce,
                "common_energy": (ne + ce + 2 * nc) / 2,
                "difference_energy": (ne + ce - 2 * nc) / 2,
                "cosine": ratio(nc, np.sqrt(ne * ce)),
            }
        )
    if max(checks.values()) > 1e-8:
        raise ValueError("Nested decomposition identity failed")
    return rows, checks


def aggregate(raw):
    """Keep cohorts separate and aggregate response projections, not raw token r."""
    result = {}
    keys = {
        "cell_geometry": [],
        "gradient_energy": ["query", "part"],
        "responses": ["query", "factor", "part"],
        "cross_terms": ["query", "factor", "left", "right"],
        "query_geometry": ["part"],
    }
    for table in TABLES:
        data = raw[table]
        if table in ("responses", "cross_terms"):
            data = pd.concat([data, data.assign(factor="all4")], ignore_index=True)
        records = []
        for key, part in data.groupby(["cohort", *keys[table]], sort=True):
            if not isinstance(key, tuple):
                key = (key,)
            row = dict(zip(["cohort", *keys[table]], key, strict=True))
            row["authors"] = part.author_key.nunique()
            if table == "cell_geometry":
                for column in data.columns.difference(["cohort", "author_key"]):
                    for stat in ("min", "median", "max"):
                        row[f"{column}_{stat}"] = float(getattr(part[column], stat)())
            elif table == "gradient_energy":
                row.update(
                    energy=float(part.energy.sum()),
                    native_energy=float(part.native_energy.sum()),
                )
                row["energy_fraction"] = ratio(row["energy"], row["native_energy"])
            elif table == "responses":
                n, a = part.native.to_numpy(float), part.action.to_numpy(float)
                row.update(
                    observations=len(n),
                    native_squared=float(n @ n),
                    action_squared=float(a @ a),
                    signed_projection=ratio(float(a @ n), float(n @ n)),
                    squared_ratio=ratio(float(a @ a), float(n @ n)),
                )
            elif table == "cross_terms":
                row["signed_cross_ratio"] = ratio(
                    float(part.cross.sum()), float(part.native_squared.sum())
                )
            else:
                for column in ("pair_energy", "common_energy", "difference_energy"):
                    row[column] = float(part[column].sum())
                row["common_fraction"] = ratio(row["common_energy"], row["pair_energy"])
                row["mean_author_cosine"] = float(part.cosine.mean())
            records.append(row)
        result[table] = pd.DataFrame(records)
    return result
