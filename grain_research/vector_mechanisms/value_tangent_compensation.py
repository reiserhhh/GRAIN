#!/usr/bin/env python3
"""Exact cross-view score increments and source-observed row-space geometry."""

import numpy as np

from .compiled_query_transfer import cosine


def score_parts(phi, dphi, value, joint):
    """Split matched-normalization coefficient changes without fitting a correction."""
    for name in ("center", "scale", "origin", "target_scale", "members"):
        if not np.array_equal(value[name], joint[name]):
            raise ValueError(
                "The two objectives must share source normalization/membership"
            )
    x, d = np.asarray(phi, float), np.asarray(dphi, float)
    p = len(value["center"])
    if x.shape != (p,) or p % 2 or d.ndim != 2 or d.shape[1] != p:
        raise ValueError("Two equal views and aligned directional features required")
    if (
        not np.isfinite(x).all()
        or not np.isfinite(d).all()
        or np.any(d[:, p // 2 :] != 0)
    ):
        raise ValueError("Finite BODY-only directions required")
    z, dz = (x - value["center"]) / value["scale"], d / value["scale"]
    dc = (joint["coef"] - value["coef"]) * value["target_scale"]
    pieces = np.asarray(
        [dc[0], z[: p // 2] @ dc[1 : 1 + p // 2], z[p // 2 :] @ dc[1 + p // 2 :]]
    )
    return pieces, dz[:, : p // 2] @ dc[1 : 1 + p // 2]


def compensation_summary(parts, error):
    """Keep signed covariance and assign an exact error account, not causal shares."""
    a, e = np.asarray(parts, float), np.asarray(error, float)
    if a.ndim != 2 or a.shape[1] != 3 or e.shape != (len(a),) or len(a) < 3:
        raise ValueError("At least three aligned bias/BODY/query rows required")
    if not np.isfinite(a).all() or not np.isfinite(e).all():
        raise ValueError("Finite values required")
    total = a.sum(axis=1)
    b, q = a[:, 1], a[:, 2]
    bc, qc = b - b.mean(), q - q.mean()
    denom = b.var() + q.var()
    result = {
        "rows": len(a),
        "body_query_r": cosine(bc, qc),
        "body_query_covariance": float(np.mean(bc * qc)),
        "body_query_sum_variance": float((b + q).var()),
        "cancellation_ratio": float((b + q).var() / denom) if denom else float("nan"),
        "delta_mean": float(total.mean()),
        "delta_variance": float(total.var()),
        "delta_rms": float(np.sqrt(np.mean(total**2))),
        "mse_change": float(np.mean((e + total) ** 2 - e**2)),
        "variance_identity_error": float(
            abs((b + q).var() - denom - 2 * np.mean(bc * qc))
        ),
    }
    contributions = []
    for i, name in enumerate(("intercept", "body", "query")):
        v = a[:, i]
        contribution = float(np.mean(2 * e * v + total * v))
        contributions.append(contribution)
        result.update(
            {
                name + "_mean": float(v.mean()),
                name + "_variance": float(v.var()),
                name + "_rms": float(np.sqrt(np.mean(v**2))),
                name + "_error_covariance": float(
                    np.mean((e - e.mean()) * (v - v.mean()))
                ),
                name + "_signed_mse_contribution": contribution,
            }
        )
    result["mse_identity_error"] = abs(sum(contributions) - result["mse_change"])
    return result


def row_space(source_phi, source_dphi, model):
    """Return only the data row space, excluding the full-rank Ridge penalty."""
    x, d = np.asarray(source_phi, float), np.asarray(source_dphi, float)
    if x.ndim != 2 or d.ndim != 3 or d.shape[0] != len(x) or d.shape[2] != x.shape[1]:
        raise ValueError("Aligned source score/directional features required")
    z, dz = (x - model["center"]) / model["scale"], d / model["scale"]
    value = np.column_stack((np.ones(len(z)), z)) / np.sqrt(len(z))
    direction = np.column_stack(
        (np.zeros(dz.shape[0] * dz.shape[1]), dz.reshape(-1, dz.shape[2]))
    ) / np.sqrt(dz.shape[0] * dz.shape[1])
    matrix = np.vstack((value, direction))
    basis, singular, threshold = observed_basis(matrix)
    return basis, {
        "rank": len(basis),
        "columns": matrix.shape[1],
        "data_rows": len(matrix),
        "nullity": matrix.shape[1] - len(basis),
        "svd_threshold": threshold,
        "largest_singular": float(singular[0]),
        "smallest_retained_singular": float(singular[len(basis) - 1]),
    }


def observed_basis(matrix):
    """SVD with the frozen machine-precision threshold, no rank selection."""
    a = np.asarray(matrix, float)
    if a.ndim != 2 or min(a.shape) < 1 or not np.isfinite(a).all():
        raise ValueError("Finite nonempty data matrix required")
    _, s, vt = np.linalg.svd(a, full_matrices=False)
    threshold = float(max(a.shape) * np.finfo(float).eps * s[0])
    return vt[s > threshold], s, threshold


def coverage(rows, basis):
    """Squared projection coverage; zero rows remain explicitly undefined."""
    r, v = np.asarray(rows, float), np.asarray(basis, float)
    if (
        r.ndim != 2
        or v.ndim != 2
        or r.shape[1] != v.shape[1]
        or not np.isfinite(r).all()
    ):
        raise ValueError("Aligned finite row coordinates required")
    np.testing.assert_allclose(v @ v.T, np.eye(len(v)), atol=1e-10)
    energy = np.sum(r * r, axis=1)
    projected = np.sum((r @ v.T) ** 2, axis=1)
    result = np.full(len(r), np.nan)
    np.divide(projected, energy, out=result, where=energy > 0)
    if np.any(result < -1e-10) or np.any(result > 1 + 1e-10):
        raise ValueError("Projection is not within the declared coordinate energy")
    return result


def author_centered(values):
    """Remove each author's common action level across four directions, not a trait score."""
    v = np.asarray(values, float)
    if v.ndim != 2 or v.shape[1] != 4 or not np.isfinite(v).all():
        raise ValueError("Author-by-four finite action table required")
    return v - v.mean(axis=1, keepdims=True)
