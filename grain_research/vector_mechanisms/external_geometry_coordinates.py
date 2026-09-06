"""Separate absent landmark directions from nonlinear readout coordinates."""

from __future__ import annotations

import numpy as np

from suica_core.v8_bridge import canonical_orbit_distance_signatures


def geometry_basis(landmarks):
    """Fix an orthonormal affine basis using landmarks only, never query subjects."""
    l = np.asarray(landmarks, dtype=float)
    if l.ndim != 2 or len(l) < 2 or not np.isfinite(l).all():
        raise ValueError("Invalid frozen landmarks")
    a = l[1:] - l[0]
    _, singular, vt = np.linalg.svd(a, full_matrices=True)
    tolerance = max(a.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.sum(singular > tolerance))
    return vt[:rank], vt[rank:]


def decode_identified_distances(
    canonical_distances, landmarks, *, relative_tolerance=1e-8
):
    """Recover affine projection and perpendicular squared norm from distances alone."""
    l = np.asarray(landmarks, dtype=float)
    d = np.asarray(canonical_distances, dtype=float)
    if d.ndim != 2 or d.shape[1] != len(l) or not np.isfinite(d).all() or np.any(d < 0):
        raise ValueError("Invalid identified landmark distances")
    reference_d, _, _ = canonical_orbit_distance_signatures(
        l, l, relative_tolerance=relative_tolerance
    )
    original_at_column = np.argmin(reference_d, axis=0)
    if len(set(original_at_column)) != len(l):
        raise ValueError("Landmark self-distances do not identify canonical columns")
    original = np.empty_like(d)
    original[:, original_at_column] = d
    d2 = original**2
    a = l[1:] - l[0]
    rhs = (d2[:, [0]] - d2[:, 1:] + np.sum(a * a, axis=1)) / 2
    parallel = np.linalg.lstsq(a, rhs.T, rcond=None)[0].T
    perp2 = d2[:, 0] - np.sum(parallel**2, axis=1)
    if np.min(perp2) < -1e-9 * max(1, float(np.max(d2))):
        raise ValueError("Distances do not describe the specified Euclidean landmarks")
    reconstructed = (
        np.sum((parallel[:, None, :] - (l - l[0])[None, :, :]) ** 2, axis=2)
        + perp2[:, None]
    )
    if not np.allclose(reconstructed, d2, rtol=1e-10, atol=1e-9):
        raise ValueError("Inconsistent identified distance system")
    basis, _ = geometry_basis(l)
    return parallel @ basis.T, np.maximum(perp2, 0)


def coordinate_channels(
    whitened, canonical_distances, landmarks, *, relative_tolerance=1e-8
):
    """Construct label-free channels and verify distance-decoded coordinates."""
    q, l = np.asarray(whitened, dtype=float), np.asarray(landmarks, dtype=float)
    if q.ndim != 2 or q.shape[1] != l.shape[1] or not np.isfinite(q).all():
        raise ValueError("Invalid complete whitened query points")
    basis, perpendicular = geometry_basis(l)
    centered = q - l[0]
    direct = centered @ basis.T
    residual = centered @ perpendicular.T
    decoded, perp2 = decode_identified_distances(
        canonical_distances, l, relative_tolerance=relative_tolerance
    )
    error = float(np.max(abs(decoded - direct)))
    norm_error = float(np.max(abs(perp2 - np.sum(residual**2, axis=1))))
    if error > 1e-8 or norm_error > 1e-8:
        raise ValueError("Distance decoder differs from direct projection")
    return {
        "whitened": q,
        "parallel": decoded,
        "parallel_norm": np.column_stack([decoded, perp2]),
        "perpendicular": residual,
        "squared_distances": np.asarray(canonical_distances, dtype=float) ** 2,
    }, {
        "parallel_dimensions": len(basis),
        "perpendicular_dimensions": len(perpendicular),
        "distance_decoder_max_error": error,
        "perpendicular_norm_squared_max_error": norm_error,
        "basis_uses_query_authors": False,
        "labels_used": False,
    }
