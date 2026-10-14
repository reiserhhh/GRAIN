"""Source-supported symmetric observation increments, without new target fitting."""

from __future__ import annotations

import numpy as np


def gram_blocks(x, context, kappa):
    """Return U=sym(xcT), V=kappa*xxT/2, W=U+V observation Grams."""
    x, c = np.asarray(x, dtype=float), np.asarray(context, dtype=float)
    if x.ndim != 2 or x.shape != c.shape or not np.isfinite(kappa):
        raise ValueError("paired coordinates and finite step ratio required")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(c)):
        raise ValueError("finite coordinates required")
    xx, cc, xc = x @ x.T, c @ c.T, x @ c.T
    uu = (xx * cc + xc * xc.T) / 2
    vv = kappa**2 * xx**2 / 4
    uv = kappa * xx * xc.T / 2
    ww = uu + uv + uv.T + vv
    return {"U": uu, "V": vv, "UV": uv, "W": ww}


def spectrum(gram, multiplier=100.0):
    """Report signed roundoff and fixed numerical rank; never clip eigenvalues."""
    g = np.asarray(gram, dtype=float)
    if g.ndim != 2 or g.shape[0] != g.shape[1] or not np.all(np.isfinite(g)):
        raise ValueError("finite square Gram required")
    if len(g) == 0 or multiplier <= 0:
        raise ValueError("nonempty design and positive tolerance multiplier required")
    eigen, vectors = np.linalg.eigh((g + g.T) / 2)
    tol = multiplier * np.finfo(float).eps * max(1.0, eigen[-1]) * len(g)
    keep = eigen > tol
    return {
        "eigenvalues": eigen,
        "vectors": vectors,
        "keep": keep,
        "rank": int(sum(keep)),
        "tolerance": float(tol),
        "eigen_min": float(eigen[0]),
        "largest": float(eigen[-1]),
        "negative_beyond_tolerance": int(sum(eigen < -tol)),
    }


def source_projection(uu, uv, vv, weights, multiplier=100.0):
    """Build a geometric row-span projection using only the source U Gram."""
    uu, uv, vv = (np.asarray(a, dtype=float) for a in (uu, uv, vv))
    w = np.asarray(weights, dtype=float)
    if uu.shape != uv.shape or uu.shape != vv.shape or uu.shape != (len(w), len(w)):
        raise ValueError("paired source Gram blocks required")
    if np.any(w <= 0) or not np.isclose(sum(w), 1.0, atol=1e-12, rtol=0):
        raise ValueError("positive source probability weights required")
    d = np.sqrt(w)
    eig = spectrum(d[:, None] * uu * d[None, :], multiplier)
    e, val = eig["vectors"][:, eig["keep"]], eig["eigenvalues"][eig["keep"]]
    inverse = (e / val) @ e.T
    coefficient = d[:, None] * inverse * d[None, :]
    residual = vv - uv.T @ coefficient @ uv
    return {
        "coefficient": coefficient,
        "residual": (residual + residual.T) / 2,
        "u_rank": eig["rank"],
        "u_tolerance": eig["tolerance"],
        "range_defect_max": float(np.max(abs(uu @ coefficient @ uv - uv))),
    }


def projected_action(x, source_x, source_context, kappa, coefficient, source_u_action):
    """Evaluate <S,P_U V> from source U actions, with no response regression."""
    xu = np.asarray(x) @ np.asarray(source_x).T
    xc = np.asarray(x) @ np.asarray(source_context).T
    cross = kappa * xu * xc / 2
    return cross @ np.asarray(coefficient) @ np.asarray(source_u_action)


def explicit_observations(x, context, kappa):
    """Materialize small-coordinate dyads for tests, not full ambient d^2 arrays."""
    x, c = np.asarray(x), np.asarray(context)
    dyad = np.einsum("ij,ik->ijk", x, c)
    u = (dyad + dyad.transpose(0, 2, 1)) / 2
    v = kappa * np.einsum("ij,ik->ijk", x, x) / 2
    return u.reshape(len(x), -1), v.reshape(len(x), -1)
