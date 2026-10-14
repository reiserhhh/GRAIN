"""HM17: exact source-dyad decomposition, without new fits or ambient matrices."""

from __future__ import annotations

import numpy as np

MODELS = (
    "full_linear",
    "receiver_tensor",
    "receiver_without_skew",
    "receiver_potential",
    "endpoint_potential",
    "shuffled_context",
    "shuffled_without_skew",
    "shuffled_potential",
)


def matrices(*values):
    """Require finite matrices in a common Euclidean coordinate space."""
    result = tuple(np.asarray(v, float) for v in values)
    if any(v.ndim != 2 or not np.isfinite(v).all() for v in result):
        raise ValueError("finite matrices required")
    if len({v.shape[1] for v in result}) != 1:
        raise ValueError("common coordinate dimension required")
    return result


def action_parts(x, context, source_x, source_c, alpha, kappa):
    """Compute b*x, x*S*c, x*H*c and kappa*x*S*x/2 from implicit dyads."""
    x, c, a, b = matrices(x, context, source_x, source_c)
    alpha = np.asarray(alpha, float)
    if (
        len(x) != len(c)
        or len(a) != len(b)
        or alpha.shape != (len(a),)
        or not np.isfinite(alpha).all()
        or not np.isfinite(kappa)
        or kappa <= 0
    ):
        raise ValueError("matched finite dyads and positive kappa required")
    xa, xb, ca, cb = x @ a.T, x @ b.T, c @ a.T, c @ b.T
    direct, transpose = (xa * cb) @ alpha, (xb * ca) @ alpha
    linear = xa @ alpha
    symmetric, skew = (direct + transpose) / 2, (direct - transpose) / 2
    step = kappa * ((xa * xb) @ alpha) / 2
    return {
        "linear": linear,
        "symmetric": symmetric,
        "skew": skew,
        "finite_step": step,
        "receiver": linear + symmetric + skew,
        "potential": linear + symmetric + step,
        "receiver_minus_potential": skew - step,
    }


def potential_value(z, source_x, source_c, alpha, kappa):
    """Evaluate the scalar potential of the symmetric part, not its left sum."""
    z, a, b = matrices(z, source_x, source_c)
    alpha = np.asarray(alpha, float)
    if len(a) != len(b) or alpha.shape != (len(a),) or kappa <= 0:
        raise ValueError("matched dyads and positive kappa required")
    za, zb = z @ a.T, z @ b.T
    return ((za + za * zb / 2) @ alpha) / kappa


def gram_products(source_x, source_c, coefficients):
    """Return b/A/S/H inner-product matrices across frozen query functions."""
    x, c = matrices(source_x, source_c)
    a = np.asarray(coefficients, float)
    if (
        len(x) != len(c)
        or a.ndim != 2
        or a.shape[1] != len(x)
        or not np.isfinite(a).all()
    ):
        raise ValueError("matched source rows and coefficient matrix required")
    gx, gc, cross = x @ x.T, c @ c.T, c @ x.T
    linear = a @ gx @ a.T
    operator = a @ (gx * gc) @ a.T
    transpose = a @ (cross * cross.T) @ a.T
    return {
        "b": linear,
        "A": operator,
        "S": (operator + transpose) / 2,
        "H": (operator - transpose) / 2,
    }


def source_basis(source_x, source_c, multiplier=100.0):
    """Compress common source support by a fixed machine-tolerance Gram spectrum."""
    x, c = matrices(source_x, source_c)
    u = np.vstack((x, c))
    if not len(u) or multiplier <= 0 or not np.isfinite(multiplier):
        raise ValueError("nonempty source rows and finite positive tolerance required")
    gram = u @ u.T
    eigen, vectors = np.linalg.eigh((gram + gram.T) / 2)
    tolerance = multiplier * np.finfo(float).eps * max(1.0, eigen[-1]) * len(u)
    if eigen[0] < -tolerance:
        raise ValueError("common source Gram not PSD")
    keep = eigen > tolerance
    basis = (u.T @ vectors[:, keep]) / np.sqrt(eigen[keep])
    coordinates = u @ basis
    residual = u - coordinates @ basis.T
    return {
        "basis": basis,
        "source_coordinates": coordinates,
        "eigenvalues": eigen,
        "rank": int(np.sum(keep)),
        "tolerance": float(tolerance),
        "discarded_positive_eigen_sum": float(np.sum(np.maximum(eigen[~keep], 0))),
        "source_reconstruction_max": float(np.max(np.abs(residual))),
        "basis_orthogonality_max": float(
            np.max(np.abs(basis.T @ basis - np.eye(np.sum(keep))), initial=0)
        ),
    }


def compact_operator(source_x, source_c, alpha, basis, symmetric=False):
    """Construct the supported operator in common coordinates, not a fitted rank."""
    x, c = matrices(source_x, source_c)
    q, alpha = np.asarray(basis, float), np.asarray(alpha, float)
    if (
        q.ndim != 2
        or q.shape[0] != x.shape[1]
        or alpha.shape != (len(x),)
        or len(x) != len(c)
    ):
        raise ValueError("compatible source basis and dyads required")
    px, pc = x @ q, c @ q
    a = px.T @ (alpha[:, None] * pc)
    if symmetric:
        a = (a + a.T) / 2
    return a, alpha @ px


def spectra(operator, multiplier=100.0):
    """Keep all operator spectra and signed symmetric eigenvalues in source support."""
    a = np.asarray(operator, float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise ValueError("finite square supported operator required")
    s, h = (a + a.T) / 2, (a - a.T) / 2
    values = {
        k: np.linalg.svd(v, compute_uv=False) for k, v in (("A", a), ("S", s), ("H", h))
    }
    values["S_signed"] = np.linalg.eigvalsh(s)
    return {
        "values": values,
        "ranks": {
            k: int(
                np.sum(
                    v
                    > multiplier
                    * np.finfo(float).eps
                    * max(1.0, np.max(v, initial=0))
                    * max(1, len(a))
                )
            )
            for k, v in values.items()
            if k != "S_signed"
        },
        "frobenius_squared": {
            "A": float(np.sum(a * a)),
            "S": float(np.sum(s * s)),
            "H": float(np.sum(h * h)),
        },
        "s_h_inner_product": float(np.sum(s * h)),
    }
