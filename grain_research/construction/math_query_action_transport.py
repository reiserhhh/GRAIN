"""HM22: source-fixed score covectors and query-conditioned native displacement.

These are observation-law accounts. They neither identify psychological traits
nor assert that a local logit direction is a unique internal transformer path.
"""

from __future__ import annotations

import numpy as np

from grain_research.construction.math_action_observability import (
    family_weights,
    moments,
)

PARTS = ("B", "G", "C")


def source_anchor(rows, gradients):
    """Average source-only reference covectors in the common fixed answer chart."""
    g = np.asarray(gradients, float)
    if not rows or g.ndim != 2 or len(g) != len(rows) or not np.isfinite(g).all():
        raise ValueError("matched finite source covectors required")
    if any(r["role"] != "source" for r in rows):
        raise ValueError("held observations cannot define a source anchor")
    if len({r["query"] for r in rows}) != 1:
        raise ValueError("one query per observation covector anchor required")
    w = family_weights([r["family"] for r in rows])
    return w @ g, w


def accounts(displacements, reference_gradients, anchor, curvature):
    """Keep fixed sensitivity B, reference variation G and curvature C separately."""
    d, g, a, c = (
        np.asarray(x, float)
        for x in (displacements, reference_gradients, anchor, curvature)
    )
    if (
        d.ndim != 2
        or g.shape != d.shape
        or a.shape != (d.shape[1],)
        or c.shape != (len(d),)
        or not all(np.isfinite(x).all() for x in (d, g, a, c))
    ):
        raise ValueError("matched finite native displacements/covectors required")
    b = d @ a
    ref = np.sum(d * (g - a), axis=1)
    return np.column_stack((b, ref, c))


def covariance_accounts(a, b, weights):
    """Return all nine signed cross-query terms, without independence assumptions."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 2 or a.shape[1] != 3 or b.shape != a.shape:
        raise ValueError("paired three-account action matrices required")
    cross = moments(np.column_stack((a, b)), weights)["covariance"][:3, 3:]
    terms = {
        qa + qb: float(cross[i, j])
        for i, qa in enumerate(PARTS)
        for j, qb in enumerate(PARTS)
    }
    tangent = sum(terms[k] for k in ("BB", "BG", "GB", "GG"))
    sensitivity = tangent - terms["BB"]
    link = sum(terms[k] for k in ("BC", "GC", "CB", "CG", "CC"))
    native = float(
        moments(np.column_stack((a.sum(axis=1), b.sum(axis=1))), weights)["covariance"][
            0, 1
        ]
    )
    return {
        **terms,
        "native_covariance": native,
        "fixed_covariance": terms["BB"],
        "tangent_covariance": tangent,
        "sensitivity_increment": sensitivity,
        "link_increment": link,
        "recombination_error": abs(native - tangent - link),
        "fixed_native_sign_agree": bool(np.sign(native) == np.sign(terms["BB"])),
        "sensitivity_native_ratio": sensitivity / native if native else None,
        "link_native_ratio": link / native if native else None,
    }


def displacement_cross_covariance(a, b, weights):
    """Preserve the complete cross-query natural-coordinate covariance block."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 2 or b.shape != a.shape:
        raise ValueError("matched native natural-coordinate displacements required")
    m = moments(np.column_stack((a, b)), weights)
    return m["covariance"][: a.shape[1], a.shape[1] :]
