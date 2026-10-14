"""Source-coordinate decomposition and native response accounting for HM124.

The source-sample projection is ordinary least squares, not concept erasure or
a personality model. Its complement means linearly unrepresented, not noise.
"""

from __future__ import annotations

import numpy as np

ARMS = ("baseline", "grain_part", "remainder", "whole", "random_matched")


def source_decomposition(coordinates, values, rows):
    """Fit one affine source observation map without target answers or held data."""
    g, x = np.asarray(coordinates, float), np.asarray(values, float)
    if g.shape != (96, 3, 4) or x.ndim != 2 or x.shape[0] != 96:
        raise ValueError(
            "All96 source records, three GRAIN slices and vectors required"
        )
    if len(rows) != 96 or any(r["variant"] != 0 for r in rows):
        raise ValueError("Original-expression source only")
    if not np.isfinite(g).all() or not np.isfinite(x).all():
        raise ValueError("Finite source observations required")
    keys = {
        (r["wording"], r["family"], r["condition"], r["polarity"], r["owner"])
        for r in rows
    }
    expected = {
        (w, f, c, b, o)
        for w in range(2)
        for f in range(6)
        for c in (-1, 1)
        for b in (-1, 1)
        for o in (-1, 1)
    }
    if keys != expected:
        raise ValueError("Every source corner must appear once")
    a = np.column_stack((np.ones(96), g.reshape(96, 12)))
    u, singular, vh = np.linalg.svd(a, full_matrices=False)
    keep = singular > np.finfo(float).eps * max(a.shape) * singular[0]
    p = u[:, keep] @ u[:, keep].T
    represented, remainder = p @ x, x - p @ x
    coefficient = (vh[keep].T / singular[keep]) @ (u[:, keep].T @ x)
    actions = {
        k: np.zeros((6, 2, x.shape[1])) for k in ("grain_part", "remainder", "whole")
    }
    counts = np.zeros((6, 2), int)
    for i, row in enumerate(rows):
        f, c = row["family"], (row["condition"] + 1) // 2
        sign = row["polarity"] * row["owner"]
        for name, array in (
            ("grain_part", represented),
            ("remainder", remainder),
            ("whole", x),
        ):
            actions[name][f, c] += sign * array[i]
        counts[f, c] += 1
    if not np.all(counts == 8):
        raise ValueError("Both wordings and four behavior/owner corners required")
    actions = {name: array / counts[..., None] for name, array in actions.items()}
    return {
        "design": a,
        "projector": p,
        "coefficient": coefficient,
        "singular": singular,
        "rank": np.asarray(keep.sum()),
        "represented": represented,
        "remainder_values": remainder,
        **actions,
    }


def random_control(reference, physical_shape, seed=124):
    """Match remainder norms per physical KV head, without native response access."""
    ref = np.asarray(reference, float).reshape(6, 2, *physical_shape)
    output = np.zeros_like(ref)
    for f, c in np.ndindex(6, 2):
        rng = np.random.default_rng(np.random.SeedSequence([seed, f, c]))
        z = rng.normal(size=physical_shape)
        target = np.linalg.norm(ref[f, c], axis=(0, 2), keepdims=True)
        norm = np.linalg.norm(z, axis=(0, 2), keepdims=True)
        output[f, c] = z * target / norm
    return output


def actor_focus_field(readings):
    """Preserve behavior x actor x queried-person response, in four native units."""
    n = np.asarray(readings, float)
    if n.shape != (5, 2, 6, 2, 2, 2, 6, 4) or not np.isfinite(n).all():
        raise ValueError(
            "All arms/expressions/families/behavior/owner/focus/targets/units required"
        )
    fields = np.zeros((5, 2, 6, 6, 4))
    for b, o, focus in np.ndindex(2, 2, 2):
        fields += (
            (2 * b - 1) * (2 * o - 1) * (1 - 2 * focus) * n[:, :, :, b, o, focus] / 8
        )
    effects = fields[0:1] - fields
    return {
        "baseline": fields[0],
        "fields": fields,
        "effects": effects,
        "interaction": effects[3] - effects[1] - effects[2],
    }


def response_summary(fields):
    """Describe full matrices and signed directional effects, never prediction accuracy."""
    rows = []
    for wording, unit in np.ndindex(2, 2):
        baseline = fields["baseline"][wording, :, :, unit]
        denominator = float(np.sum(baseline**2))
        for arm in range(1, 5):
            effect = fields["effects"][arm, wording, :, :, unit]
            norm = float(np.linalg.norm(effect))
            base_norm = float(np.linalg.norm(baseline))
            rows.append(
                {
                    "wording": wording,
                    "unit": ("score", "log_odds")[unit],
                    "arm": ARMS[arm],
                    "baseline_rms": float(np.sqrt(np.mean(baseline**2))),
                    "effect_rms": float(np.sqrt(np.mean(effect**2))),
                    "effect_to_baseline_norm": norm / base_norm
                    if base_norm > 0
                    else None,
                    "baseline_projection": float(np.sum(effect * baseline))
                    / denominator
                    if denominator > 0
                    else None,
                    "cosine": float(np.sum(effect * baseline)) / (norm * base_norm)
                    if norm * base_norm > 0
                    else None,
                    "diagonal_rms": float(np.sqrt(np.mean(np.diag(effect) ** 2))),
                    "off_diagonal_rms": float(
                        np.sqrt(np.mean(effect[~np.eye(6, dtype=bool)] ** 2))
                    ),
                }
            )
    return rows
