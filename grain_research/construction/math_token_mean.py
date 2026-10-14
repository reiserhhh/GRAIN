"""HM12: separate token-mean displacement from within-span organization."""

from __future__ import annotations

import numpy as np


def mean_centered_displacements(current, donor):
    """Return two matched matrices whose sum is the full donor displacement."""
    h, target = (np.asarray(a, dtype=float) for a in (current, donor))
    if (
        h.ndim != 2
        or min(h.shape) < 1
        or target.shape != h.shape
        or not np.isfinite(h).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("finite aligned nonempty token matrices required")
    delta = target - h
    mean = np.broadcast_to(delta.mean(axis=0), delta.shape).copy()
    return mean, delta - mean


def patch_target(current, donor, arm):
    """Reconstruct the actual float32 assignment, retaining one-token exact no-ops."""
    mean, centered = mean_centered_displacements(current, donor)
    if arm not in ("mean_only", "centered_only"):
        raise ValueError("unknown mean/centered intervention")
    h = np.asarray(current, float)
    displacement = mean if arm == "mean_only" else centered
    return (h + displacement).astype(np.float32)


def geometry_audit(current, donor, arm):
    """Measure ideal decomposition and actual full-dimensional rounded mean changes."""
    h, target = (np.asarray(a, float) for a in (current, donor))
    mean, centered = mean_centered_displacements(h, target)
    actual = patch_target(h, target, arm).astype(float)
    shift = actual - h
    ideal = mean if arm == "mean_only" else centered
    return {
        "tokens": len(h),
        "full_displacement_norm": float(np.linalg.norm(target - h)),
        "mean_displacement_norm": float(np.linalg.norm(mean)),
        "centered_displacement_norm": float(np.linalg.norm(centered)),
        "intervention_norm": float(np.linalg.norm(shift)),
        "ideal_split_error": float(np.max(np.abs(mean + centered - target + h))),
        "ideal_centered_mean_max": float(np.max(np.abs(centered.mean(axis=0)))),
        "rounded_mean_shift_norm": float(np.linalg.norm(shift.mean(axis=0))),
        "rounded_mean_shift_max": float(np.max(np.abs(shift.mean(axis=0)))),
        "rounding_max": float(np.max(np.abs(shift - ideal))),
        "is_exact_input_noop": bool(np.array_equal(actual, h)),
    }


def finite_response_split(baseline, full, mean, centered):
    """Separate mean/centered actions and their finite mixed response without additivity."""
    b, f, m, c = (float(a) for a in (baseline, full, mean, centered))
    if not np.isfinite([b, f, m, c]).all():
        raise ValueError("four finite reader observations required")
    return {
        "full_action": f - b,
        "mean_action": m - b,
        "centered_action": c - b,
        "mixed_action": f - m - c + b,
        "full_minus_mean": f - m,
        "centered_minus_baseline": c - b,
    }


def enumerate_interventions(pairs, inputs, spec):
    """Freeze every pair/query/arm with its original property-token coordinates."""
    if spec["arms"] != ["mean_only", "centered_only"] or spec["layer"] != 16:
        raise ValueError("fixed two arms and original layer16 required")
    if len({p["pair_id"] for p in pairs}) != len(pairs):
        raise ValueError("unique pair IDs required")
    rows = []
    for pair in pairs:
        for query in spec["queries"]:
            receiver = inputs[pair["receiver_doc"], query]
            donor = inputs[pair["donor_doc"], query]
            name = f"{pair['axis']}_property"
            a, b = receiver["spans"][name], donor["spans"][name]
            if (
                not a["positions"]
                or a["token_ids"] != b["token_ids"]
                or a["raw"] != b["raw"]
                or len(a["positions"]) != len(b["positions"])
                or max(a["positions"]) >= receiver["question_start"]
            ):
                raise ValueError("same-word causal property alignment required")
            for arm in spec["arms"]:
                rows.append(
                    {
                        **pair,
                        "unit": len(rows),
                        "layer": spec["layer"],
                        "query": query,
                        "arm": arm,
                        "span": name,
                        "positions": a["positions"],
                        "donor_positions": b["positions"],
                        "token_ids": a["token_ids"],
                        "tokens": len(a["positions"]),
                        "opportunity": "single"
                        if len(a["positions"]) == 1
                        else "multi",
                    }
                )
    return rows
