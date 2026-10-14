#!/usr/bin/env python3
"""Source-fixed relational observations on new expressions of declared events."""

from __future__ import annotations

import itertools

import numpy as np

from .math_common_observables import finite_matrix, quadratic_functions
from .math_heterogeneous_evidence_cube import TARGETS, grid
from .math_passive_event_composition import action_line

FORMS = ("paraphrase", "reordered")
PAIRS = tuple(itertools.permutations(TARGETS, 2))


def documents(spec):
    """Keep factual cell identities fixed and alter wording/presentation only."""
    if tuple(spec["forms"]) != FORMS or spec["source_family"] != 0:
        raise ValueError("Fixed forms and family0 source required")
    if set(spec["behaviours"]) != set(TARGETS):
        raise ValueError("All six targets required")
    result = []
    for form, (pair_id, (focal, companion)) in itertools.product(
        FORMS, enumerate(PAIRS)
    ):
        for corner, (x, y, g) in enumerate(grid(3)):
            actor = "A" if g == 1 else "B"
            alternate = "B" if g == 1 else "A"
            events = [
                {"event": "K1", "actor": "A", "target": focal, "polarity": int(x)},
                {
                    "event": "K2",
                    "actor": actor,
                    "target": companion,
                    "polarity": int(y),
                },
                {"event": "K3", "actor": alternate, "target": "anchor", "polarity": 0},
            ]
            lines = [
                action_line(
                    e["event"],
                    e["actor"],
                    spec["anchor"]
                    if e["target"] == "anchor"
                    else spec["behaviours"][e["target"]][int(e["polarity"] > 0)],
                )
                for e in events
            ]
            if form == "reordered":
                lines.reverse()
            result.append(
                {
                    "doc_id": len(result),
                    "form": form,
                    "pair_id": pair_id,
                    "focal": focal,
                    "companion": companion,
                    "corner": corner,
                    "coordinates": [int(x), int(y), int(g)],
                    "events": events,
                    "presentation": ["K1", "K2", "K3"]
                    if form == "paraphrase"
                    else ["K3", "K2", "K1"],
                    "body": "\n".join(lines),
                }
            )
    return result


def cold_chart(values, calibration):
    """Apply frozen source coefficients, without cold fitting or recentering."""
    values, calibration = finite_matrix(values), finite_matrix(calibration)
    if calibration.shape[0] != values.shape[1] + 1:
        raise ValueError("Source coefficient/input shape mismatch")
    return np.column_stack((np.ones(len(values)), values)) @ calibration


def moments(probabilities, z):
    """Retain event-local products and separately return independent pairing."""
    p, z = finite_matrix(probabilities), finite_matrix(z)
    if p.shape[1] != len(z) or np.any(p < 0) or np.max(abs(p.sum(1) - 1)) > 1e-12:
        raise ValueError("Normalized profiles on identical declared units required")
    means = p @ z
    second = np.einsum("ue,ei,ej->uij", p, z, z)
    return {
        "means": means,
        "second": second,
        "independent_second": np.einsum("ui,uj->uij", means, means),
        "lifted": p @ quadratic_functions(z)[0],
    }


def contrast(probabilities, z):
    """Observe one fixed paired-profile difference, not a fitted predictor."""
    if np.asarray(probabilities).shape[0] != 2:
        raise ValueError("Exactly two profiles required")
    obs = moments(probabilities, z)
    return {name: value[1] - value[0] for name, value in obs.items()}


def feature_delta(delta):
    """Use the full second tensor to preserve orthogonal coordinate invariance."""
    return np.concatenate((delta["means"], delta["second"].ravel()))


def polynomial_kernel(z):
    """Fixed degree-two linear-plus-quadratic kernel, not universally injective."""
    z = finite_matrix(z)
    gram = z @ z.T
    return gram + gram * gram


def cosine(a, b):
    """Return direction alignment only when both vectors are nonzero."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Comparable finite directions required")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denominator == 0 else float(np.clip((a @ b) / denominator, -1, 1))


def perturbation_certificate(probabilities, source, cold):
    """Bound cold moment discrepancies without choosing a tolerance for success."""
    source, cold = finite_matrix(source), finite_matrix(cold)
    if source.shape != cold.shape:
        raise ValueError("Same declared support and source coordinates required")
    ds, dc = contrast(probabilities, source), contrast(probabilities, cold)
    epsilon = float(np.max(np.linalg.norm(cold - source, axis=1)))
    radius = float(np.max(np.linalg.norm(source, axis=1)))
    total_variation_l1 = float(np.sum(abs(probabilities[1] - probabilities[0])))
    mean_bound = total_variation_l1 * epsilon
    second_bound = total_variation_l1 * epsilon * (2 * radius + epsilon)
    source_distance = float(np.linalg.norm(feature_delta(ds)))
    joint_bound = float(np.hypot(mean_bound, second_bound))
    return {
        "epsilon": epsilon,
        "source_radius": radius,
        "contrast_l1": total_variation_l1,
        "mean_bound": mean_bound,
        "second_bound": second_bound,
        "joint_bound": joint_bound,
        "mean_discrepancy": float(np.linalg.norm(dc["means"] - ds["means"])),
        "second_discrepancy": float(np.linalg.norm(dc["second"] - ds["second"])),
        "source_distance": source_distance,
        "certified_separation_lower": max(0.0, source_distance - joint_bound),
    }
