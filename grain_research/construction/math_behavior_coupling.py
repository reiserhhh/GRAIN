#!/usr/bin/env python3
"""Read criterion-qualified joint behavior laws from paired vector proxies."""

from __future__ import annotations

import numpy as np

from grain_research.construction import math_proxy_measurement as M

VERSION = "HM276-paired-behavior-coupling-v1"
FRAME = ("fixed_opportunity_A", "fixed_opportunity_B")
BITS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])


def pair_codes(vectors, alphabet):
    """Decode a role-bound ordered pair, not an arbitrary natural embedding."""
    v = np.asarray(vectors, float)
    if v.ndim != 3 or v.shape[1] != 2:
        raise ValueError("Two ordered vector events per paired observation required")
    code = M.decode(v.reshape(-1, v.shape[-1]), alphabet).reshape(-1, 2)
    if np.any(code[:, 0] > 1) or np.any(code[:, 1] < 2):
        raise ValueError("A/B role-bound alphabet addresses required")
    return 2 * code[:, 0] + code[:, 1] - 2


def observed_joint(z_vectors, w_vectors, outcomes, weights, units, alphabet, frame):
    """Compile qualified observed paired records without a latent-regime input."""
    z, w = pair_codes(z_vectors, alphabet), pair_codes(w_vectors, alphabet)
    y, p, ids = np.asarray(outcomes), np.asarray(weights, float), np.asarray(units)
    if (
        tuple(frame) != FRAME
        or len(z) != len(w)
        or y.shape != (len(w), 2)
        or p.shape != (len(w),)
        or ids.shape != (len(w), 6)
        or not np.isin(y, (0, 1)).all()
        or not np.isfinite(p).all()
        or np.any(p <= 0)
        or len(set(ids.flatten().tolist())) != ids.size
    ):
        raise ValueError(
            "Distinct source events, fixed frame and positive census required"
        )
    joint = np.zeros((4, 4, 4))
    np.add.at(joint, (z, w, 2 * y[:, 0] + y[:, 1]), p)
    return joint / joint.sum()


def compile_bundle(joint, alphabet, criterion_id, reference_law):
    """Build four outcome bridges from observed moments, not hidden parameters."""
    j = np.asarray(joint, float)
    if (
        j.shape != (4, 4, 4)
        or not np.isfinite(j).all()
        or np.any(j < 0)
        or not np.isclose(j.sum(), 1)
        or not isinstance(criterion_id, str)
        or not criterion_id
    ):
        raise ValueError("Complete paired source law and criterion ID required")
    zw, zy = j.sum(axis=2), j.sum(axis=1)
    mass = zw.sum(axis=1)
    if np.any(mass <= 0):
        raise ValueError("Unobserved proxy support, no pseudocount bridge")
    bridge = M.bridge_solution(zw / mass[:, None], zy / mass[:, None])
    reference = read_law(bridge, reference_law)
    if reference["status"] != "ADMISSIBLE_JOINT_BEHAVIOR_LAW":
        raise ValueError("Qualified compatible reference law required")
    return {
        "version": VERSION,
        "frame": list(FRAME),
        "criterion_id": criterion_id,
        "alphabet": np.asarray(alphabet, float).tolist(),
        "bridge": bridge,
        "reference": reference,
        "contract": "Source paired operational actions; not named human traits",
    }


def read_law(bridge, probability):
    """Retain whole behavior distribution and coupling, without clipping weights."""
    readings = M.functional_reading(bridge, probability)
    if any(x["status"] != "IDENTIFIED_MOMENT_FUNCTIONAL" for x in readings):
        return {"status": "JOINT_FUNCTIONAL_NOT_IDENTIFIED", "cells": readings}
    q = np.array([x["value"] for x in readings])
    if np.any(q < -1e-9) or np.any(q > 1 + 1e-9) or abs(q.sum() - 1) > 1e-9:
        return {"status": "INADMISSIBLE_JOINT_READING", "unclipped_cells": q.tolist()}
    mean = q @ BITS
    weights = np.asarray(bridge["representative"]) @ BITS
    event_weight_product = float(np.asarray(probability) @ np.prod(weights, axis=1))
    return {
        "status": "ADMISSIBLE_JOINT_BEHAVIOR_LAW",
        "probabilities_00_01_10_11": q.tolist(),
        "marginal_means": mean.tolist(),
        "coupling": float(q[3] - np.prod(mean)),
        "product_of_author_means_coupling": 0.0,
        "paired_single_weight_product_coupling": float(
            event_weight_product - np.prod(mean)
        ),
        "cells_are_author_law_not_single_event_probabilities": True,
    }


def cold_profile(bundle, record):
    """Score a new paired vector law; no application behavior, U or model needed."""
    if (
        set(bundle)
        != {
            "version",
            "frame",
            "criterion_id",
            "alphabet",
            "bridge",
            "reference",
            "contract",
        }
        or bundle["version"] != VERSION
    ):
        raise ValueError("Strict source-only bundle schema required")
    if set(record) != {"id", "frame", "vectors", "weights", "units"}:
        raise ValueError("Cold input excludes behavior outcomes and hidden parameters")
    if tuple(bundle["frame"]) != FRAME or tuple(record["frame"]) != FRAME:
        raise ValueError("Fixed opportunity frame required")
    codes = pair_codes(record["vectors"], bundle["alphabet"])
    p, ids = np.asarray(record["weights"], float), np.asarray(record["units"])
    if (
        p.shape != codes.shape
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or p.sum() <= 0
        or ids.shape != (len(p), 2)
        or len(set(ids.flatten().tolist())) != ids.size
    ):
        raise ValueError(
            "Distinct paired evidence units and nonnegative weights required"
        )
    law = np.bincount(codes, weights=p, minlength=4) / p.sum()
    reading = read_law(bundle["bridge"], law)
    return {
        "id": record["id"],
        "criterion_id": bundle["criterion_id"],
        "frame": list(FRAME),
        "vector_pair_law": law.tolist(),
        "reading": reading,
        "reference": bundle["reference"],
        "relative_coupling": (
            reading["coupling"] - bundle["reference"]["coupling"]
            if reading["status"] == "ADMISSIBLE_JOINT_BEHAVIOR_LAW"
            else None
        ),
    }
