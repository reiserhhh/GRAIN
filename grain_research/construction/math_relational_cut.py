#!/usr/bin/env python3
"""Complete relational cuts and signed finite interactions, not output fitting."""

from __future__ import annotations

import numpy as np

from grain_research.construction.math_construct_routing import token_positions
from grain_research.construction.math_relational_carrier import aligned_pair

ARMS = ("names", "names_properties", "body", "suffix", "full", "random_names", "noop")


def partition(row, encoded):
    """Address body, suffix and name tokens from actual rendered character spans."""
    if list(encoded["input_ids"]) != row["input_ids"]:
        raise ValueError("Tokenizer changed actual input")
    rendered, body = row["rendered"], row["body"]
    if rendered.count(body) != 1:
        raise ValueError("Require one unique full-body address")
    start = rendered.index(body)
    offsets = encoded["offset_mapping"]
    body_positions = token_positions(offsets, start, start + len(body))
    if body_positions != list(range(body_positions[0], body_positions[-1] + 1)):
        raise ValueError("Noncontiguous full body")
    if body_positions[-1] != row["body_endpoint"]:
        raise ValueError("Body endpoint changed")
    markers = []
    for actor in ("A", "B"):
        a = row["properties"][actor][0]
        if body[a - 9 : a] != "Person " + actor + " ":
            raise ValueError("Name marker is not adjacent to its property")
        markers.extend(token_positions(offsets, start + a - 2, start + a - 1))
    names = sorted(set(markers))
    properties = sorted(row["property_positions"])
    suffix = list(range(body_positions[-1] + 1, len(row["input_ids"])))
    if len(names) != 2 or set(names) & set(properties) or not suffix:
        raise ValueError("Require two disjoint names and a complete nonempty suffix")
    if not set(names + properties) <= set(body_positions):
        raise ValueError("Evidence address outside the full body")
    return {
        "prefix": list(range(body_positions[0])),
        "body": body_positions,
        "suffix": suffix,
        "names": names,
        "properties": properties,
        "length": len(row["input_ids"]),
    }


def prove_pair(receiver, donor, a, b):
    """Prove that the only different input IDs are the intended two owner markers."""
    aligned_pair(receiver, donor)
    if a != b:
        raise ValueError("Different complete-cut token addresses")
    changed = [
        i
        for i, (x, y) in enumerate(
            zip(receiver["input_ids"], donor["input_ids"], strict=True)
        )
        if x != y
    ]
    if changed != a["names"]:
        raise ValueError("Extra changed tokens or ineffective name exchange")
    coverage = a["prefix"] + a["body"] + a["suffix"]
    if coverage != list(range(a["length"])):
        raise ValueError("Incomplete or overlapping full-cut partition")
    return {**a, "changed_input_positions": changed}


def patch_values(proof, current, donor, arm, pair, seed):
    """Replace exact position states; random controls match each fresh name delta norm."""
    current, donor = np.asarray(current), np.asarray(donor)
    if (
        arm not in ARMS
        or current.shape != donor.shape
        or current.ndim != 2
        or current.shape[0] != proof["length"]
        or not np.isfinite(current).all()
        or not np.isfinite(donor).all()
    ):
        raise ValueError("Finite complete aligned states and planned arm required")
    if arm == "names_properties":
        positions = sorted(proof["names"] + proof["properties"])
    elif arm in ("full", "noop"):
        positions = proof["body"] + proof["suffix"]
    else:
        positions = proof["names"] if arm == "random_names" else proof[arm]
    if arm == "noop":
        values = current[positions].copy()
    elif arm == "random_names":
        delta = donor[positions].astype(float) - current[positions].astype(float)
        random = np.random.default_rng(seed + pair // 2).normal(size=delta.shape)
        random *= (np.linalg.norm(delta, axis=1) / np.linalg.norm(random, axis=1))[
            :, None
        ]
        values = (
            current[positions].astype(float) + (1 - 2 * (pair % 2)) * random
        ).astype(current.dtype)
    else:
        values = donor[positions].copy()
    return positions, values


def four_cells(y00, y10, y01, y11):
    """Retain the body/query mixed finite difference without an independence assumption."""
    if not np.isfinite([y00, y10, y01, y11]).all():
        raise ValueError("Finite four-cell scores required")
    body, suffix = y10 - y00, y01 - y00
    interaction = y11 - y10 - y01 + y00
    return {
        "y00": float(y00),
        "y10": float(y10),
        "y01": float(y01),
        "y11": float(y11),
        "body_action": float(body),
        "suffix_action": float(suffix),
        "interaction": float(interaction),
        "joint_action": float(y11 - y00),
        "reconstruction_error": float(y11 - y00 - body - suffix - interaction),
    }


def call_allowed(status, spec, elapsed):
    """Unknown or terminal attempts cannot be repeated by a resumed collector."""
    return (
        status["in_flight"] is None
        and status["status"] not in ("COLLECTED", "STOPPED")
        and status["attempted_forward_calls"] < spec["max_forwards"]
        and elapsed < spec["max_model_seconds"]
    )
