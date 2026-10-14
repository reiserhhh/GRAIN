#!/usr/bin/env python3
"""Fixed ownership interventions, with explicit source averaging and no fitting."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

ARMS = (
    "high",
    "low",
    "both",
    "body_endpoint",
    "query_end",
    "source_mean",
    "random",
    "noop",
)


def span(row, which):
    """Return actual token positions and captured-array indices for semantic polarity."""
    if which not in ("high", "low") or row["behaviour"] not in (0, 1):
        raise ValueError("Explicit high/low behavioural ownership required")
    high_actor = "A" if row["behaviour"] == 0 else "B"
    actor = high_actor if which == "high" else ("B" if high_actor == "A" else "A")
    n = row["property_a_count"]
    indices = (
        list(range(n))
        if actor == "A"
        else list(range(n, len(row["property_positions"])))
    )
    return [row["property_positions"][i] for i in indices], indices


def aligned_pair(receiver, donor):
    """Prove fixed semantic tokens, positions, order, length and query suffix."""
    for key in ("role", "content", "query", "person", "body_endpoint"):
        if receiver[key] != donor[key]:
            raise ValueError("Changed matched field " + key)
    if any(
        r["appraisal"] != "none" or r["order"] != r["behaviour"]
        for r in (receiver, donor)
    ):
        raise ValueError("Use absent-appraisal, high-first ownership pairs only")
    if donor["behaviour"] != 1 - receiver["behaviour"]:
        raise ValueError("Require a genuine ownership exchange")
    a, b = receiver["input_ids"], donor["input_ids"]
    if (
        len(a) != len(b)
        or a[receiver["body_endpoint"] + 1 :] != b[donor["body_endpoint"] + 1 :]
    ):
        raise ValueError("Changed full length or query suffix")
    result = {}
    for which in ("high", "low"):
        p, _ = span(receiver, which)
        q, _ = span(donor, which)
        if not p or p != q or [a[i] for i in p] != [b[i] for i in q]:
            raise ValueError("No resampling: actual property identity required")
        result[which] = p
    if set(result["high"]) & set(result["low"]):
        raise ValueError("Overlapping properties")
    return {**result, "length": len(a), "query_end": len(a) - 1}


def pairs(rows, targets):
    """Select all predeclared matched-query pairs, including both directions."""
    selected = [
        r
        for r in rows
        if r["appraisal"] == "none"
        and r["query"] == r["content"]
        and r["order"] == r["behaviour"]
    ]
    lookup = {
        (r["role"], r["content"], r["person"], r["behaviour"]): r for r in selected
    }
    result = []
    for role in ("source", "application"):
        for content in targets:
            for person in ("A", "B"):
                for direction in (0, 1):
                    receiver = lookup[role, content, person, direction]
                    donor = lookup[role, content, person, 1 - direction]
                    proof = aligned_pair(receiver, donor)
                    result.append(
                        {
                            "pair": len(result),
                            "role": role,
                            "content": content,
                            "person": person,
                            "direction": direction,
                            "receiver": receiver["unit"],
                            "donor": donor["unit"],
                            "alignment": proof,
                        }
                    )
    if len(lookup) != len(result) or len(result) != 8 * len(targets):
        raise ValueError("Incomplete matched scope")
    return result


def source_vectors(pair_rows, rows, captures, targets):
    """Average tokens, persons, then contents equally; never read application arrays."""
    groups = defaultdict(list)
    for pair in pair_rows:
        if pair["role"] != "source" or pair["direction"] != 0:
            continue
        receiver, donor = rows[pair["receiver"]], rows[pair["donor"]]
        if receiver["role"] != "source" or donor["role"] != "source":
            raise ValueError("Application capture cannot construct a source vector")
        values = []
        for which in ("high", "low"):
            _, i = span(receiver, which)
            _, j = span(donor, which)
            values.append(
                np.mean(
                    np.asarray(captures[donor["unit"]]["property"])[j].astype(float)
                    - np.asarray(captures[receiver["unit"]]["property"])[i].astype(
                        float
                    ),
                    axis=0,
                )
            )
        groups[pair["content"]].append(np.stack(values))
    if set(groups) != set(targets) or any(len(v) != 2 for v in groups.values()):
        raise ValueError("Exactly two queried people for each source content")
    per_content = np.stack([np.mean(groups[t], axis=0) for t in targets])
    return np.mean(per_content, axis=0), per_content


def random_vectors(shared, seed):
    """Draw one frozen direction per property, with source-mean vector norms."""
    shared = np.asarray(shared, float)
    if shared.ndim != 2 or shared.shape[0] != 2 or not np.isfinite(shared).all():
        raise ValueError("Two finite source vectors required")
    random = np.random.default_rng(seed).normal(size=shared.shape)
    return (
        random
        * (np.linalg.norm(shared, axis=1) / np.linalg.norm(random, axis=1))[:, None]
    )


def patch_values(receiver, donor, current, other, shared, random, arm):
    """Build exact frozen-layer replacement values; retain all unselected positions."""
    aligned_pair(receiver, donor)
    if arm not in ARMS:
        raise ValueError("Unplanned arm")
    if arm in ("body_endpoint", "query_end"):
        key = "endpoint" if arm == "body_endpoint" else "query_end"
        p = (
            receiver["body_endpoint"]
            if key == "endpoint"
            else len(receiver["input_ids"]) - 1
        )
        return [p], np.asarray(other[key])[None].copy()
    selected = (arm,) if arm in ("high", "low") else ("high", "low")
    positions, values = [], []
    sign = 1 if receiver["behaviour"] == 0 else -1
    for which in selected:
        p, i = span(receiver, which)
        _, j = span(donor, which)
        original = np.asarray(current["property"])[i]
        if arm in ("source_mean", "random"):
            vector = (shared if arm == "source_mean" else random)[
                0 if which == "high" else 1
            ]
            value = (original.astype(float) + sign * vector).astype(original.dtype)
        elif arm == "noop":
            value = original.copy()
        else:
            value = np.asarray(other["property"])[j].copy()
        positions.extend(p)
        values.append(value)
    if len(set(positions)) != len(positions):
        raise ValueError("Duplicate patch position")
    return positions, np.concatenate(values)


def effects(native, donor, patched):
    """Keep signed action, full input exchange and remaining distance, not accuracy."""
    full, action = float(donor - native), float(patched - native)
    return {
        "full_change": full,
        "patch_change": action,
        "remaining": float(donor - patched),
        "same_direction": bool(action * full > 0),
        "ratio": action / full if full != 0 else None,
    }


def call_allowed(status, spec, elapsed):
    """Reject unknown calls, repeated terminal acquisition, or finite budget overflow."""
    return (
        status["in_flight"] is None
        and status["status"] != "COLLECTED"
        and status["attempted_forward_calls"] < spec["max_forwards"]
        and elapsed < spec["max_model_seconds"]
    )
