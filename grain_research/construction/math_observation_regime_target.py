#!/usr/bin/env python3
"""SOURCE-valued composition/succession, not a native-answer predictor."""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from .math_path_functional import profile_bits


def bag_map():
    """Map all eight-state triples to120 multiplicity-preserving unordered bags."""
    cells = np.indices((8, 8, 8)).reshape(3, -1).T
    bags, inverse = np.unique(np.sort(cells, axis=1), axis=0, return_inverse=True)
    return bags, inverse


def disintegrate(tables):
    """Retain target values and split their observed-law conditional expectation."""
    t = np.asarray(tables, float)
    if (
        t.ndim != 5
        or t.shape[1:] != (32, 8, 8, 8)
        or not np.isfinite(t).all()
        or np.any(t < 0)
        or not np.allclose(t.sum((1, 2, 3, 4)), 1, atol=2e-12, rtol=0)
    ):
        raise ValueError(
            "Normalized nonnegative gap/profile/triple SOURCE law required"
        )
    ref = t.sum(1).reshape(len(t), 512)
    positive = np.einsum("gq,dgabc->dqabc", profile_bits(), t).reshape(len(t), 5, 512)
    field = np.divide(
        positive,
        ref[:, None],
        out=np.full_like(positive, np.nan),
        where=ref[:, None] > 0,
    )
    bags, mapping = bag_map()
    bag_ref = np.zeros((len(t), len(bags)))
    bag_positive = np.zeros((len(t), 5, len(bags)))
    for cell, group in enumerate(mapping):
        bag_ref[:, group] += ref[:, cell]
        bag_positive[:, :, group] += positive[:, :, cell]
    bag_field = np.divide(
        bag_positive,
        bag_ref[:, None],
        out=np.full_like(bag_positive, np.nan),
        where=bag_ref[:, None] > 0,
    )
    composition = bag_field[:, :, mapping]
    return {
        "reference": ref,
        "field": field,
        "composition": composition,
        "succession": field - composition,
        "bag_reference": bag_ref,
        "bag_field": bag_field,
        "bag_mapping": mapping,
        "prior": positive.sum(2),
    }


def canonical_events(record):
    """Validate identities and recover attested order, never treating display as time."""
    expected = {"record_id", "author", "regime", "events"}
    if set(record) != expected or any(
        not isinstance(record[k], str) or not record[k] for k in ("record_id", "author")
    ):
        raise ValueError("Only addressed observed inputs allowed, not answers/labels")
    if record["regime"] != "raw_pandora_publication":
        raise ValueError("No silent raw-source calibration for another representation")
    events = record["events"]
    schema = {"event_id", "owner", "position", "timestamp", "cell"}
    if (
        not isinstance(events, list)
        or len(events) < 3
        or any(not isinstance(e, dict) or set(e) != schema for e in events)
    ):
        raise ValueError("At least three addressed event observations required")
    ids = [e["event_id"] for e in events]
    if len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("Repeated carriers are not independent event evidence")
    if any(e["owner"] != record["author"] for e in events):
        raise ValueError("Mixed-author evidence requires a different observation law")
    if any(
        type(e["position"]) is not int
        or e["position"] < 0
        or type(e["cell"]) is not int
        or not 0 <= e["cell"] < 8
        or not isinstance(e["timestamp"], (int, float))
        or isinstance(e["timestamp"], bool)
        or not np.isfinite(e["timestamp"])
        for e in events
    ):
        raise ValueError("Finite time and integer original position/cell required")
    ordered = sorted(events, key=lambda e: e["timestamp"])
    if any(
        b["timestamp"] <= a["timestamp"] or b["position"] <= a["position"]
        for a, b in pairwise(ordered)
    ):
        raise ValueError("Strict times and consistent original positions required")
    return ordered


def score_record(fields, composition, prior, gaps, record):
    """Score complete qualified history accounts; preserve unavailable components."""
    events = canonical_events(record)
    f, b, pi = map(np.asarray, (fields, composition, prior))
    if f.ndim != 3 or f.shape[1:] != (5, 512) or b.shape != f.shape or pi.shape != (5,):
        raise ValueError("Five-target frozen SOURCE fields/prior required")
    lookup = {tuple(map(int, gap)): i for i, gap in enumerate(gaps)}
    ledger, fv, bv = [], [], []
    for a, mid, c in zip(events[:-2], events[1:-1], events[2:], strict=True):
        gap = (mid["position"] - a["position"], c["position"] - mid["position"])
        cell = 64 * a["cell"] + 8 * mid["cell"] + c["cell"]
        j = lookup.get(gap)
        value = f[j, :, cell] if j is not None else np.full(5, np.nan)
        bag_value = b[j, :, cell] if j is not None else np.full(5, np.nan)
        fv.append(value)
        bv.append(bag_value)
        ledger.append(
            {
                "event_ids": [a["event_id"], mid["event_id"], c["event_id"]],
                "original_gap": list(gap),
                "cell": cell,
                "full_supported": bool(np.isfinite(value).all()),
                "composition_supported": bool(np.isfinite(bag_value).all()),
            }
        )
    fv, bv = np.array(fv), np.array(bv)
    full_ok, bag_ok = np.isfinite(fv).all(), np.isfinite(bv).all()
    full = fv.mean(0) - pi if full_ok else None
    bag = bv.mean(0) - pi if bag_ok else None
    order = (fv - bv).mean(0) if full_ok and bag_ok else None
    return {
        "record_id": record["record_id"],
        "author": record["author"],
        "regime": record["regime"],
        "semantics": "source-target-associated evidence index,not author posterior",
        "event_ids_in_attested_order": [e["event_id"] for e in events],
        "triples": len(ledger),
        "complete": bool(full_ok and bag_ok),
        "full": full.tolist() if full is not None else None,
        "composition": bag.tolist() if bag is not None else None,
        "succession": order.tolist() if order is not None else None,
        "ledger": ledger,
    }


def conditional_residual_error(reference, residual, mapping):
    """Check weighted zero per bag without treating unsupported cells as observed0."""
    ref, r = np.asarray(reference), np.asarray(residual)
    maximum = 0.0
    for group in np.unique(mapping):
        mask = mapping == group
        mass = ref[:, mask].sum(1)
        weighted = np.sum(
            np.where(
                ref[:, None, mask] > 0,
                np.nan_to_num(r[:, :, mask]) * ref[:, None, mask],
                0,
            ),
            axis=2,
        )
        supported = mass > 0
        if np.any(supported):
            maximum = max(maximum, float(np.max(abs(weighted[supported]))))
    return maximum
