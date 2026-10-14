#!/usr/bin/env python3
"""Event-resolved edit accounts under fixed SOURCE path observation laws."""

from __future__ import annotations

import numpy as np

from .math_path_functional import profile_bits


def gap_patterns(removed=4):
    """Enumerate the complete triple gap support for an at-most-k deletion."""
    if type(removed) is not int or removed < 0:
        raise ValueError("Nonnegative integer deletion budget required")
    return np.array(
        [(a, b) for a in range(1, removed + 2) for b in range(1, removed + 3 - a)],
        int,
    )


def gap_mass(path, gap):
    """Return equally weighted overlapping observations for one declared gap."""
    x, d = np.asarray(path), np.asarray(gap)
    if (
        x.ndim != 1
        or x.dtype.kind not in "iu"
        or np.any((x < 0) | (x >= 8))
        or d.shape != (2,)
        or d.dtype.kind not in "iu"
        or np.any(d < 1)
        or len(x) <= d.sum()
    ):
        raise ValueError("Supported eight-state history and positive two-gap required")
    a, b = map(int, d)
    n = len(x) - a - b
    out = np.zeros((8, 8, 8))
    np.add.at(out, (x[:n], x[a : a + n], x[a + b :]), 1 / n)
    return out


def compile_source(paths, targets, removed=4):
    """Compile all full-profile/gap laws; never fit native or application answers."""
    x, y = np.asarray(paths), np.asarray(targets)
    if (
        x.ndim != 3
        or x.shape[1:] != (2, 18)
        or len(x) < 2
        or y.shape != (len(x), 5)
        or not np.isin(y, (0, 1)).all()
    ):
        raise ValueError(
            "Aligned two18-record source windows/five binary targets required"
        )
    codes = y.astype(int) @ (1 << np.arange(5))
    gaps = gap_patterns(removed)
    tables = np.zeros((len(gaps), 32, 8, 8, 8))
    for i, d in enumerate(gaps):
        mass = np.array([[gap_mass(row, d) for row in author] for author in x])
        np.add.at(tables[i], codes, mass.mean(1) / len(x))
    reference = tables.sum(1)
    positive = np.einsum("gq,dgabc->dqabc", profile_bits(), tables)
    fields = np.divide(
        positive,
        reference[:, None],
        out=np.full_like(positive, np.nan),
        where=reference[:, None] > 0,
    )
    prior = y.mean(0)
    if (
        np.max(abs(tables.sum((2, 3, 4)) - np.bincount(codes, minlength=32) / len(x)))
        > 1e-12
    ):
        raise ValueError("Gap changed the SOURCE profile prior")
    return {"gaps": gaps, "fields": fields, "prior": prior, "tables": tables}


def validate_record(record):
    """Require exact event identities, chronology and declared original positions."""
    if set(record) != {"author_key", "events", "edits"}:
        raise ValueError("Only observed event metadata and declared edits allowed")
    events, edit = record["events"], record["edits"]
    if (
        not isinstance(record["author_key"], str)
        or not record["author_key"]
        or len(events) not in (12, 18)
        or any(set(e) != {"event_id", "timestamp", "position", "cell"} for e in events)
        or set(edit) != {"support", "counter"}
    ):
        raise ValueError("Complete registered history/edit schema required")
    ids = [e["event_id"] for e in events]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Evidence identities must be distinct")
    positions = np.array([e["position"] for e in events])
    times = np.array([e["timestamp"] for e in events], float)
    cells = np.array([e["cell"] for e in events])
    if (
        positions.dtype.kind not in "iu"
        or np.any(np.diff(positions) != 1)
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or cells.dtype.kind not in "iu"
        or np.any((cells < 0) | (cells >= 8))
    ):
        raise ValueError(
            "Consecutive original record positions/strict times/valid states required"
        )
    for group in edit.values():
        if (
            not isinstance(group, list)
            or len(group) != 4
            or len(set(group)) != 4
            or any(i not in ids for i in group)
        ):
            raise ValueError(
                "Four distinct removal identities inside this history required"
            )
    if set(edit["support"]) & set(edit["counter"]):
        raise ValueError("Removal comparisons must be disjoint")
    return ids, cells


def triplets(indices):
    """Return actual adjacency in the supplied event-identity subsequence."""
    ix = list(indices)
    return [tuple(ix[k : k + 3]) for k in range(len(ix) - 2)]


def mean_supported(values, dimensions=5):
    """An incomplete history is unobserved, never a partial-score success."""
    a = np.asarray(values, float).reshape(-1, dimensions)
    if not len(a) or not np.isfinite(a).all():
        return np.full(dimensions, np.nan)
    return a.mean(0)


def edit_accounts(fields, gaps, prior, record):
    """Keep removal, splice, normalization and known-gap law actions separately."""
    ids, cells = validate_record(record)
    fields, gaps, prior = map(np.asarray, (fields, gaps, prior))
    if fields.shape != (15, 5, 8, 8, 8) or prior.shape != (5,):
        raise ValueError("Complete frozen fifteen-gap five-target instrument required")
    if not np.array_equal(gaps, gap_patterns()):
        raise ValueError("Changed entire gap support")
    lookup = {tuple(d): i for i, d in enumerate(gaps)}
    old = triplets(range(len(cells)))

    def values(triples, known=False):
        rows = []
        for a, b, c in triples:
            d = (b - a, c - b) if known else (1, 1)
            rows.append(fields[lookup[d], :, cells[a], cells[b], cells[c]] - prior)
        return np.asarray(rows, float).reshape(-1, 5)

    original = values(old)
    result = {"full": mean_supported(original).tolist(), "arms": {}}
    ledgers = []
    for name, removed in record["edits"].items():
        keep = [i for i, identity in enumerate(ids) if identity not in removed]
        new = triplets(keep)
        retained = [t for t in old if t in new]
        destroyed = [t for t in old if t not in new]
        created = [t for t in new if t not in old]
        v_r, v_d, v_b = [values(t) for t in (retained, destroyed, created)]
        v_gap = values(created, True)
        m, mp = len(old), len(new)
        blind, known = mean_supported(values(new)), mean_supported(values(new, True))
        parts = np.stack(
            (
                -v_d.sum(0) / m,
                v_b.sum(0) / mp,
                (1 / mp - 1 / m) * v_r.sum(0),
                (v_gap - v_b).sum(0) / mp,
            )
        )
        result["arms"][name] = {
            "blind": blind.tolist(),
            "known_gap": known.tolist(),
            "parts": parts.tolist(),
            "full_supported": bool(np.isfinite(original).all()),
            "blind_supported": bool(np.isfinite(values(new)).all()),
            "known_gap_supported": bool(np.isfinite(values(new, True)).all()),
            "blind_triple_coverage": float(np.isfinite(values(new)).all(1).mean()),
            "known_gap_triple_coverage": float(
                np.isfinite(values(new, True)).all(1).mean()
            ),
            "retained": len(retained),
            "destroyed": len(destroyed),
            "created": len(created),
            "old_triples": m,
            "edited_triples": mp,
        }
        for role, triples in (
            ("retained", retained),
            ("destroyed", destroyed),
            ("created", created),
        ):
            for a, b, c in triples:
                ledgers.append(
                    {
                        "arm": name,
                        "role": role,
                        "event_ids": [ids[i] for i in (a, b, c)],
                        "indices": [a, b, c],
                        "gap": [b - a, c - b],
                        "cells": [int(cells[i]) for i in (a, b, c)],
                    }
                )
    s, c = (result["arms"][k] for k in ("support", "counter"))
    result["blind_action"] = (np.array(c["blind"]) - np.array(s["blind"])).tolist()
    result["known_gap_action"] = (
        np.array(c["known_gap"]) - np.array(s["known_gap"])
    ).tolist()
    result["part_actions"] = (np.array(c["parts"]) - np.array(s["parts"])).tolist()
    result["ledger"] = ledgers
    return result
