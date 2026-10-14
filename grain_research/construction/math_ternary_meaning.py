#!/usr/bin/env python3
"""Coherent source-defined target meanings with every profile-pair law fixed."""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import numpy as np

VERSION = "HM180_source_ternary_target_meaning_v1"
MODELS = ("pair_constrained", "observed_triple_assignment")
TARGETS = ("E", "N", "A", "C", "O")


def code_hash():
    """Identify the constructor independently of outputs or targets selected later."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def margins(t):
    """Return GAB,GAC,GBC,ABC, the four three-way marginal constraints."""
    return tuple(np.asarray(t).sum(axis) for axis in (3, 2, 1, 0))


def homogeneous(t, tolerance=1e-10, max_sweeps=10000):
    """Maximize finite-support entropy keeping every declared lower margin."""
    t = np.asarray(t, float)
    if t.ndim != 4 or not np.isfinite(t).all() or np.any(t < 0):
        raise ValueError("Finite nonnegative profile-by-three-event law required")
    targets = margins(t)
    support = np.ones(t.shape, bool)
    for axis, target in zip((3, 2, 1, 0), targets, strict=True):
        support &= np.expand_dims(target > 0, axis)
    x = support.astype(float)
    if t.sum() == 0:
        return x, {"sweeps": 0, "margin_error": 0.0, "active_cells": 0}
    x *= t.sum() / x.sum()
    for sweep in range(1, max_sweeps + 1):
        for axis, target in zip((3, 2, 1, 0), targets, strict=True):
            current = x.sum(axis)
            if np.any((target > 0) & (current == 0)):
                raise ValueError("Incompatible constrained support")
            ratio = np.divide(
                target, current, out=np.zeros_like(target), where=current > 0
            )
            x *= np.expand_dims(ratio, axis)
        error = max(
            float(np.max(np.abs(a - b)))
            for a, b in zip(margins(x), targets, strict=True)
        )
        if error <= tolerance:
            return x, {
                "sweeps": sweep,
                "margin_error": error,
                "active_cells": int(support.sum()),
            }
    raise ValueError(f"Fixed IPF budget exhausted; sweeps={max_sweeps},error={error}")


def construct(p, y, tolerance=1e-10, max_sweeps=10000):
    """Use one author-balanced sampling law, retaining full profile coherence."""
    p, y = np.asarray(p, float), np.asarray(y)
    if (
        p.ndim != 3
        or p.shape[1:] != (3, 8)
        or len(p) < 2
        or y.shape != (len(p), 5)
        or not np.isin(y, (0, 1)).all()
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.allclose(p.sum(2), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError(
            "Normalized author three-role laws and five binary targets required"
        )
    profiles, group = np.unique(y, axis=0, return_inverse=True)
    full = np.array(
        [
            np.einsum(
                "ua,ub,uc->abc", p[group == g, 0], p[group == g, 1], p[group == g, 2]
            )
            / len(p)
            for g in range(len(profiles))
        ]
    )
    reference, receipts = np.zeros_like(full), []
    for z in itertools.product(range(2), repeat=3):
        block = tuple(slice(4 * c, 4 * c + 4) for c in z)
        key = (slice(None), *block)
        reference[key], receipt = homogeneous(full[key], tolerance, max_sweeps)
        receipts.append({"conditions": list(z), **receipt})
    return profiles, np.stack((reference, full)), receipts


def profile_fields(tables):
    """Keep a joint conditional profile probability, including missing support."""
    tables = np.asarray(tables, float)
    total = tables.sum(1)
    return np.divide(
        tables,
        total[:, None],
        out=np.full_like(tables, np.nan),
        where=total[:, None] > 0,
    )


def fields(profiles, tables):
    """Derive all five probabilities from the SAME coherent joint profile law."""
    t, c = np.asarray(tables, float), np.asarray(profiles)
    result = []
    for q in range(5):
        positive = t[:, c[:, q] == 1].sum(1)
        negative = t[:, c[:, q] == 0].sum(1)
        total = positive + negative
        result.append(
            np.divide(positive, total, out=np.full_like(total, np.nan), where=total > 0)
        )
    return np.stack(result, axis=1)


def statistics(profiles, tables):
    """Separate generic joint divergence from five target projections and signs."""
    h, f = np.asarray(tables, float)
    active = f > 0
    if np.any(h[active] <= 0):
        raise ValueError("Observed law outside constrained reference support")
    kl = float(np.sum(f[active] * np.log(f[active] / h[active])))
    psi = fields(profiles, tables)
    weight, delta = f.sum(0), psi[1] - psi[0]
    used = weight > 0
    d = delta[:, used]
    rms = np.sqrt(np.sum(weight[used] * d * d, axis=1))
    gram = (d * weight[used]) @ d.T
    target_kl = []
    for q in range(5):
        pf, ph = psi[1, q][used], psi[0, q][used]
        value = np.zeros(len(pf))
        for a, b in ((pf, ph), (1 - pf, 1 - ph)):
            positive = a > 0
            if np.any(b[positive] <= 0):
                raise ValueError("Target projection outside reference support")
            value[positive] += a[positive] * np.log(a[positive] / b[positive])
        target_kl.append(float(weight[used] @ value))
    error = max(
        float(np.max(np.abs(a - b)))
        for a, b in zip(margins(f), margins(h), strict=True)
    )
    return {
        "joint_kl_nats": kl,
        "target_kl_nats": target_kl,
        "target_field_rms": rms.tolist(),
        "target_gram": gram.tolist(),
        "margin_error": error,
        "supported_triples": int(used.sum()),
        "unsupported_triples": int((~used).sum()),
    }


def build_bundle(p, y, tolerance=1e-10, max_sweeps=10000):
    """Compile two fixed views, without natural outputs or application labels."""
    p = np.asarray(p, float)
    if p.ndim != 4 or p.shape[1:] != (2, 3, 8):
        raise ValueError("Two fixed views of three eight-cell roles required")
    tables, receipts = [], []
    for v in range(2):
        profiles, t, receipt = construct(p[:, v], y, tolerance, max_sweeps)
        tables.append(t)
        receipts.append(receipt)
    return {
        "version": VERSION,
        "code_hash": code_hash(),
        "models": list(MODELS),
        "targets": list(TARGETS),
        "profiles": profiles.tolist(),
        "profile_tables": np.asarray(tables).tolist(),
        "solver_receipts": receipts,
        "denominator": 64,
        "role_size": 4,
        "constraints": ["GAB", "GAC", "GBC", "ABC"],
        "interpretation": "random_one_event_per_role;not_whole_history_posterior",
    }


def validate_bundle(b):
    """Reject undeclared fields, missing profiles and inconsistent lower laws."""
    keys = {
        "version",
        "code_hash",
        "models",
        "targets",
        "profiles",
        "profile_tables",
        "solver_receipts",
        "denominator",
        "role_size",
        "constraints",
        "interpretation",
    }
    if (
        set(b) != keys
        or b["version"] != VERSION
        or b["code_hash"] != code_hash()
        or b["models"] != list(MODELS)
        or b["targets"] != list(TARGETS)
        or b["denominator"] != 64
        or b["role_size"] != 4
        or b["constraints"] != ["GAB", "GAC", "GBC", "ABC"]
        or b["interpretation"]
        != "random_one_event_per_role;not_whole_history_posterior"
    ):
        raise ValueError("Unknown observation law or unexpected outcome field")
    c, t = np.asarray(b["profiles"]), np.asarray(b["profile_tables"], float)
    if (
        c.ndim != 2
        or c.shape[1] != 5
        or not np.isin(c, (0, 1)).all()
        or len(np.unique(c, axis=0)) != len(c)
        or t.shape != (2, 2, len(c), 8, 8, 8)
        or not np.isfinite(t).all()
        or np.any(t < 0)
    ):
        raise ValueError("Invalid coherent SOURCE tables")
    for v in range(2):
        if not np.allclose(t[v].sum((1, 2, 3, 4)), 1, rtol=0, atol=1e-9):
            raise ValueError("Source law is not normalized")
        if statistics(c, t[v])["margin_error"] > 8e-10:
            raise ValueError("Changed lower constraints")
    return b


def triple_sum(field, cells, slots):
    """Preserve original-frame mass; absent triples are not a neutral score."""
    f, c, s = np.asarray(field, float), np.asarray(cells), np.asarray(slots)
    if (
        f.shape[-3:] != (8, 8, 8)
        or c.ndim != 1
        or c.shape != s.shape
        or not np.issubdtype(c.dtype, np.integer)
        or not np.issubdtype(s.dtype, np.integer)
        or np.any((c < 0) | (c > 7))
        or np.any((s < 0) | (s > 11))
        or len(set(s.tolist())) != len(s)
    ):
        raise ValueError("Distinct original slots and eight-cell events required")
    roles = [c[(s // 4) == r] for r in range(3)]
    n = int(np.prod([len(r) for r in roles]))
    value, known = np.zeros(f.shape[:-3]), np.ones(f.shape[:-3], bool)
    for a, b, d in itertools.product(*roles):
        x = f[..., a, b, d]
        known &= np.isfinite(x)
        value += x
    value = np.where(known, value / 64, np.nan)
    average = value * 64 / n if n else np.full_like(value, np.nan)
    return value, average, known, n


def contact_action(field, cells, selected, control):
    """Direct three-role R&S versus R&C contact account on addressed evidence."""
    c, s, ctrl = np.asarray(cells), set(selected), set(control)
    if (
        c.shape != (12,)
        or len(s) != 4
        or len(ctrl) != 4
        or s & ctrl
        or not (s | ctrl) <= set(range(12))
    ):
        raise ValueError("Twelve addressed events, disjoint four-event S/C required")
    rest, f = set(range(12)) - s - ctrl, np.asarray(field, float)
    value = np.zeros(f.shape[:-3])
    for ids in itertools.product(range(4), range(4, 8), range(8, 12)):
        chosen = set(ids)
        sign = int(chosen <= rest | s and bool(chosen & rest) and bool(chosen & s))
        sign -= int(
            chosen <= rest | ctrl and bool(chosen & rest) and bool(chosen & ctrl)
        )
        if sign:
            value += sign * f[..., *(c[list(ids)])] / 64
    return value


def cold_record(bundle, record):
    """Strict numerical event-only inference with jointly coherent target scores."""
    if set(record) != {"author_key", "view", "events"}:
        raise ValueError("Cold input excludes outcomes/native answers")
    v, events = record["view"], record["events"]
    if (
        type(v) is not int
        or v not in (0, 1)
        or not isinstance(record["author_key"], str)
        or not isinstance(events, list)
    ):
        raise ValueError("Declared view, author key and event list required")
    if any(set(e) != {"event_id", "slot", "cell"} for e in events):
        raise ValueError("Only event ID, original slot and cell allowed")
    ids = [e["event_id"] for e in events]
    if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Distinct event IDs required")
    slots, cells = [e["slot"] for e in events], [e["cell"] for e in events]
    if any(type(i) is not int for i in slots + cells):
        raise ValueError("Integer slots/cells required")
    profile = profile_fields(np.asarray(bundle["profile_tables"], float)[v])
    mass, mean, known, n = triple_sum(
        profile, np.asarray(cells, int), np.asarray(slots, int)
    )
    c = np.asarray(bundle["profiles"], float)
    clean = lambda a: np.where(np.isfinite(a), a, None).tolist()
    return {
        "author_key": record["author_key"],
        "view": v,
        "events": len(events),
        "triples": n,
        "profile_mass_readout": clean(mass),
        "conditional_profile_average": clean(mean),
        "target_mass_readout": clean(mass @ c),
        "conditional_target_average": clean(mean @ c),
        "supported": known.all(1).tolist(),
        "status": "TRIPLE_OBSERVED" if n else "NO_TRIPLE_OPPORTUNITY",
    }


def source_folds(source_ids, targets):
    """Fix two author-disjoint folds, balancing full profiles without any search."""
    ids, y = np.asarray(source_ids), np.asarray(targets)
    if y.shape != (len(ids), 5) or len(np.unique(ids)) != len(ids):
        raise ValueError("Unique SOURCE author IDs and full profiles required")
    fold = np.empty(len(ids), int)
    _, group = np.unique(y, axis=0, return_inverse=True)
    for g in np.unique(group):
        indices = np.flatnonzero(group == g)
        if len(indices) < 2:
            raise ValueError("Profile has no author-disjoint support")
        ordered = indices[np.argsort(ids[indices])]
        fold[ordered] = np.arange(len(ordered)) % 2
    return fold
