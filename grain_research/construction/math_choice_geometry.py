#!/usr/bin/env python3
"""Support-qualified choice/displacement accounts without precision fitting."""

import numpy as np

from . import math_condition_pairing as P

VERSION = "HM188_choice_geometry_v1"
COMPONENTS = ("coverage", "choice", "geometry", "coupling")
KINDS = COMPONENTS + ("full", "product", "joint_minus_product")


def observations(vectors, conditions):
    """Keep unobserved geometry missing and known weighted accounts explicit."""
    conditions = np.asarray(conditions)
    first = P.moments(vectors, conditions)
    p, support = first["p"], first["support"]
    a = (1 - 2 * p) * first["opportunity"]
    geometry = np.einsum("...i,...j->...ij", first["delta"], first["delta"])
    weighted = np.zeros_like(geometry)
    weighted[support] = a[support, None, None] * geometry[support]
    between = np.zeros_like(geometry)
    between[support] = first["opportunity"][support, None, None] * geometry[support]
    return first | {
        "a": a,
        "geometry": geometry,
        "weighted": weighted,
        "between": between,
        "counts": np.stack(((conditions == 0).sum(-1), (conditions == 1).sum(-1)), -1),
    }


def reference(a, geometry, support):
    """Separate observed coverage, marginal means and centered matrix covariance."""
    a, g, s = np.asarray(a, float), np.asarray(geometry, float), np.asarray(support)
    if (
        a.ndim != 1
        or g.ndim != 3
        or g.shape[0] != len(a)
        or g.shape[-1] != g.shape[-2]
        or s.shape != a.shape
        or s.dtype != bool
        or not len(a)
        or not np.isfinite(a).all()
        or not np.isfinite(g[s]).all()
    ):
        raise ValueError(
            "Finite aligned supported scalar/operator observations required"
        )
    if not s.any():
        raise ValueError("No supported calibration geometry; do not impute a reference")
    aa, gg = a[s], g[s]
    A, G = aa.mean(), gg.mean(0)
    L = np.mean((aa - A)[:, None, None] * (gg - G), axis=0)
    f = A * G + L
    return {
        "q": float(s.mean()),
        "A": float(A),
        "G": G,
        "L": L,
        "f": f,
        "total": float(s.mean()) * f,
        "product": float(s.mean()) * A * G,
        "authors": len(a),
        "supported": int(s.sum()),
    }


def contrast(low, high):
    """Exactly decompose group contrast; retain coverage-coupling interactions."""
    dq, da = high["q"] - low["q"], high["A"] - low["A"]
    q, a = (high["q"] + low["q"]) / 2, (high["A"] + low["A"]) / 2
    G, f = (high["G"] + low["G"]) / 2, (high["f"] + low["f"]) / 2
    parts = np.stack(
        (dq * f, q * da * G, q * a * (high["G"] - low["G"]), q * (high["L"] - low["L"]))
    )
    total = high["total"] - low["total"]
    product = high["product"] - low["product"]
    return np.concatenate((parts, np.stack((total, product, total - product))))


def fields(a, geometry, support, labels, folds=None):
    """Construct all five target accounts on unchanged natural marginal groups."""
    y = np.asarray(labels)
    P.target_weights(y, folds)
    folds = None if folds is None else np.asarray(folds)
    a, g, s = np.asarray(a), np.asarray(geometry), np.asarray(support)
    if a.shape != (len(y), 2) or g.shape[:2] != a.shape or s.shape != a.shape:
        raise ValueError("Two aligned author views required")
    masks = [np.ones(len(y), bool)] if folds is None else [folds == b for b in (0, 1)]
    bank, rows = [], []
    for b, mask in enumerate(masks):
        views = []
        for v in (0, 1):
            targets = []
            for k, name in enumerate(P.TARGETS):
                refs = []
                for side in (0, 1):
                    ids = mask & (y[:, k] == side)
                    ref = reference(a[ids, v], g[ids, v], s[ids, v])
                    refs.append(ref)
                    rows.append(
                        {
                            "fold": b,
                            "view": v,
                            "target": name,
                            "side": side,
                            "authors": ref["authors"],
                            "supported": ref["supported"],
                            "coverage": ref["q"],
                            "choice_mean": ref["A"],
                        }
                    )
                targets.append(contrast(*refs))
            views.append(targets)
        bank.append(views)
    return {"fields": np.asarray(bank), "coverage": rows}


def statistics(bank):
    """Report all signed cross-fold contractions, not selected predictor scores."""
    f = np.asarray(bank, float)
    if f.ndim != 6 or f.shape[:4] != (2, 2, 5, 7) or not np.isfinite(f).all():
        raise ValueError(
            "Complete two-fold/two-view/five-target/seven-account bank required"
        )
    rows = []
    for k, name in enumerate(KINDS):
        for target in list(range(5)) + [None]:
            ids = [target] if target is not None else list(range(5))
            a, b = f[0][:, ids, k], f[1][:, ids, k]
            cross = float(np.sum(a.mean(0) * b.mean(0)))
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            rows.append(
                {
                    "kind": name,
                    "target": P.TARGETS[target] if target is not None else "all",
                    "signed_product": cross,
                    "coherence": float(2 * cross / norm) if norm else None,
                    "same_view_crossauthor": float(np.sum(a * b) / 2),
                    "different_view_crossauthor": float(np.sum(a * b[::-1]) / 2),
                    "within_fold_crossview": float(
                        (np.sum(a[0] * a[1]) + np.sum(b[0] * b[1])) / 2
                    ),
                }
            )
    return rows


def cross_terms(bank):
    """Retain every oriented component pair, including negative interference."""
    f = np.asarray(bank)
    statistics(f)
    rows = []
    for target in list(range(5)) + [None]:
        ids = [target] if target is not None else list(range(5))
        a, b = f[0][:, ids, :4].mean(0), f[1][:, ids, :4].mean(0)
        for i, first in enumerate(COMPONENTS):
            for j, second in enumerate(COMPONENTS):
                rows.append(
                    {
                        "target": P.TARGETS[target] if target is not None else "all",
                        "first": first,
                        "second": second,
                        "signed_product": float(np.sum(a[:, i] * b[:, j])),
                    }
                )
    return rows


def coupling_permutation(labels, folds, support, rng):
    """Move two-view choices together only inside declared profile/support cells."""
    y, f, s = np.asarray(labels), np.asarray(folds), np.asarray(support)
    P.target_weights(y, f)
    if s.shape != (len(y), 2) or s.dtype != bool:
        raise ValueError("Two-view availability signature required")
    profile = y.astype(int) @ (2 ** np.arange(5))
    signature = s.astype(int) @ np.array([1, 2])
    cells = f * 128 + profile * 4 + signature
    order = np.arange(len(y))
    sizes = []
    for cell in np.unique(cells):
        ids = np.flatnonzero(cells == cell)
        order[ids] = rng.permutation(ids)
        sizes.append(len(ids))
    return order, {
        "groups": len(sizes),
        "singleton_groups": sizes.count(1),
        "movable_authors": sum(n for n in sizes if n > 1),
        "fixed_points": int(np.sum(order == np.arange(len(y)))),
    }


def local_account(ref, a, geometry, supported):
    """Cold decomposition relative to fixed source, never current-batch centering."""
    cover = (int(supported) - ref["q"]) * ref["f"]
    if supported:
        if not np.isfinite(geometry).all():
            raise ValueError("Observed conditional geometry required")
        choice = (a - ref["A"]) * ref["G"]
        response = ref["A"] * (geometry - ref["G"])
        coupling = (a - ref["A"]) * (geometry - ref["G"]) - ref["L"]
    else:
        choice = response = coupling = np.zeros_like(cover)
    return np.stack((cover, choice, response, coupling))


def cold_record(fields, calibration, basis, mapping, record):
    """Read addressed numeric events; return all channels and missing geometry flags."""
    f, q, mapping = np.asarray(fields), np.asarray(basis), np.asarray(mapping)
    if f.shape != (2, 5, 15, 15) or q.shape != (1024, 15) or mapping.shape != q.shape:
        raise ValueError("Fixed complete source bundle required")
    if not all(np.isfinite(x).all() for x in (f, q, mapping)):
        raise ValueError("Finite source bundle required")
    if set(record) != {"frame_key", "events"} or not isinstance(
        record["frame_key"], str
    ):
        raise ValueError("Addressed numeric input only")
    ids, slots, vectors, conditions = [], [], [], []
    for e in record["events"]:
        if set(e) != {"event_id", "slot", "vector", "condition"}:
            raise ValueError("No labels, text or native answers")
        if (
            not isinstance(e["event_id"], str)
            or type(e["slot"]) is not int
            or e["slot"] not in range(12)
            or type(e["condition"]) is not int
            or e["condition"] not in (0, 1)
        ):
            raise ValueError("Original IDs, slots and binary conditions required")
        ids.append(e["event_id"])
        slots.append(e["slot"])
        vectors.append(e["vector"])
        conditions.append(e["condition"])
    if not ids or len(set(ids)) != len(ids) or len(set(slots)) != len(slots):
        raise ValueError("Nonempty unique events required")
    h, c = np.asarray(vectors, float), np.asarray(conditions)
    if h.shape != (len(ids), 1024):
        raise ValueError("Original 1024-dimensional events required")
    m = observations(h @ q, c)
    refs = [{k: np.asarray(v)[view] for k, v in calibration.items()} for view in (0, 1)]
    parts = np.stack(
        [local_account(ref, m["a"], m["geometry"], bool(m["support"])) for ref in refs]
    )
    part_readings = np.einsum("vqij,vkij->vkq", f, parts)
    full = np.einsum(
        "vqij,vij->vq", f, m["weighted"] - np.stack([ref["total"] for ref in refs])
    )
    original = P.moments(h @ mapping, c)
    return {
        "frame_key": record["frame_key"],
        "count": len(ids),
        "counts": m["counts"].tolist(),
        "p": float(m["p"]),
        "a": float(m["a"]),
        "conditional_support": bool(m["support"]),
        "dispersion_support": bool((m["counts"] >= 2).all()),
        "original_GRAIN_mean": original["mean"].tolist(),
        "readings": {
            name: part_readings[:, k].tolist() for k, name in enumerate(COMPONENTS)
        },
        "relative_full": full.tolist(),
        "source_reference": np.einsum(
            "vqij,vij->vq", f, np.stack([ref["total"] for ref in refs])
        ).tolist(),
        "weighted_displacement": np.einsum("vqij,ij->vq", f, m["between"]).tolist(),
        "choice_displacement": np.einsum("vqij,ij->vq", f, m["weighted"]).tolist(),
        "conditional_geometry": np.einsum("vqij,ij->vq", f, m["geometry"]).tolist()
        if m["support"]
        else None,
    }
