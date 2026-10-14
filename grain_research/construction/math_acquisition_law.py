#!/usr/bin/env python3
"""SOURCE acquisition-conditioned fields, distinct from native prediction."""

from __future__ import annotations

import numpy as np

MODES = ("full", "selected", "remainder")


def select_top(vectors, poles, k=4):
    """Apply the frozen absolute semantic rank with historical stable ties."""
    v, p = np.asarray(vectors, float), np.asarray(poles, float)
    if (
        v.ndim < 2
        or p.shape != (2, v.shape[-1])
        or not np.isfinite(v).all()
        or not np.isfinite(p).all()
        or not isinstance(k, int)
        or k < 1
        or k >= v.shape[-2]
    ):
        raise ValueError("Finite event vectors and two compatible poles required")
    vn, pn = np.linalg.norm(v, axis=-1, keepdims=True), np.linalg.norm(p, axis=1)
    if (vn <= 0).any() or (pn <= 0).any():
        raise ValueError("Zero vectors do not define semantic selection")
    signed = (v / vn) @ (p[1] / pn[1] - p[0] / pn[0])
    ids = np.sort(np.argsort(-np.abs(signed), axis=-1, kind="stable")[..., :k])
    return ids, signed


def selection_mask(ids, size=12):
    """Retain selected event identities without replacing their original order."""
    a = np.asarray(ids)
    if (
        a.ndim < 1
        or not np.issubdtype(a.dtype, np.integer)
        or a.shape[-1] < 1
        or a.shape[-1] >= size
        or ((a < 0) | (a >= size)).any()
        or (np.diff(np.sort(a, axis=-1), axis=-1) == 0).any()
    ):
        raise ValueError("Distinct in-range event identities required")
    m = np.zeros(a.shape[:-1] + (size,), bool)
    np.put_along_axis(m, a, True, axis=-1)
    return m


def compile_fields(cells, selected, targets):
    """Compile three equal-author laws; missing cells stay unobserved, not0."""
    c, s, y = np.asarray(cells), np.asarray(selected), np.asarray(targets, float)
    if (
        c.ndim != 3
        or not np.issubdtype(c.dtype, np.integer)
        or ((c < 0) | (c > 7)).any()
        or s.shape != c.shape
        or s.dtype != np.dtype(bool)
        or y.ndim != 2
        or y.shape[0] != c.shape[0]
        or c.shape[0] == 0
        or not np.isin(y, [0, 1]).all()
        or not np.all(s.sum(-1) == s.sum(-1).flat[0])
        or np.any((s.sum(-1) == 0) | (s.sum(-1) == c.shape[-1]))
    ):
        raise ValueError(
            "Author-linked cells, fixed-size selection and binary targets required"
        )
    masks = (np.ones_like(s), s, ~s)
    masses, joints, fields = [], [], []
    for take in masks:
        counts = np.stack([(take & (c == a)).sum(axis=(1, 2)) for a in range(8)], 1)
        prob = counts / take.sum(axis=(1, 2))[:, None]
        rho, joint = prob.mean(0), prob.T @ y / len(y)
        psi = np.full_like(joint, np.nan)
        np.divide(joint, rho[:, None], out=psi, where=rho[:, None] > 0)
        masses.append(rho)
        joints.append(joint)
        fields.append(psi)
    masses, joints, fields = map(np.asarray, (masses, joints, fields))
    fraction = float(s.mean())
    rate = np.full(8, np.nan)
    np.divide(fraction * masses[1], masses[0], out=rate, where=masses[0] > 0)
    mixture = np.nansum(
        np.stack([rate[:, None] * fields[1], (1 - rate[:, None]) * fields[2]]),
        axis=0,
    )
    mixture[masses[0] == 0] = np.nan
    return {
        "prior": y.mean(0),
        "reference": masses,
        "joint": joints,
        "fields": fields,
        "support": masses > 0,
        "selection_probability": rate,
        "mixture": mixture,
    }


def evidence_accounts(cells, selected, fields, prior):
    """Separate a fixed reader's evidence action from a changed protocol reader."""
    c, ids, f, pi = map(np.asarray, (cells, selected, fields, prior))
    if (
        c.shape != (12,)
        or not np.issubdtype(c.dtype, np.integer)
        or ((c < 0) | (c > 7)).any()
        or ids.shape != (4,)
        or f.ndim != 3
        or f.shape[:2] != (3, 8)
        or pi.shape != (f.shape[2],)
        or not np.isfinite(pi).all()
    ):
        raise ValueError("Fixed12-event/4-selection/three-field interface required")
    take = selection_mask(ids)
    rows = []
    full = f[0, c].mean(0) - pi
    for mode, use, index in (("selected", take, 1), ("remainder", ~take, 2)):
        fixed_value = f[0, c[use]].mean(0) - pi
        mode_value = f[index, c[use]].mean(0) - pi
        rows.append(
            {
                "mode": mode,
                "full": full,
                "fixed_action": fixed_value - full,
                "protocol_drift": mode_value - fixed_value,
                "total": mode_value - full,
                "value": mode_value,
            }
        )
    return rows


def cross_transport(fields, reference):
    """Describe a field across different SOURCE authors AND disjoint text views."""
    f, rho = np.asarray(fields, float), np.asarray(reference, float)
    if f.ndim != 5 or f.shape[:4] != (2, 2, 3, 8) or rho.shape != (8,):
        raise ValueError("Two banks/two views/three fixed observation modes required")
    d = f[:, :, 1] - f[:, :, 2]
    if not np.isfinite(d).all() or not np.isfinite(rho).all() or (rho < 0).any():
        raise ValueError("Cross-transport requires shared observed support")
    pairs = ((d[0, 0], d[1, 1]), (d[0, 1], d[1, 0]))
    products = np.array([(rho[:, None] * a * b).sum(0) for a, b in pairs])
    energy = (rho[None, None, :, None] * d**2).sum(2).mean((0, 1))
    signed = products.mean(0)
    coherence = np.full_like(energy, np.nan)
    np.divide(signed, energy, out=coherence, where=energy > 0)
    return {"cross_product": signed, "field_energy": energy, "coherence": coherence}
