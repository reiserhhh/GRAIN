#!/usr/bin/env python3
"""Discrete multi-target evidence compatibility, not prediction optimization."""

from __future__ import annotations

import itertools

import numpy as np


def corners(n: int) -> np.ndarray:
    """Return ordered -1/+1 corners for a declared finite design."""
    if not isinstance(n, int) or n < 2:
        raise ValueError("At least two finite design variables required")
    return np.array(list(itertools.product((-1, 1), repeat=n)), dtype=float)


def table(values: np.ndarray, n: int) -> np.ndarray:
    """Validate and flatten a complete vector-valued corner table."""
    a = np.asarray(values, dtype=float)
    if a.ndim < 2 or a.shape[0] != 2**n or not np.isfinite(a).all():
        raise ValueError("Complete finite vector observations required")
    return a.reshape(2**n, -1)


def conditional_edges(values: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Give every oriented edge and its low-corner address."""
    a = table(values, n)
    x = corners(n)
    low = np.array([np.flatnonzero(x[:, j] < 0) for j in range(n)])
    d = np.array([a[ids + 2 ** (n - 1 - j)] - a[ids] for j, ids in enumerate(low)])
    return d, low


def unit_line(direction: np.ndarray) -> np.ndarray | None:
    """Normalize a nonzero finite line; flag cancellation instead of inventing it."""
    b = np.asarray(direction, dtype=float)
    if b.ndim != 1 or not np.isfinite(b).all():
        raise ValueError("Finite vector direction required")
    length = np.linalg.norm(b)
    return b / length if length > 1e-12 else None


def line_split(values: np.ndarray, direction: np.ndarray) -> dict:
    """Orthogonally separate arbitrary signed scalar gains from changing directions."""
    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all() or a.shape[1:] != direction.shape:
        raise ValueError("Matched finite edge vectors and direction required")
    u = unit_line(direction)
    if u is None:
        return {"identified": False, "total_energy": float(np.square(a).sum())}
    gain = a @ u
    normal = a - gain[:, None] * u
    total = float(np.square(a).sum())
    normal_energy = float(np.square(normal).sum())
    return {
        "identified": True,
        "unit": u,
        "gain": gain,
        "normal": normal,
        "total_energy": total,
        "normal_energy": normal_energy,
        "parallel_energy": float(np.square(gain).sum()),
        "normal_fraction": normal_energy / total if total > 0 else None,
        "negative_gain_count": int(np.count_nonzero(gain < 0)),
    }


def square_rows(values: np.ndarray, directions: np.ndarray, n: int) -> list[dict]:
    """Enumerate compatibility identities and exact four-edge projection bounds."""
    a = table(values, n)
    b = np.asarray(directions, dtype=float)
    if b.shape != (n, a.shape[1]) or not np.isfinite(b).all():
        raise ValueError("One fixed finite direction per evidence required")
    x, rows = corners(n), []
    for j, k in itertools.combinations(range(n), 2):
        sj, sk = 2 ** (n - 1 - j), 2 ** (n - 1 - k)
        for i in np.flatnonzero((x[:, j] < 0) & (x[:, k] < 0)):
            mm, pm, mp, pp = int(i), int(i + sj), int(i + sk), int(i + sj + sk)
            dj0, dj1 = a[pm] - a[mm], a[pp] - a[mp]
            dk0, dk1 = a[mp] - a[mm], a[pp] - a[pm]
            interaction, alternate = dj1 - dj0, dk1 - dk0
            ej = line_split(np.stack([dj0, dj1]), b[j])
            ek = line_split(np.stack([dk0, dk1]), b[k])
            row = {
                "j": j,
                "k": k,
                "low_corner": mm,
                "corners": [mm, pm, mp, pp],
                "interaction": interaction,
                "interaction_energy": float(interaction @ interaction),
                "compatibility_error": float(np.max(np.abs(interaction - alternate))),
                "identified": ej["identified"] and ek["identified"],
            }
            if row["identified"]:
                uj, uk = ej["unit"], ek["unit"]
                pj = interaction - (interaction @ uj) * uj
                pk = interaction - (interaction @ uk) * uk
                cosine = float(np.clip(uj @ uk, -1, 1))
                exact_bound = float((pj @ pj + pk @ pk) / 2)
                row.update(
                    absolute_line_cosine=abs(cosine),
                    separation=1 - abs(cosine),
                    four_edge_normal_energy=ej["normal_energy"] + ek["normal_energy"],
                    projection_bound=exact_bound,
                    angle_bound=(1 - abs(cosine)) * row["interaction_energy"] / 2,
                    j_normal_interaction_energy=float(pj @ pj),
                    k_normal_interaction_energy=float(pk @ pk),
                )
            rows.append(row)
    return rows


def synthetic_world(kind: str, seed: int, n: int = 6) -> dict:
    """Generate actual finite mechanism worlds with distinct declared explanations."""
    rng = np.random.default_rng(seed)
    x = corners(n)
    q = np.linalg.qr(rng.normal(size=(n, n)))[0]
    if kind == "additive":
        values = x @ q.T
        return {"native": values, "preanswer": values.copy()}
    if kind == "common_scalar_nonlinear":
        scalar = x @ np.linspace(0.2, 0.7, n)
        values = (scalar + 0.15 * scalar**3)[:, None] * q[:, 0]
        return {"native": values, "preanswer": values.copy()}
    if kind == "selective_interaction":
        values = x @ q.T + 0.4 * (x[:, 0] * x[:, 1])[:, None] * q[:, 2]
        return {"native": values, "preanswer": values.copy()}
    if kind == "link_only":
        slopes = rng.normal(size=(n, n, 10)) * 0.4
        logits = np.einsum("cj,jqt->cqt", x, slopes)
        logits += rng.normal(size=(n, 10))[None]
        shifted = logits - logits.max(axis=-1, keepdims=True)
        prob = np.exp(shifted)
        prob /= prob.sum(-1, keepdims=True)
        return {
            "native": prob @ (np.arange(10) / 9),
            "preanswer": logits - logits.mean(-1, keepdims=True),
        }
    raise ValueError("Unknown finite mechanism world")
