#!/usr/bin/env python3
"""Preserve condition composition and relative expression in relational fields."""

import itertools

import numpy as np

from grain_research.construction import math_relational_target_operator as P

ACCOUNTS = ("CC", "CE", "EC", "EE")
LEGS = tuple(itertools.product(range(2), repeat=2))


def target_accounts(contractions):
    """Sum the five target characters without moving source-account axes."""
    x = np.asarray(contractions)
    if x.shape[-3:] != (4, 4, 31):
        raise ValueError("Four by four oriented accounts and31 characters required")
    return x[..., P.TARGET_MODES].sum(axis=-1)


def templates(vectors, conditions):
    """Fit label-free, event-weighted templates in each observed source window."""
    x, c = np.asarray(vectors, float), np.asarray(conditions)
    if x.ndim != 4 or x.shape[1] != 2 or c.shape != x.shape[:-1]:
        raise ValueError("Two addressed source event windows required")
    if not np.isfinite(x).all() or not np.isin(c, (0, 1)).all():
        raise ValueError("Finite events and binary registered conditions required")
    values = []
    for v in (0, 1):
        row = []
        for z in (0, 1):
            events = x[:, v][c[:, v] == z]
            if not len(events):
                raise ValueError("Unobserved source condition cannot be imputed")
            row.append(events.mean(0))
        values.append(row)
    return np.asarray(values)


def parts(means, proportions, tau):
    """Keep C and E exactly; unknown conditional response is never imputed."""
    x, p, t = (
        np.asarray(means, float),
        np.asarray(proportions, float),
        np.asarray(tau, float),
    )
    if x.ndim < 2 or x.shape[-2] != 2 or p.shape != x.shape[:-1]:
        raise ValueError("Matched two-window means and condition proportions required")
    if t.shape != (2, 2, x.shape[-1]):
        raise ValueError("Two windows by two registered condition templates required")
    if not all(np.isfinite(a).all() for a in (x, p, t)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Finite observations and proportions in [0,1] required")
    choice = t[:, 0] + p[..., None] * (t[:, 1] - t[:, 0])
    return np.stack((choice, x - choice), axis=-2)


def compile_bank(means, proportions, codes, tau):
    """Qualify all HM207 pairing characters without changing its total operator."""
    base = P.compile_bank(means, codes)
    x = parts(means, proportions, tau)
    g = base["codes"]
    cells = np.array([x[g == k].mean(0) for k in range(32)])
    r = x - cells[g]
    pg = np.array([np.asarray(proportions)[g == k].mean(0) for k in range(32)])
    return {
        "a": r[:, 0],
        "b": r[:, 1],
        "w": base["pair_w"],
        "centers": cells.mean(0),
        "templates": np.asarray(tau, float),
        "codes": g,
        "proportion_residuals": np.asarray(proportions) - pg[g],
    }


def contractions(left, right, permutation0=None, permutation1=None):
    """Return all16 signed cross-bank accounts using compact dual products."""
    a, b, c, d = left["a"], left["b"], right["a"], right["b"]
    ga = np.array([[a[:, i] @ c[:, j].T for j in (0, 1)] for i in (0, 1)])
    gb = np.array([[b[:, i] @ d[:, j].T for j in (0, 1)] for i in (0, 1)])
    if permutation0 is not None:
        gb = gb[..., np.asarray(permutation0), :]
    if permutation1 is not None:
        gb = gb[..., np.asarray(permutation1)]
    return gram_contractions(ga, gb, left["w"], right["w"])


def gram_contractions(ga, gb, w, v):
    """Reuse fixed Gram accounts for frozen references, not new model fits."""
    values = np.empty((4, 4, w.shape[0]))
    for i, (a, b) in enumerate(LEGS):
        for j, (c, d) in enumerate(LEGS):
            values[i, j] = np.sum((w @ (ga[a, c] * gb[b, d])) * v, axis=1)
    return values


def read(bank, left_parts, right_parts):
    """Keep four source by four observation accounts, not only matched ones."""
    x, y = np.asarray(left_parts, float), np.asarray(right_parts, float)
    if x.shape != y.shape or x.shape[-2:] != bank["centers"].shape[-2:]:
        raise ValueError("Matched C/E observation legs required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite cold observations required")
    xx, yy = x - bank["centers"][0], y - bank["centers"][1]
    out = np.empty((*x.shape[:-2], 4, 4, bank["w"].shape[0]))
    for i, (a, b) in enumerate(LEGS):
        for j, (c, d) in enumerate(LEGS):
            out[..., i, j, :] = (
                (xx[..., c, :] @ bank["a"][:, a].T)
                * (yy[..., d, :] @ bank["b"][:, b].T)
            ) @ bank["w"].T
    return out


def field(bank, companion_parts, view):
    """Return source-account fields for each own C/E leg and companion account."""
    if view not in (0, 1):
        raise ValueError("Window0 or1 required")
    y = np.asarray(companion_parts, float) - bank["centers"][1 - view]
    if y.shape != bank["centers"][1 - view].shape or not np.isfinite(y).all():
        raise ValueError("One finite companion C/E pair required")
    out = np.empty((4, 4, bank["w"].shape[0], y.shape[-1]))
    for i, (a, b) in enumerate(LEGS):
        own, other = (
            (bank["a"][:, a], bank["b"][:, b])
            if view == 0
            else (bank["b"][:, b], bank["a"][:, a])
        )
        for j, (c, d) in enumerate(LEGS):
            companion = d if view == 0 else c
            out[i, j] = (bank["w"] * (other @ y[companion])[None, :]) @ own
    return out


def cold(bank, record):
    """Score addressed conditions and vectors only, with frozen source references."""
    if set(record) != {"frame_key", "windows"} or len(record["windows"]) != 2:
        raise ValueError("Exactly two addressed windows required")
    means, ps, seen = [], [], set()
    for events in record["windows"]:
        if not events:
            raise ValueError("Missing window is not a zero companion")
        vectors, conditions = [], []
        for event in events:
            if set(event) != {"event_id", "vector", "condition"}:
                raise ValueError("Only event, vector and known condition permitted")
            uid = event["event_id"]
            if not isinstance(uid, str) or not uid or uid in seen:
                raise ValueError("Distinct identified events required")
            seen.add(uid)
            v = np.asarray(event["vector"], float)
            if v.shape != (bank["templates"].shape[-1],) or not np.isfinite(v).all():
                raise ValueError("Finite registered vector dimension required")
            if event["condition"] not in (0, 1):
                raise ValueError("Unobserved condition cannot be inferred from labels")
            vectors.append(v)
            conditions.append(event["condition"])
        means.append(np.mean(vectors, 0))
        ps.append(np.mean(conditions))
    x = parts(np.asarray(means), np.asarray(ps), bank["templates"])
    values = read(bank, x[0], x[1])
    return {
        "frame_key": record["frame_key"],
        "source_by_observation_readings": values.tolist(),
        "total_pair_readings": values.sum((0, 1)).tolist(),
        "condition_proportions": ps,
    }
