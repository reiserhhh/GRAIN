#!/usr/bin/env python3
"""Fixed counterfactual sectors and their observation/composition consequences."""

from __future__ import annotations

import hashlib

import numpy as np

from .math_passive_event_composition import action_line
from .math_target_bound_actions import CHARACTERS, POINTS, SIGNS, TARGETS

ARMS = (
    "remove_yg",
    "remove_xyg",
    "remove_both",
    "remove_xy",
    "random_both",
    "exact_owner",
    "noop",
)


def coefficients(values):
    """Retain all characters of an exactly complete cube with arbitrary tail axes."""
    a = np.asarray(values, dtype=float)
    if a.shape[0] != 8 or not np.isfinite(a).all():
        raise ValueError("Exactly eight finite corners required")
    return np.tensordot(SIGNS / 8, a, axes=(1, 0))


def reconstruct(coef):
    """Evaluate the fixed full character expansion without fitting or truncation."""
    c = np.asarray(coef, dtype=float)
    if c.shape[0] != 8 or not np.isfinite(c).all():
        raise ValueError("Exactly eight finite coefficients required")
    return np.tensordot(SIGNS.T, c, axes=(1, 0))


def documents(spec):
    """Enumerate every directed pair and neutral control without named body traits."""
    if spec["targets"] != list(TARGETS):
        raise ValueError("All six declared targets required")
    rows = []
    for focal in TARGETS:
        for companion in (*[q for q in TARGETS if q != focal], "neutral"):
            for corner, (x, y, g) in enumerate(POINTS.astype(int)):
                a = spec["behaviours"][focal][int(x > 0)]
                b = (
                    spec["neutral"]
                    if companion == "neutral"
                    else spec["behaviours"][companion]
                )[int(y > 0)]
                body = "\n".join(
                    (
                        action_line("K1", "A", a),
                        action_line("K2", "A" if g > 0 else "B", b),
                        action_line("K3", "B" if g > 0 else "A", spec["anchor"]),
                    )
                )
                rows.append(
                    {
                        "doc_id": len(rows),
                        "focal": focal,
                        "companion": companion,
                        "corner": corner,
                        "coordinates": [int(x), int(y), int(g)],
                        "body": body,
                    }
                )
    return rows


def array_hash(value):
    """Bind actual C-contiguous shape/dtype/bytes, separately from archive bytes."""
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(f"{a.dtype.str}:{a.shape}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def payload(cube, arm, seed):
    """Construct all eight assignments using fixed sectors and cube-wise randomness."""
    a = np.asarray(cube)
    if a.ndim != 3 or a.shape[0] != 8 or a.dtype != np.float32:
        raise ValueError("Eight float32 complete suffix states required")
    if arm not in ARMS or not np.isfinite(a).all():
        raise ValueError("Finite states and declared intervention required")
    if arm == "noop":
        return a.copy()
    if arm == "exact_owner":
        return a[np.arange(8) ^ 1].copy()
    c = coefficients(a)
    indices = {
        "remove_yg": (3,),
        "remove_xyg": (7,),
        "remove_both": (3, 7),
        "remove_xy": (6,),
        "random_both": (3, 7),
    }[arm]
    change = np.einsum("ji,jld->ild", SIGNS[list(indices)], c[list(indices)])
    if arm == "random_both":
        random = np.random.default_rng(seed).normal(size=a.shape[1:])
        random /= np.linalg.norm(random, axis=-1, keepdims=True)
        change = (
            SIGNS[3, :, None, None]
            * np.linalg.norm(change, axis=-1, keepdims=True)
            * random[None]
        )
    return (a.astype(float) - change).astype(np.float32)


def ownership_actions(values):
    """Compute both conditional four-cell differences and their equal-x average."""
    a = np.asarray(values, dtype=float).reshape(2, 2, 2, -1)
    conditional = a[:, 1, 1] - a[:, 0, 1] - a[:, 1, 0] + a[:, 0, 0]
    return np.concatenate((conditional.mean(axis=0, keepdims=True), conditional))


def composition_contrasts(values, amplitude=0.5):
    """Score fixed parity populations, retaining their actual probability vectors."""
    if not 0 < amplitude <= 1:
        raise ValueError("A positive probability-preserving amplitude required")
    a = np.asarray(values, float)
    if a.shape[0] != 8 or not np.isfinite(a).all():
        raise ValueError("Complete finite observations required")
    result = {}
    for j in (3, 7):
        plus, minus = (1 + amplitude * SIGNS[j]) / 8, (1 - amplitude * SIGNS[j]) / 8
        p, m = np.tensordot(plus, a, axes=(0, 0)), np.tensordot(minus, a, axes=(0, 0))
        result[CHARACTERS[j]] = {
            "plus_probability": plus,
            "minus_probability": minus,
            "plus_score": p,
            "minus_score": m,
            "difference": p - m,
        }
    return result


def joint_ablation(native, without_yg, without_xyg, without_both):
    """Account exactly for interaction of two interventions, not semantic mediation."""
    a, b, c, d = (
        np.asarray(v, float) for v in (native, without_yg, without_xyg, without_both)
    )
    if not (a.shape == b.shape == c.shape == d.shape):
        raise ValueError("Matched four arms required")
    return a - b - c + d


def toy_world(kind):
    """Distinguish explicit storage from nonlinear reconstruction of identical action."""
    x, y, g = POINTS.T
    if kind == "linear_storage":
        states = np.stack((y, g, y * g), axis=1)[:, None].astype(np.float32)
        reader = lambda v: v[:, 0, 2]
    elif kind == "recomputed":
        states = np.stack((y, g, np.zeros(8)), axis=1)[:, None].astype(np.float32)
        reader = lambda v: v[:, 0, 0] * v[:, 0, 1]
    elif kind == "conditional":
        states = np.stack((x, y, g, x * y * g), axis=1)[:, None].astype(np.float32)
        reader = lambda v: v[:, 0, 3]
    else:
        raise ValueError("Declared toy mechanism required")
    return {
        arm: reader(payload(states, arm, 1))
        for arm in ("noop", "remove_yg", "remove_xyg", "remove_both")
    }
