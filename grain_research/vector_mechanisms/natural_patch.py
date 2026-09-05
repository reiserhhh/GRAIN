"""Matched natural-history interventions in a fixed decoder coordinate system."""

from __future__ import annotations

import numpy as np

from .context import QUESTIONS
from .controls import material_hash
from .memory import CURRENT_END
from .scope import DEFINITIONS


def patch_message(history: str, current: str, task: str) -> tuple[str, tuple[int, int]]:
    """Share the entire evidence prefix; task instructions occur strictly after it."""
    prefix = (
        "The earlier and current excerpts are presented as writing by one anonymous writer.\n\n"
        + history
        + "\n\nCurrent excerpt:\n"
    )
    interval = (len(prefix), len(prefix) + len(current))
    prefix += current + CURRENT_END + "\nEvidence ends here.\n\n"
    if task == "disposition":
        question = QUESTIONS[task]
    elif task == "current_guidance":
        question = (
            DEFINITIONS["enacted_guidance"]
            + " Ignore earlier excerpts for this question. Does the CURRENT excerpt "
            "itself perform this action? Assess its communicative function, "
            "not a stable personality trait."
        )
    else:
        raise ValueError("Unknown intervention task")
    return prefix + "Question: " + question + "\nAnswer:", interval


def current_difference_basis(delta: np.ndarray, rank: int) -> tuple[np.ndarray, float]:
    """Compress a paired token-state difference, not a shared personality axis."""
    x = np.asarray(delta, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or rank < 1:
        raise ValueError("Need a finite token-by-hidden difference and positive rank")
    _, singular, vh = np.linalg.svd(x, full_matrices=False)
    if len(singular) < rank or singular[rank - 1] <= 1e-10:
        raise ValueError("Insufficient numerical rank for the declared intervention")
    energy = float(np.sum(singular[:rank] ** 2) / np.sum(singular**2))
    return vh[:rank].T.copy(), energy


def project_rows(delta: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project every token through the same orthonormal hidden-space basis."""
    x, q = np.asarray(delta), np.asarray(basis)
    if x.ndim != 2 or q.ndim != 2 or x.shape[1] != q.shape[0]:
        raise ValueError("Expected token-by-hidden states and hidden-by-rank basis")
    if not np.allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-7):
        raise ValueError("Basis must be orthonormal")
    return (x @ q) @ q.T


def matched_random_rows(
    delta: np.ndarray, target: np.ndarray, rank: int, key
) -> np.ndarray:
    """Use a random hidden subspace with each token's candidate-effect norm."""
    x, target = np.asarray(delta, dtype=float), np.asarray(target, dtype=float)
    if x.ndim != 2 or x.shape != target.shape or not 1 <= rank <= x.shape[1]:
        raise ValueError("Invalid row-matched random intervention")
    seed = int(material_hash(key)[:16], 16)
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(x.shape[1], rank)), mode="reduced")
    random = project_rows(x, q)
    have = np.linalg.norm(random, axis=1, keepdims=True)
    need = np.linalg.norm(target, axis=1, keepdims=True)
    if np.any((have < 1e-12) & (need > 1e-12)):
        raise ValueError("Random subspace cannot match a nonzero candidate effect")
    return random * (need / np.maximum(have, 1e-12))


def match_quantized_row_norms(current, direction, target, quantize, *, steps=40):
    """Match rounded displacement by bounded monotone row-scale search.

    Scaling preserves the random subspace before rounding. Discrete rounding
    can prevent exact matching; report the mismatch without editing coordinates.
    """
    h, d, target = (np.asarray(x, dtype=float) for x in (current, direction, target))
    if h.ndim != 2 or h.shape != d.shape or h.shape != target.shape:
        raise ValueError("Need aligned row matrices")
    if not np.array_equal(quantize(h), h):
        raise ValueError("Current state is not exactly representable")
    need = np.linalg.norm(quantize(target) - h, axis=1)
    if np.any((np.linalg.norm(d, axis=1) == 0) & (need > 0)):
        raise ValueError("Cannot scale a zero direction to a nonzero target")
    lo, hi = np.zeros(len(h)), np.ones(len(h))
    norms = lambda scale: np.linalg.norm(quantize(h + scale[:, None] * d) - h, axis=1)
    for _ in range(12):
        missing = norms(hi) < need
        if not missing.any():
            break
        hi[missing] *= 2
    if np.any(norms(hi) < need):
        raise ValueError("Quantized norm match exceeds scaling budget")
    for _ in range(steps):
        mid = (lo + hi) / 2
        below = norms(mid) < need
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    scale = np.where(np.abs(norms(lo) - need) <= np.abs(norms(hi) - need), lo, hi)
    scale = np.where(need == 0, 0, scale)
    value = h + scale[:, None] * d
    error = np.abs(norms(scale) - need) / np.maximum(need, 1e-12)
    return value, scale, error
