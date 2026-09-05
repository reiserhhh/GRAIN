"""Discovery-only contrast spaces and amplitude-matched random controls."""

from __future__ import annotations

import hashlib

import numpy as np


def contrast_basis(differences: np.ndarray, rank: int) -> np.ndarray:
    """Find shared contrast directions without removing the mean contrast."""
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 2 or rank < 1 or rank > min(differences.shape):
        raise ValueError("Unsupported contrast rank")
    if not np.isfinite(differences).all():
        raise ValueError("Non-finite contrast matrix")
    _, singular, right = np.linalg.svd(differences, full_matrices=False)
    if singular[rank - 1] < 1e-8:
        raise ValueError("Requested rank exceeds numerical contrast rank")
    return right[:rank].T


def matched_random_effect(delta, target_effect, rank: int, key: str) -> np.ndarray:
    """Project into a reproducible random space, then match intervention norm.

    Matching the final effect prevents high-rank candidate capture from winning
    simply because its intervention was larger. It is an artificial control,
    not a natural donor activation.
    """
    delta = np.asarray(delta, dtype=float)
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(delta.size, rank)), mode="reduced")
    effect = basis @ (basis.T @ delta)
    desired = np.linalg.norm(target_effect)
    norm = np.linalg.norm(effect)
    if desired == 0:
        return np.zeros_like(delta)
    if norm < 1e-12:
        raise ValueError("Random projection too small for norm matching")
    return effect * (desired / norm)


def fit_margin_threshold(margins, truth) -> float:
    """Fit a single balanced-accuracy threshold using ONLY discovery records."""
    from sklearn.metrics import balanced_accuracy_score

    margins, truth = np.asarray(margins, dtype=float), np.asarray(truth, dtype=int)
    if not np.isfinite(margins).all() or set(truth) != {0, 1}:
        raise ValueError(
            "Threshold fit requires finite discovery scores and both classes"
        )
    unique = np.unique(margins)
    candidates = np.r_[unique[0] - 1, (unique[1:] + unique[:-1]) / 2, unique[-1] + 1]
    scored = [
        (balanced_accuracy_score(truth, margins >= value), -abs(value), value)
        for value in candidates
    ]
    return float(max(scored)[2])
