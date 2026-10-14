#!/usr/bin/env python3
"""Separate addressed author evidence from a conditional observer background."""

import numpy as np
from scipy.special import logsumexp

from grain_research.vector_mechanisms import direct_traits as D

MODES = ("alone", "duplicate", "other")


def render(texts, task, background=None):
    """Keep the original target block intact; prefix only quoted background."""
    if len(texts) != 12:
        raise ValueError("The fixed twelve-event target history is required")
    target, _ = D.render(texts, task)
    if background is None:
        return target
    if len(background) != 12 or any(
        not isinstance(t, str) or not t for t in background
    ):
        raise ValueError("A complete twelve-event background is required")
    prefix = (
        "Additional quoted background for context only. Do not assess this block.\n\n"
    )
    prefix += "\n\n".join(
        f"Background quotation {i + 1}:\n{body}" for i, body in enumerate(background)
    )
    return prefix + "\nEnd of background quotations.\n\n" + target


def partners(n=24):
    """Fix a cyclic, outcome-free partner assignment; retain shared endpoints."""
    if n != 24:
        raise ValueError("Do not shrink or expand the registered author panel")
    return (np.arange(n) + 1) % n


def readouts(logits, normalizer, greedy, alphabet):
    """Preserve the ten-candidate law, high/low odds and vocabulary compliance."""
    z, n = np.asarray(logits, float), np.asarray(normalizer, float)
    g, a = np.asarray(greedy), np.asarray(alphabet)
    if (
        z.shape[-1:] != (10,)
        or z.shape[:-1] != n.shape
        or g.shape != n.shape
        or a.shape != (10,)
        or a.dtype.kind not in "iu"
        or len(set(a.tolist())) != 10
        or not np.isfinite(z).all()
        or not np.isfinite(n).all()
    ):
        raise ValueError("Finite complete digit observations are required")
    local = logsumexp(z, axis=-1)
    log_p = z - local[..., None]
    p = np.exp(log_p)
    log_mass = local - n
    if np.any(log_mass > 1e-5):
        raise ValueError("Candidate normalizer cannot exceed the vocabulary normalizer")
    return {
        "probabilities": p,
        "score": p @ (np.arange(10) / 9),
        "log_odds": logsumexp(z[..., 5:], axis=-1) - logsumexp(z[..., :5], axis=-1),
        "candidate_log_mass": log_mass,
        "candidate_mass": np.exp(log_mass),
        "greedy_compliant": np.isin(g, a),
    }


def accounts(values):
    """Retain baseline, duplicate increment and other-minus-duplicate increment."""
    x = np.asarray(values, float)
    if x.ndim != 3 or x.shape[1:] != (3, 5) or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Finite author x three modes x five targets required")
    pieces = np.stack((x[:, 0], x[:, 1] - x[:, 0], x[:, 2] - x[:, 1]), axis=1)
    centered = pieces - pieces.mean(axis=0)
    cov = np.einsum("niq,njr->ijqr", centered, centered) / len(x)
    total = x[:, 2] - x[:, 2].mean(axis=0)
    return {
        "pieces": pieces,
        "cross_covariance": cov,
        "other_covariance": total.T @ total / len(x),
        "closure_max": float(np.max(abs(pieces.sum(axis=1) - x[:, 2]))),
    }


def geometry(values):
    """Describe centered finite-panel target geometry, not psychological accuracy."""
    x = np.asarray(values, float)
    if x.ndim != 2 or x.shape[1] != 5 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Finite author x five-target positions required")
    c = x - x.mean(axis=0)
    gram = c.T @ c / len(x)
    norms = np.sqrt(np.diag(gram))
    denom = np.outer(norms, norms)
    cosine = np.divide(gram, denom, out=np.full_like(gram, np.nan), where=denom > 0)
    eigen = np.linalg.eigvalsh(gram)
    positive = eigen[eigen > 0]
    ratio = positive / positive.sum() if len(positive) else np.array([])
    return {
        "gram": gram,
        "cosine": cosine,
        "rms": norms,
        "first_squared_singular_ratio": float(ratio[-1]) if len(ratio) else None,
        "effective_rank": float(np.exp(-np.sum(ratio * np.log(ratio))))
        if len(ratio)
        else None,
    }


def bootstrap_en(values, draws, seed):
    """Resample whole author rows conditionally on fixed cyclic backgrounds.

    These are descriptive finite-panel resampling bands. Background authors
    recur across rows, so this is not an independent-author population CI.
    """
    x = np.asarray(values, float)
    accounts(x)
    if draws < 1:
        raise ValueError("At least one registered resampling draw is required")
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(draws):
        y = x[rng.integers(0, len(x), len(x))]
        contacts = np.array([geometry(y[:, m])["cosine"][0, 1] for m in range(3)])
        rows.append([*contacts, contacts[1] - contacts[0], contacts[2] - contacts[1]])
    return np.asarray(rows)
