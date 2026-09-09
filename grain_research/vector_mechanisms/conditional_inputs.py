#!/usr/bin/env python3
"""Fixed early evidence summaries, excluding instruction/question scaffolds."""

import numpy as np


def evidence_mask(offsets, spans, question_mask):
    """Select tokens fully inside an excerpt; omit mixed boundary tokens."""
    q = np.asarray(question_mask, bool)
    if len(offsets) != len(q):
        raise ValueError("Token masks must match offsets")
    selected = np.asarray(
        [b > a and any(lo <= a < b <= hi for lo, hi in spans) for a, b in offsets]
    )
    if not selected.any() or (selected & q).any() or selected[-1]:
        raise ValueError("Evidence must be separate from query/final positions")
    return selected


def evidence_summaries(hidden, body_mask, question_mask):
    """Return mean and cosine-softmax query-weighted evidence means, temperature1."""
    h = np.asarray(hidden, np.float64)
    body, query = np.asarray(body_mask, bool), np.asarray(question_mask, bool)
    if h.ndim != 2 or body.shape != (len(h),) or query.shape != body.shape:
        raise ValueError("Token states and masks must agree")
    if (
        not np.isfinite(h).all()
        or not body.any()
        or not query.any()
        or (body & query).any()
    ):
        raise ValueError("Finite disjoint evidence/query states required")
    values, anchor = h[body], h[query].mean(0)
    score = (values @ anchor) / np.maximum(
        np.linalg.norm(values, axis=1) * np.linalg.norm(anchor), 1e-12
    )
    weights = np.exp(score - score.max())
    weights /= weights.sum()
    return np.stack([values.mean(0), weights @ values])
