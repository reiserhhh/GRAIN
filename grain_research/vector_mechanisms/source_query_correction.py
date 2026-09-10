#!/usr/bin/env python3
"""Paired source-query adjoints with an explicit common/difference split."""

import numpy as np

from .compiled_query_transfer import QUERIES
from .source_common_counteraction import compile_readers, select_contexts


def query_readers(bank, cohort, index, shared, weights):
    """Compile separate query averages from the same allowed source author set."""
    _, members = select_contexts(bank, cohort, index)
    contexts = [
        bank["x"][np.isin(bank["index"], members) & (bank["query"] == query)]
        for query in QUERIES
    ]
    return (
        np.stack([compile_readers(x, shared, weights) for x in contexts]),
        contexts,
        members,
    )


def apply_correction(v, d):
    """Read current d only; exact symmetric split of fixed query-specific actions."""
    v, d = np.asarray(v, float), np.asarray(d, float)
    if v.ndim != 3 or v.shape[:2] != (2, 2) or d.shape != (2, 4, v.shape[-1]):
        raise ValueError("Two query/mode readers and four incoming directions required")
    if not np.isfinite(v).all() or not np.isfinite(d).all():
        raise ValueError("Finite reader/direction values required")
    common_v = v.mean(0)
    diff_v = v[0] - v[1]
    common = np.einsum("mh,fh->mf", common_v, d[0] - d[1])
    correction = np.einsum("mh,fh->mf", diff_v, d.mean(0))
    actions = np.einsum("qmh,qfh->qmf", v, d)
    return {
        "common_v": common_v,
        "difference_v": diff_v,
        "common": common,
        "correction": correction,
        "contrast": common + correction,
        "actions": actions,
    }


def error_account(predicted_common, predicted_operator, carried, operator, native):
    """Retain source, query and numeric errors, and the exact MSE-change components."""
    p, q, c, o, n = [
        np.asarray(x, float)
        for x in (predicted_common, predicted_operator, carried, operator, native)
    ]
    if (
        not (p.shape == q.shape == c.shape == o.shape == n.shape)
        or not np.isfinite([p, q, c, o, n]).all()
    ):
        raise ValueError("Matched finite components required")
    return {
        "source_error": p - c,
        "query_error": q - o,
        "parity_error": c + o - n,
        "old_error": p - n,
        "new_error": p + q - n,
        "mse_change_cross": 2 * (p - n) * q,
        "mse_change_self": q * q,
    }
