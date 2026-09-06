#!/usr/bin/env python3
"""Shared raw windows and descriptive geometry/function association estimators."""

import numpy as np
from scipy.stats import rankdata


def shared_windows(text, tokenizers):
    """Preserve every character while fitting every frozen tokenizer's512 budget."""
    if not text:
        raise ValueError("Empty input")

    def counts(piece):
        return {
            name: len(tok(prefix + piece, truncation=False, verbose=False)["input_ids"])
            for name, (tok, prefix) in tokenizers.items()
        }

    rows = []
    start = 0
    while start < len(text):
        end = len(text)
        if max(counts(text[start:end]).values()) > 512:
            low, high = 0, end - start
            while low + 1 < high:
                mid = (low + high) // 2
                if max(counts(text[start : start + mid]).values()) <= 512:
                    low = mid
                else:
                    high = mid
            if low == 0:
                raise ValueError("One complete character does not fit all observers")
            end = start + low
        sizes = counts(text[start:end])
        if max(sizes.values()) > 512:
            raise ValueError("Nonfeasible shared span")
        rows.append({"start": start, "end": end, "tokens": sizes})
        start = end
    if "".join(text[r["start"] : r["end"]] for r in rows) != text:
        raise ValueError("Shared windows lost source material")
    return rows


def correlation(x, y):
    """Finite Pearson association or NaN for an undefined constant input."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.clip(np.dot(x, y) / norm, -1, 1)) if norm > 1e-14 else float("nan")


def association(x, y, length):
    """Descriptive raw/rank/OLS-residual associations, not causal decompositions."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    length = np.asarray(length, dtype=float)
    if (
        not (
            np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(length).all()
        )
        or min(len(x), len(y), len(length)) < 4
        or len(x) != len(y)
        or len(x) != len(length)
    ):
        raise ValueError("Require matched finite case arrays")
    z = np.column_stack([np.ones(len(x)), length])
    rx = x - z @ np.linalg.lstsq(z, x, rcond=None)[0]
    ry = y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
    return np.array(
        [correlation(x, y), correlation(rankdata(x), rankdata(y)), correlation(rx, ry)]
    )


def association_bootstrap(x, y, length):
    """Resample fixed cases jointly; flag undefined draws instead of replacing them."""
    x = np.asarray(x)
    y = np.asarray(y)
    length = np.asarray(length)
    point = association(x, y, length)
    rng = np.random.default_rng(731)
    bootstrap = []
    for _ in range(1000):
        index = rng.integers(len(x), size=len(x))
        bootstrap.append(association(x[index], y[index], length[index]))
    bootstrap = np.array(bootstrap)
    result = {}
    for j, name in enumerate(("pearson", "spearman", "partial_length")):
        valid = bootstrap[:, j][np.isfinite(bootstrap[:, j])]
        low, high = (
            np.quantile(valid, [0.025, 0.975]) if len(valid) else (np.nan, np.nan)
        )
        result.update(
            {
                name: float(point[j]),
                f"{name}_low": float(low),
                f"{name}_high": float(high),
                f"{name}_undefined_draws": int(1000 - len(valid)),
            }
        )
    return result
