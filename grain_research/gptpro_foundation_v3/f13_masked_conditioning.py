#!/usr/bin/env python3
"""Batched Gaussian conditioning across actual source bootstrap fits."""

import numpy as np


def condition_source_fits(fits, x, y, kernel):
    """Factor a masked observation kernel once, preserving arbitrary time gaps."""
    x, y, kernel = np.asarray(x, float), np.asarray(y, float), np.asarray(kernel, float)
    if (
        x.ndim != 2
        or x.shape[1] != 2
        or y.shape != (len(x),)
        or kernel.shape != (len(x), len(x))
    ):
        raise ValueError("aligned masked observation arrays required")
    if not all(np.isfinite(a).all() for a in (x, y, kernel)):
        raise ValueError("finite observation arrays required")
    if not fits or any(f["status"] != "SUPPORTED" for f in fits):
        raise ValueError("unsupported source fits cannot be silently discarded")
    mean = np.asarray([f["mean"] for f in fits])
    prior = np.asarray([f["prior"] for f in fits])
    sigma = np.asarray([f["sigma2"] for f in fits])
    if not len(x):
        return mean, prior
    # Positive definite by declared AR kernel; do not use an IID kernel on
    # compressed indices, which would erase observed time gaps.
    chol = np.linalg.cholesky(kernel)
    wx = np.linalg.solve(chol, x)
    wy = np.linalg.solve(chol, y)
    precision = np.linalg.inv(prior)
    posterior = np.linalg.inv(
        precision + (wx.T @ wx)[None, :, :] / sigma[:, None, None]
    )
    rhs = (
        np.einsum("bij,bj->bi", precision, mean) + (wx.T @ wy)[None, :] / sigma[:, None]
    )
    centers = np.einsum("bij,bj->bi", posterior, rhs)
    return centers, posterior
