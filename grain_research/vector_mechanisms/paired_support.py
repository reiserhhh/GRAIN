#!/usr/bin/env python3
"""Fixed input-by-support factorial with source-only reference coordinates."""

import numpy as np
from sklearn.linear_model import Ridge


def fit_support(blocks, targets, train, paired, penalty=0.01):
    """Fit on original or paired endpoints with identical source-original scales."""
    y = np.asarray(targets, float)
    idx = np.asarray(train, int)
    xs = [np.asarray(x, float) for x in blocks]
    if (
        len(xs) != 2
        or y.ndim != 3
        or y.shape[1] != 2
        or len(idx) == 0
        or len(np.unique(idx)) != len(idx)
        or idx.min() < 0
        or idx.max() >= len(y)
        or penalty <= 0
        or not np.isfinite(y).all()
        or any(x.ndim != 3 or x.shape[:2] != y.shape[:2] for x in xs)
        or any(not np.isfinite(x).all() for x in xs)
    ):
        raise ValueError(
            "Two finite input blocks and matched two-order targets required"
        )
    centers = [x[idx, 0].mean(0) for x in xs]
    scales = [
        max(float(np.sqrt(np.mean(np.sum((x[idx, 0] - c) ** 2, axis=-1)))), 1e-12)
        for x, c in zip(xs, centers, strict=True)
    ]
    model = {"centers": centers, "scales": scales}
    z = support_features(model, xs)[idx]
    train_x = z.reshape(-1, z.shape[-1]) if paired else z[:, 0]
    train_y = y[idx].reshape(-1, y.shape[-1]) if paired else y[idx, 0]
    model["head"] = Ridge(alpha=penalty * len(train_x)).fit(train_x, train_y)
    model["train_rows"] = len(train_x)
    return model


def support_features(model, blocks):
    """Apply frozen reference coordinates without fitting on an inference batch."""
    return np.concatenate(
        [
            (np.asarray(x) - c) / s
            for x, c, s in zip(blocks, model["centers"], model["scales"], strict=True)
        ],
        axis=-1,
    ) / np.sqrt(2)


def predict_support(model, blocks):
    """Return both endpoint scores from one fixed function, without edit labels."""
    x = support_features(model, blocks)
    return model["head"].predict(x.reshape(-1, x.shape[-1])).reshape(*x.shape[:2], -1)
