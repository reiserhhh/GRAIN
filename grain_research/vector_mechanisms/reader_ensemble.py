#!/usr/bin/env python3
"""Equal frozen reader ensemble in field space, preserving component gauges."""

from __future__ import annotations

import numpy as np

from grain_research.vector_mechanisms.query_residual_input import predict_extension


def pack_models(models):
    """Pack exactly four numerical readers without fitting or averaging bases."""
    if len(models) != 4:
        raise ValueError("Exactly four frozen source-fold models required")
    packed = {"component_count": np.asarray(4)}
    for i, model in enumerate(models):
        for key, value in model.items():
            value = np.asarray(value)
            if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                raise ValueError("Finite numerical model assets required")
            packed[f"c{i}__{key}"] = value
    packed["ranks"] = np.asarray([len(m["basis"]) for m in models], int)
    return packed


def unpack_models(packed):
    """Recover all components with their own centers, scales and output bases."""
    if int(packed["component_count"]) != 4 or np.asarray(packed["ranks"]).shape != (4,):
        raise ValueError("Invalid four-component ensemble")
    models = []
    for i in range(4):
        prefix = f"c{i}__"
        model = {k[len(prefix) :]: v for k, v in packed.items() if k.startswith(prefix)}
        if model["basis"].ndim != 2 or len(model["basis"]) != packed["ranks"][i]:
            raise ValueError("Component rank disagrees with packed layout")
        models.append(model)
    return models


def predict_ensemble_coefficients(h, g, packed):
    """Return concatenated coefficients, never average unlike coordinates."""
    return np.concatenate(
        [predict_extension(h, g, m) for m in unpack_models(packed)], axis=1
    )


def ensemble_field(coefficients, packed):
    """Reconstruct each density field, divide by token count, then average."""
    coefficients = np.asarray(coefficients, float)
    if (
        coefficients.ndim != 2
        or len(coefficients) < 2
        or coefficients.shape[1] != sum(packed["ranks"])
        or not np.isfinite(coefficients).all()
    ):
        raise ValueError("Finite aligned coefficient stacks required")
    models = unpack_models(packed)
    field = np.zeros((len(coefficients), models[0]["basis"].shape[1]))
    offset = 0
    for model in models:
        rank = len(model["basis"])
        field += coefficients[:, offset : offset + rank] @ model["basis"]
        offset += rank
    return field / (4 * len(coefficients))
