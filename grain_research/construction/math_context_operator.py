#!/usr/bin/env python3
"""Exact finite tensor-kernel action and context response, not a fitted new model."""

from __future__ import annotations

import numpy as np


def compile_tensor(model):
    """Expand a source-centered tensor kernel into additive and bilinear terms.

    The centering row mean also multiplies sum(dual); subtract it exactly rather
    than assuming that sum is numerically zero. All units remain source-fixed.
    """
    if str(model["kind"]) != "tensor":
        raise ValueError("only the frozen tensor kernel has this expansion")
    a, o, d = (np.asarray(model[k], float) for k in ("a", "b", "dual"))
    if a.ndim != 2 or o.ndim != 2 or len(a) != len(o) or d.shape != (len(a),):
        raise ValueError("source vectors and dual coefficients are not aligned")
    e = d - d.sum() / len(d)
    constant = float(
        e.sum()
        - model["column_mean"] @ d
        + model["grand_mean"] * d.sum()
        + model["bias"]
    )
    return {
        "source_action": a,
        "source_object": o,
        "effective_dual": e,
        "constant": constant,
        "action_linear": a.T @ e,
        "object_linear": o.T @ e,
    }


def score_compiled(operator, action, obj):
    """Evaluate the exact expansion without the original centered-kernel scorer."""
    action, obj = np.asarray(action, float), np.asarray(obj, float)
    if action.ndim != 2 or obj.ndim != 2 or len(action) != len(obj):
        raise ValueError("aligned two-dimensional action/object rows required")
    interaction = (
        (action @ operator["source_action"].T) * (obj @ operator["source_object"].T)
    ) @ operator["effective_dual"]
    return (
        operator["constant"]
        + action @ operator["action_linear"]
        + obj @ operator["object_linear"]
        + interaction
    )


def context_response(operator, action, before, after):
    """Separate additive object and action-conditioned responses to actual drift."""
    action, before, after = (np.asarray(x, float) for x in (action, before, after))
    if action.ndim != 2 or before.shape != after.shape or len(action) != len(before):
        raise ValueError("aligned action and before/after context vectors required")
    change = after - before
    additive = change @ operator["object_linear"]
    coupled = (
        (action @ operator["source_action"].T) * (change @ operator["source_object"].T)
    ) @ operator["effective_dual"]
    return additive, coupled, additive + coupled


def four_corner(operator, a0, a1, o0, o1):
    """Compute the exact mixed finite difference from vector displacements."""
    da, do = np.asarray(a1) - np.asarray(a0), np.asarray(o1) - np.asarray(o0)
    return (
        (da @ operator["source_action"].T) * (do @ operator["source_object"].T)
    ) @ operator["effective_dual"]
