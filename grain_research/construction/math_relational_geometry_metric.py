#!/usr/bin/env python3
"""Preserved common/specific Gram blocks and direction-sensitive deformation."""

import numpy as np


def decompose_gram(gram):
    """Retain all four orthogonal-design blocks, rather than deleting a common mode."""
    g = np.asarray(gram, float)
    if g.ndim != 2 or g.shape[0] != g.shape[1] or not np.isfinite(g).all():
        raise ValueError("Finite square Gram required")
    u = np.ones(len(g)) / np.sqrt(len(g))
    common = np.outer(u, u)
    contrast = np.eye(len(g)) - common
    blocks = {
        "common": common @ g @ common,
        "specific": contrast @ g @ contrast,
        "common_specific": common @ g @ contrast,
        "specific_common": contrast @ g @ common,
    }
    share = float(u @ g @ u / np.trace(g))
    return blocks, share


def metric_deformation(source, target):
    """Return all generalized directional gains without selecting a scoring metric."""
    gx, gy = np.asarray(source, float), np.asarray(target, float)
    if gx.ndim != 2 or gx.shape != gy.shape or gx.shape[0] != gx.shape[1]:
        raise ValueError("Equal square source/target Gram required")
    s, v = np.linalg.eigh(gx)
    if (
        not np.isfinite(gx).all()
        or not np.isfinite(gy).all()
        or min(s) <= np.finfo(float).eps * max(gx.shape) * max(s)
    ):
        raise ValueError("Identifiable finite source metric required")
    inverse_root = (v / np.sqrt(s)) @ v.T
    pulled_back = inverse_root @ gy @ inverse_root
    values, vectors = np.linalg.eigh((pulled_back + pulled_back.T) / 2)
    directions = inverse_root @ vectors
    if min(values) < -1e-10:
        raise ValueError("Target relation metric must be positive semidefinite")
    return {
        "pulled_back_metric": pulled_back,
        "directional_energy_gains": values,
        "coefficient_directions": directions,
    }
