#!/usr/bin/env python3
"""Exact level/departure decomposition in the frozen tensor observation space."""

from __future__ import annotations

from itertools import product

import numpy as np

from . import math_configuration_kernel as K


def component_grams(full_full, mean_full, full_mean, mean_mean):
    """Return kernels for phi(mean) and phi(parcel)-phi(mean) in each view."""
    if not all(len(g) == 3 for g in (full_full, mean_full, full_mean, mean_mean)):
        raise ValueError("three matched kernels per substitution required")
    out = []
    for ff, mf, fm, mm in zip(full_full, mean_full, full_mean, mean_mean, strict=True):
        ff, mf, fm, mm = map(K.finite_matrix, (ff, mf, fm, mm))
        if len({g.shape for g in (ff, mf, fm, mm)}) != 1:
            raise ValueError("all substitutions must keep source/application addresses")
        out.append(
            {(0, 0): mm, (0, 1): mf - mm, (1, 0): fm - mm, (1, 1): ff - mf - fm + mm}
        )
    return out


def tensor_blocks(components):
    """Contract all eight-by-eight signed components, with no positive clipping."""
    if len(components) != 3:
        raise ValueError("three view components required")
    bits = list(product((0, 1), repeat=3))
    matrix = np.empty((8, 8))
    for i, left in enumerate(bits):
        for j, right in enumerate(bits):
            matrix[i, j] = K.interaction_inner(
                [components[k][left[k], right[k]] for k in range(3)]
            )
    return matrix


def grouped_cells(full_full, mean_full, full_mean, mean_mean, w=None, v=None):
    """Retain the four signed contractions of level and total nonconstant mode."""
    ff = K.interaction_inner(full_full, w, v)
    mf = K.interaction_inner(mean_full, w, v)
    fm = K.interaction_inner(full_mean, w, v)
    mm = K.interaction_inner(mean_mean, w, v)
    return np.array([[mm, mf - mm], [fm - mm, ff - mf - fm + mm]])


def substitution_kernels(source, application, widths):
    """Evaluate original/level-clone kernels at fixed units, without model fitting."""
    s, a = np.asarray(source, float), np.asarray(application, float)
    if s.ndim != 3 or a.ndim != 3 or s.shape[1] != 3 or a.shape[1:] != s.shape[1:]:
        raise ValueError("aligned three-parcel observations required")
    ms, ma = s.mean(1), a.mean(1)
    ff, mf, fm, mm = [], [], [], []
    for k, width in enumerate(widths):
        ff.append(K.rbf(s[:, k], a[:, k], width))
        mf.append(K.rbf(ms, a[:, k], width))
        fm.append(K.rbf(s[:, k], ma, width))
        mm.append(K.rbf(ms, ma, width))
    return ff, mf, fm, mm
