#!/usr/bin/env python3
"""Dual-space error witnesses; never promote a failed full-Jacobian packet."""

from __future__ import annotations

import numpy as np


def action_error_witness(delta_j, directions, c, jp, query_displacement):
    """Pair actual physical directions before bounding an observed replay error."""
    delta = np.asarray(delta_j, float).reshape(len(c), -1)
    ds = np.asarray(directions, float).reshape(len(directions), -1)
    p = np.asarray(jp, float).reshape(len(c), -1)
    e = np.asarray(query_displacement, float).ravel()
    if delta.shape[1] != ds.shape[1] or p.shape[1] != len(e):
        raise ValueError("aligned physical H and query units required")
    if not all(np.isfinite(x).all() for x in (delta, ds, c, p, e)):
        raise ValueError("finite error factors required")
    w = np.asarray(c, float).T @ (p @ e)
    errors = ds @ delta.T
    return {
        "signed_error": errors @ w,
        "directional_bound": np.linalg.norm(errors, axis=1) * np.linalg.norm(w),
        "frobenius_bound": np.linalg.norm(delta)
        * np.linalg.norm(w)
        * np.linalg.norm(ds, axis=1),
        "dual_weights": w,
    }


def dual_route_actions(jacobians, directions, c, jp, query_displacement):
    """Contract the output dual first, rather than constructing P covectors."""
    jj = np.asarray(jacobians, float)
    ds = np.asarray(directions, float).reshape(len(directions), -1)
    pp = np.asarray(jp, float).reshape(len(c), -1)
    e = np.asarray(query_displacement, float).ravel()
    w = np.asarray(c, float).T @ (pp @ e)
    return np.stack([ds @ (j.reshape(len(c), -1).T @ w) for j in jj])
