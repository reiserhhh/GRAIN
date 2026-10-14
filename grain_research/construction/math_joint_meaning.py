#!/usr/bin/env python3
"""Separate common pair association from its assignment to target profiles."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from grain_research.construction import math_cross_window as W

VERSION = "HM177_joint_relation_target_assignment_v1"
MODELS = ("conditional_independence", "common_association", "observed_assignment")


def code_hash():
    """Bind the source construction, not a selected predictive probe."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def margins(t):
    """Return the three two-way marginals of a profile-by-state-by-state law."""
    return t.sum(2), t.sum(1), t.sum(0)


def homogeneous(t, tolerance=1e-10, max_sweeps=10000):
    """Find the entropy-maximizing law with CA,CB,AB margins by fixed IPF."""
    t = np.asarray(t, float)
    if t.ndim != 3 or not np.isfinite(t).all() or np.any(t < 0):
        raise ValueError("A nonnegative finite three-way law is required")
    ca, cb, ab = margins(t)
    support = (ca[:, :, None] > 0) & (cb[:, None, :] > 0) & (ab[None] > 0)
    x = support.astype(float)
    if t.sum() == 0:
        return x, {"sweeps": 0, "margin_error": 0.0, "active_cells": 0}
    x *= t.sum() / x.sum()
    for sweep in range(1, max_sweeps + 1):
        for axis, target in ((2, ca), (1, cb), (0, ab)):
            current = x.sum(axis)
            if np.any((target > 0) & (current == 0)):
                raise ValueError("Incompatible declared support")
            ratio = np.divide(
                target, current, out=np.zeros_like(target), where=current > 0
            )
            x *= np.expand_dims(ratio, axis)
        error = max(
            float(np.max(np.abs(a - b)))
            for a, b in zip(margins(x), (ca, cb, ab), strict=True)
        )
        if error <= tolerance:
            return x, {
                "sweeps": sweep,
                "margin_error": error,
                "active_cells": int(support.sum()),
            }
    raise ValueError(
        f"Fixed IPF budget exhausted after {max_sweeps} sweeps;error={error}"
    )


def profile_tables(p, y):
    """Average observed disjoint-window laws within each full SOURCE profile."""
    p, y, group, number = W.validate(p, y)
    profiles = np.unique(y, axis=0)
    full = np.array(
        [p[group == c, 0].T @ p[group == c, 1] / len(p) for c in range(len(number))]
    )
    return profiles, full


def construct(p, y, tolerance=1e-10, max_sweeps=10000):
    """Build three references while retaining choices and singleton meanings."""
    profiles, full = profile_tables(p, y)
    laws = np.zeros((3, *full.shape))
    laws[2] = full
    info = []
    for z0 in range(2):
        for z1 in range(2):
            s0, s1 = slice(4 * z0, 4 * z0 + 4), slice(4 * z1, 4 * z1 + 4)
            t = full[:, s0, s1]
            ca, cb, _ = margins(t)
            mass = t.sum((1, 2))
            product = ca[:, :, None] * cb[:, None, :]
            laws[0, :, s0, s1] = np.divide(
                product,
                mass[:, None, None],
                out=np.zeros_like(t),
                where=mass[:, None, None] > 0,
            )
            laws[1, :, s0, s1], receipt = homogeneous(t, tolerance, max_sweeps)
            info.append({"early_condition": z0, "late_condition": z1, **receipt})
    return profiles, laws, info


def target_laws(profiles, laws):
    """Project joint profile assignments to all five external target contrasts."""
    return np.array(
        [
            [[laws[k, profiles[:, q] == g].sum(0) for g in (0, 1)] for q in range(5)]
            for k in range(3)
        ]
    )


def fields(laws):
    """Keep unsupported pair fields missing instead of assigning a prior."""
    total = laws.sum(2)
    return np.divide(
        laws[:, :, 1], total, out=np.full_like(total, np.nan), where=total > 0
    )


def relation_statistics(profiles, laws):
    """One relation-assignment divergence and all five weighted target effects."""
    full, common = laws[2], laws[1]
    active = full > 0
    if np.any(common[active] <= 0):
        raise ValueError("Observed mass is outside common reference support")
    divergence = np.sum(full[active] * np.log(full[active] / common[active]))
    f = fields(target_laws(profiles, laws))
    weight = full.sum(0)
    used = weight > 0
    change = f[2, :, used] - f[1, :, used]
    return np.r_[divergence, np.sqrt(np.sum(weight[used, None] * change**2, axis=0))]


def matched_reference(p, y, repetitions, seed, tolerance, max_sweeps):
    """Exchange whole late author windows inside profiles; never expand subjects."""
    p, y, group, number = W.validate(p, y)
    profiles, laws, info = construct(p, y, tolerance, max_sweeps)
    observed = relation_statistics(profiles, laws)
    rng, null, max_iterations = np.random.default_rng(seed), [], 0
    for _ in range(repetitions):
        perm = np.arange(len(p))
        for c in range(len(number)):
            idx = np.flatnonzero(group == c)
            perm[idx] = rng.permutation(idx)
        mixed = np.stack([p[:, 0], p[perm, 1]], axis=1)
        c, candidate, receipt = construct(mixed, y, tolerance, max_sweeps)
        null.append(relation_statistics(c, candidate))
        max_iterations = max(max_iterations, max(r["sweeps"] for r in receipt))
    null = np.asarray(null)
    tail = (1 + np.sum(null >= observed, axis=0)) / (repetitions + 1)
    return observed, null, tail, {"observed": info, "null_max_sweeps": max_iterations}


def build_bundle(p, y, tolerance=1e-10, max_sweeps=10000):
    """Compile source-only target functions with one declared association model."""
    profiles, tables, info = construct(p, y, tolerance, max_sweeps)
    law = target_laws(profiles, tables)
    psi = fields(law)
    return {
        "version": VERSION,
        "code_hash": code_hash(),
        "models": list(MODELS),
        "source_window_mass": np.asarray(p).tolist(),
        "source_binary_targets": np.asarray(y, int).tolist(),
        "tolerance": tolerance,
        "max_sweeps": max_sweeps,
        "profiles": profiles.tolist(),
        "profile_tables": tables.tolist(),
        "solver_receipt": info,
        "joint_laws": law.tolist(),
        "pair_readout": np.where(np.isfinite(psi), psi, None).tolist(),
        "readout_supported": np.isfinite(psi).tolist(),
        "interpretation": "mean_two_event_target_readout;not_whole_history_posterior",
        "smoothing": 0,
        "application_outcomes": False,
    }


def validate_bundle(bundle):
    """Reject changed source tables, extra fields and altered solver specifications."""
    if bundle.get("version") != VERSION or bundle.get("code_hash") != code_hash():
        raise ValueError("Unknown joint-meaning implementation")
    expected = build_bundle(
        bundle["source_window_mass"],
        bundle["source_binary_targets"],
        bundle["tolerance"],
        bundle["max_sweeps"],
    )
    if expected != bundle:
        raise ValueError("Changed SOURCE construction or unexpected outcome field")
    return bundle


def cold_record(bundle, record):
    """Apply the frozen pair function, retaining the strict identified-event API."""
    return W.cold_record(bundle, record)
