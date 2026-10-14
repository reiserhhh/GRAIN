#!/usr/bin/env python3
"""One finite entropy-safe algebraic operation with exact lower margins."""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import numpy as np

from grain_research.construction import math_ternary_meaning as B

TARGETS = B.TARGETS
fields = B.fields
statistics = B.statistics
contact_action = B.contact_action
source_folds = B.source_folds
validate_bundle = B.validate_bundle


def entropy(t):
    """Use finite subprobability entropy, retaining exact source zeros."""
    x = np.asarray(t, float)
    active = x > 0
    return float(-np.sum(x[active] * np.log(x[active])))


def safe_step(values, signs):
    """A fixed curvature-bounded step, not a fitted or iterated entropy optimum."""
    x = np.asarray(values, float)
    if x.shape[-1] != 16 or not np.isfinite(x).all() or np.any(x <= 0):
        raise ValueError("All16 source-supported cube corners must be positive")
    signs = np.asarray(signs, float)
    if signs.shape != (16,) or not np.isin(signs, (-1, 1)).all():
        raise ValueError("Sixteen signed cube corners required")
    ell = np.log(x) @ signs
    tau = np.min(x, axis=-1) * ell / (32 + 2 * np.abs(ell))
    return tau, ell


def trade_reference(full, profiles):
    """Use every supported one-bit-profile cube once, without solving a global fit."""
    f, profiles = np.asarray(full, float), np.asarray(profiles, int)
    if profiles.ndim != 2 or profiles.shape[1] != 5:
        raise ValueError("Five-bit complete binary profiles required")
    if f.shape != (len(profiles), 8, 8, 8) or not np.isfinite(f).all() or np.any(f < 0):
        raise ValueError("Coherent eight-cell three-role SOURCE law required")
    lookup = {tuple(row): i for i, row in enumerate(profiles)}
    if (
        len(lookup) != len(profiles)
        or profiles.shape[1] != 5
        or not np.isin(profiles, (0, 1)).all()
    ):
        raise ValueError("Distinct complete binary profiles required")
    bits = np.array(list(itertools.product(range(2), repeat=4)))
    chi = (-1.0) ** bits.sum(1)
    pairs = list(itertools.combinations(range(4), 2))
    triples = np.array(list(itertools.product(pairs, repeat=3)))
    changes = np.zeros((5, *f.shape))
    incidence = np.zeros(f.shape, int)
    receipts, bound = [], 135
    for z in itertools.product(range(2), repeat=3):
        indices = [triples[:, r, bits[:, r + 1]] + 4 * z[r] for r in range(3)]
        a, b, c = indices
        counts, sum_tau_ell = np.zeros(5, int), 0.0
        for q in range(5):
            for g0, profile in enumerate(profiles):
                if profile[q] != 0:
                    continue
                other = profile.copy()
                other[q] = 1
                if tuple(other) not in lookup:
                    continue
                gs = np.broadcast_to(
                    np.array([g0, lookup[tuple(other)]])[bits[:, 0]], a.shape
                )
                values = f[gs, a, b, c]
                good = np.all(values > 0, axis=1)
                if not good.any():
                    continue
                tau, ell = safe_step(values[good], chi)
                where = (
                    gs[good].ravel(),
                    a[good].ravel(),
                    b[good].ravel(),
                    c[good].ravel(),
                )
                np.add.at(changes[q], where, (tau[:, None] * chi).ravel())
                np.add.at(incidence, where, 1)
                counts[q] += int(good.sum())
                sum_tau_ell += float(tau @ ell)
        key = (slice(None), *(slice(4 * r, 4 * r + 4) for r in z))
        h_block = f[key] - changes.sum(0)[key] / bound
        receipts.append(
            {
                "conditions": list(z),
                "method": "one_finite_entropy_safe_cube_step",
                "constructor_hash": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "ipf_used": False,
                "supported_cubes_by_target": counts.tolist(),
                "overlap_bound": bound,
                "observed_max_incidence": int(incidence[key].max()),
                "entropy_gain": entropy(h_block) - entropy(f[key]),
                "entropy_gain_lower_bound": sum_tau_ell / (2 * bound),
            }
        )
    h, change = f - changes.sum(0) / bound, changes / bound
    if (
        incidence.max() > bound
        or np.any(h < f / 2 - 1e-15)
        or not np.array_equal(h[f == 0], f[f == 0])
    ):
        raise ValueError("Finite probability/support bound failed")
    error = max(
        float(np.max(np.abs(a - b)))
        for a, b in zip(B.margins(h), B.margins(f), strict=True)
    )
    for receipt in receipts:
        receipt["margin_error"] = error
        if receipt["entropy_gain"] + 1e-14 < receipt["entropy_gain_lower_bound"]:
            raise ValueError("Analytic entropy increase bound failed")
    return h, change, receipts


def construct(p, y):
    """Construct the original author sampler plus one finite within-fiber action."""
    p, y = np.asarray(p, float), np.asarray(y)
    if (
        p.ndim != 3
        or p.shape[1:] != (3, 8)
        or y.shape != (len(p), 5)
        or not np.isin(y, (0, 1)).all()
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.allclose(p.sum(2), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError(
            "Author three-role probabilities and full binary profiles required"
        )
    profiles, group = np.unique(y, axis=0, return_inverse=True)
    full = np.array(
        [
            np.einsum(
                "ua,ub,uc->abc", p[group == g, 0], p[group == g, 1], p[group == g, 2]
            )
            / len(p)
            for g in range(len(profiles))
        ]
    )
    h, changes, receipts = trade_reference(full, profiles)
    return profiles, np.stack((h, full)), changes, receipts


def build_with_components(p, y, tolerance=1e-10, max_sweeps=10000):
    """Compile each view once and retain its signed target-specific operation."""
    del tolerance, max_sweeps  # Compatibility only; no numerical optimizer is called.
    p = np.asarray(p, float)
    if p.ndim != 4 or p.shape[1:] != (2, 3, 8):
        raise ValueError("Two fixed views of three roles required")
    tables, components, receipts = [], [], []
    for v in range(2):
        profiles, t, change, receipt = construct(p[:, v], y)
        tables.append(t)
        components.append(change)
        receipts.append(receipt)
    bundle = {
        "version": B.VERSION,
        "code_hash": B.code_hash(),
        "models": list(B.MODELS),
        "targets": list(TARGETS),
        "profiles": profiles.tolist(),
        "profile_tables": np.asarray(tables).tolist(),
        "solver_receipts": receipts,
        "denominator": 64,
        "role_size": 4,
        "constraints": ["GAB", "GAC", "GBC", "ABC"],
        "interpretation": "random_one_event_per_role;not_whole_history_posterior",
    }
    return bundle, np.asarray(components)


def build_bundle(p, y, tolerance=1e-10, max_sweeps=10000):
    """Reuse the strict cold schema; method/hash are explicit in each receipt."""
    return build_with_components(p, y, tolerance, max_sweeps)[0]
