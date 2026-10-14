#!/usr/bin/env python3
"""Separate shared linear links from author distribution and relation remnants."""

from __future__ import annotations

import itertools

import numpy as np

from grain_research.construction import math_author_coupling as C

ARMS = (
    "joint_location",
    "G_shape",
    "Q_shape",
    "joint_shape",
    "shared_link",
    "shared_shift_link",
    "coupling",
    "shifted_coupling",
    "residual_link",
    "shifted_residual_link",
)
METRICS = (*C.METRICS, "location_shape_matched_strangers")
BLOCKS = (
    "G_shape",
    "Q_shape",
    "shared_link",
    "shared_shift_link",
    "residual_link",
    "shifted_residual_link",
)
WORLDS = (
    "shared_linear",
    "personal_rotation",
    "independent_rotation",
    "shared_nonlinear",
)


def within_covariances(statistics):
    """Keep within-panel shape separate from between-author mean variation."""
    s = statistics
    return (
        s["second_g"] - np.einsum("ni,nj->nij", s["mu_g"], s["mu_g"]),
        s["second_y"] - np.einsum("ni,nj->nij", s["mu_y"], s["mu_y"]),
    )


def components(statistics, reference):
    """Exact decomposition, with separate actual/shifted source common links."""
    vg, vy = within_covariances(statistics)
    shared = vg @ reference["link"]
    shifted_shared = vg @ reference["shift_link"]
    return {
        "G_shape": vg,
        "Q_shape": vy,
        "shared_link": shared,
        "shared_shift_link": shifted_shared,
        "residual_link": statistics["cross"] - shared,
        "shifted_residual_link": statistics["shifted"] - shifted_shared,
    }


def _whiten(block, name, reference):
    """Use global source units, not a person's noisy inverse covariance."""
    left = reference["base_root_y"] if name == "Q_shape" else reference["base_root_g"]
    right = reference["base_root_g"] if name == "G_shape" else reference["base_root_y"]
    return np.einsum("ab,nbc,cd->nad", left, block, right).reshape(len(block), -1)


def fit_reference(source):
    """Calibrate two fixed common links and shape units from source panels only."""
    base = C.fit_reference(source)
    vg, _ = within_covariances(source)
    within = vg.mean(0)
    _, eigen, tolerance = C.inverse_root(within)
    reference = {"base_" + key: value.copy() for key, value in base.items()}
    reference.update(
        {
            "within_g": within,
            "within_g_eigen": eigen,
            "within_g_tolerance": np.array(tolerance),
            "link": np.linalg.solve(within, source["cross"].mean(0)),
            "shift_link": np.linalg.solve(within, source["shifted"].mean(0)),
        }
    )
    for name, block in components(source, reference).items():
        center = block.mean(0)
        whitened = _whiten(block - center, name, reference)
        energy = float(np.sqrt(np.mean(np.sum(whitened**2, axis=1))))
        neutral = energy <= 1e-14
        reference[name + "_center"] = center
        reference[name + "_scale"] = np.array(1.0 if neutral else energy)
        reference[name + "_neutral"] = np.array(neutral)
    return reference


def base_reference(reference):
    """Recover exactly the HM64 source units without a second fitting rule."""
    return {
        key[5:]: value for key, value in reference.items() if key.startswith("base_")
    }


def profiles(statistics, reference):
    """Preserve marginal shapes, shared links and residuals as distinct channels."""
    inherited = C.profiles(statistics, base_reference(reference))
    result = {
        name: inherited[name]
        for name in ("joint_location", "coupling", "shifted_coupling")
    }
    for name, block in components(statistics, reference).items():
        result[name] = (
            _whiten(block - reference[name + "_center"], name, reference)
            / reference[name + "_scale"]
        )
    result["joint_shape"] = np.column_stack(
        (result["G_shape"], result["Q_shape"])
    ) / np.sqrt(2)
    return {name: result[name] for name in ARMS}


def centered_energy(statistics, reference):
    """Report signed cross energy instead of fictitious disjoint signal shares."""
    blocks = components(statistics, reference)
    total = statistics["cross"] - reference["base_cross"]
    shared = blocks["shared_link"] - reference["shared_link_center"]
    residual = blocks["residual_link"] - reference["residual_link_center"]
    t, s, r = [_whiten(x, "coupling", reference) for x in (total, shared, residual)]
    return np.column_stack(
        (np.sum(t * t, 1), np.sum(s * s, 1), np.sum(r * r, 1), 2 * np.sum(s * r, 1))
    )


def candidate_wins(distances, location_distances, shape_distances):
    """Use identical location and location-plus-shape negatives for every arm."""
    ordinary, location = C.candidate_wins(distances, location_distances)
    _, both = C.candidate_wins(distances, (location_distances + shape_distances) / 2)
    return ordinary, location, both


def synthetic_world(mode, seed, authors=40):
    """Generate actual new numeric panels with known common/personal mechanisms."""
    if mode not in WORLDS or authors < 8 or authors % 2:
        raise ValueError("Known world with even source/application authors required")
    rng = np.random.default_rng(seed)
    bits = np.array(list(itertools.product((-1.0, 1.0), repeat=4)))
    grid = np.array(list(itertools.product((-1.0, 0.0, 2.0), repeat=4)))
    material, statistics = {}, []
    for author in range(authors):
        scale = rng.uniform(0.5, 2.0, 4)
        personal = np.linalg.qr(rng.normal(size=(4, 4)))[0]
        for panel in range(3):
            g = grid * scale if mode == "shared_nonlinear" else bits.copy()
            if mode == "shared_linear":
                g *= scale
                y = g.copy()
            elif mode == "shared_nonlinear":
                y = g**2
            else:
                link = (
                    np.linalg.qr(rng.normal(size=(4, 4)))[0]
                    if mode == "independent_rotation"
                    else personal
                )
                y = g @ link
            order = rng.permutation(len(g))
            g, y = g[order], y[order]
            key = f"a{author:03d}_p{panel}"
            material[key + "__g"], material[key + "__y"] = g, y
            statistics.append(C.panel_statistics(g, y))
    return material, C.stack_statistics(statistics)
