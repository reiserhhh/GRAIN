#!/usr/bin/env python3
"""Complete generated factorials and observation-specific mixed responses."""

from __future__ import annotations

import itertools

import numpy as np

from .math_semantic_anchor import tensor_features

STYLES = ("neutral", "newsletter", "spelling")
TARGETS = ("extraversion", "neuroticism")
FAMILIES = ("named", "described", "anchored")
SEEDS = (11, 23, 37)


def joint_materials(manifest):
    """Cross every historical assignment with style; use no generated outcomes."""
    roles = {
        task: role for role, tasks in manifest["split_roles"].items() for task in tasks
    }
    rows = []
    for task, user in manifest["situations"]:
        for seed in SEEDS:
            for style in STYLES:
                batch = f"{task}|s{seed}|{style}"
                settings = [
                    (axis, family, pole)
                    for axis, family, pole in itertools.product(
                        TARGETS, FAMILIES, (-1, 1)
                    )
                ] + [("none", "none", 0)]
                for position, (axis, family, pole) in enumerate(settings):
                    if pole:
                        persona = manifest["aligned_pairs"][f"{axis}:{family}"][
                            "high" if pole == 1 else "low"
                        ]
                    else:
                        persona = manifest["neutral"]
                    addition = (
                        ""
                        if style == "neutral"
                        else " " + manifest["nonpersona"][style]
                    )
                    rows.append(
                        {
                            "text_id": f"{axis}|{family}|p{pole}|{batch}",
                            "axis": axis,
                            "family": family,
                            "pole": pole,
                            "style": style,
                            "situation": task,
                            "seed": seed,
                            "role": roles[task],
                            "batch": batch,
                            "batch_position": position,
                            "system": persona + addition + manifest["shared_suffix"],
                            "user": user,
                        }
                    )
    if len(rows) != 702 or len({r["text_id"] for r in rows}) != 702:
        raise ValueError("Complete original6-task design must yield702 unique texts")
    return rows


def interaction_design(rows):
    """Bind actual four corners, failing rather than inventing missing observations."""
    lookup = {}
    for i, row in enumerate(rows):
        key = tuple(
            row[k] for k in ("axis", "family", "situation", "seed", "style", "pole")
        )
        if key in lookup:
            raise ValueError("Duplicate joint observation")
        lookup[key] = i
    keys = sorted(
        {
            (r["axis"], r["family"], r["situation"], r["seed"])
            for r in rows
            if r["axis"] in TARGETS
        }
    )
    corners, metadata = [], []
    for axis, family, task, seed in keys:
        for style in STYLES[1:]:
            try:
                indices = [
                    lookup[axis, family, task, seed, s, p]
                    for s, p in (
                        (style, 1),
                        ("neutral", 1),
                        (style, -1),
                        ("neutral", -1),
                    )
                ]
            except KeyError as error:
                raise ValueError("Missing actual joint corner") from error
            corners.append(indices)
            metadata.append(
                {
                    "axis": axis,
                    "family": family,
                    "situation": task,
                    "seed": seed,
                    "style": style,
                    "role": rows[indices[0]]["role"],
                }
            )
    return np.asarray(corners, int), metadata


def mixed_difference(values, corners):
    """Compute observed h_s-h_0-l_s+l_0 in arbitrary finite vector units."""
    x, c = np.asarray(values, float), np.asarray(corners, int)
    if x.ndim < 2 or c.ndim != 2 or c.shape[1] != 4:
        raise ValueError("Vector observations and four actual corners required")
    if not np.isfinite(x).all() or np.any(c < 0) or np.any(c >= len(x)):
        raise ValueError("Finite observations with valid addresses required")
    return np.einsum("j,nj...->n...", np.array([1, -1, -1, 1]), x[c])


def tensor_accounts(z, corners):
    """Separate feature curvature and actual input nonadditivity at both anchors."""
    values = np.asarray(z, float)
    h, h0, low, low0 = np.moveaxis(values[corners], 1, 0)
    observed = mixed_difference(tensor_features(values), corners)
    upper = tensor_features(h0 + low - low0)
    lower = tensor_features(h - h0 + low0)
    upper_link = (
        upper - tensor_features(h0) - tensor_features(low) + tensor_features(low0)
    )
    upper_action = tensor_features(h) - upper
    lower_link = (
        tensor_features(h) - tensor_features(h0) - lower + tensor_features(low0)
    )
    lower_action = lower - tensor_features(low)
    return {
        "observed": observed,
        "upper_link": upper_link,
        "upper_action": upper_action,
        "lower_link": lower_link,
        "lower_action": lower_action,
    }


def seed_support(values):
    """Keep signed cross-replicate support distinct from observed squared energy."""
    x = np.asarray(values, float)
    if x.ndim != 2 or len(x) != 3 or not np.isfinite(x).all():
        raise ValueError("Exactly three finite independent seed rows required")
    mean = x.mean(axis=0)
    cross = float(((x @ x.T).sum() - np.sum(x * x)) / 6)
    spread = float(np.sum((x - mean) ** 2) / 6)
    return {
        "cross_seed_support": cross,
        "mean_squared_norm": float(mean @ mean),
        "within_seed_term": spread,
        "identity_error": abs(float(mean @ mean) - cross - spread),
        "mean_rms": float(np.sqrt(np.mean(mean**2))),
        "observed_rms": float(np.sqrt(np.mean(x**2))),
    }
