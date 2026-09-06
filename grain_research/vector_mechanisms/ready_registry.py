#!/usr/bin/env python3
"""Outcome-blind common eligibility and exact official-fold membership for VM05."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from .controls import material_hash
from .external_controls import train_selection
from .external_profiles import ready_cohorts

TASKS = ("big5", "EI_cont", "SN_cont", "TF_cont", "JP_cont")
ARMS = ("raw", "cue_mask", "placebo_mask")


def registry_arrays(source, profiles, reference_members):
    """Derive masks from actual formal support without reading a single outcome."""
    ids = np.asarray([b["author_id"] for b in source])
    if not len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Common material needs unique authors")
    if not np.array_equal(ids, profiles["author_ids"]):
        raise ValueError("Profile author order differs from common material")
    groups = np.asarray([b["source_group"] for b in source])
    if not set(groups).issubset({"big5_only", "mbti_only", "bridge"}):
        raise ValueError("Unknown source group")
    masks = ready_cohorts({arm: profiles[f"{arm}_status"] for arm in ARMS})
    for key, value in masks.items():
        stored = profiles[key]
        if stored.dtype != np.dtype(bool) or not np.array_equal(stored, value):
            raise ValueError("Saved support mask differs from actual profile statuses")
    for arm in ARMS:
        g = np.asarray(profiles[f"{arm}_sorted_distances"])
        ready = profiles[f"{arm}_status"] == "GEOMETRY_PROFILE_READY"
        if g.ndim != 2 or len(g) != len(ids) or not np.isfinite(g[ready]).all():
            raise ValueError("Nonfinite or malformed accepted profile")
    if not set(reference_members).issubset(set(ids)):
        raise ValueError("Reference membership includes non-common authors")
    exposed = np.asarray([u in reference_members for u in ids])
    result = {"author_ids": ids, "source_group": groups, "reference_member": exposed}
    for name, mask in masks.items():
        result[name] = mask
        result[name + "_reference_free"] = mask & ~exposed
    for task in TASKS:
        group = "big5_only" if task == "big5" else "mbti_only"
        values = []
        for b in source:
            fold = -1
            if b["source_group"] == group:
                fold = b["official_folds"][task]
                if (
                    isinstance(fold, bool)
                    or not isinstance(fold, int)
                    or fold not in range(5)
                ):
                    raise ValueError("Invalid official source fold")
            values.append(fold)
        result[task + "_fold"] = np.asarray(values, dtype=int)
    return result


def source_indices(registry, cohort, task):
    """Return only permitted source authors, never bridge rows or a different arm set."""
    if (
        cohort
        not in (
            "raw_ready",
            "all_arms_ready",
            "raw_ready_reference_free",
            "all_arms_ready_reference_free",
        )
        or task not in TASKS
    ):
        raise ValueError("Unknown cohort/task")
    mask = registry[cohort]
    if mask.dtype != np.dtype(bool):
        raise ValueError("Eligibility must be a Boolean mask")
    group = "big5_only" if task == "big5" else "mbti_only"
    indices = np.flatnonzero(mask & (registry["source_group"] == group))
    if set(registry[task + "_fold"][indices]) != set(range(5)):
        raise ValueError("Eligible source cohort lacks an official fold")
    return indices


def membership_feasibility(source, registry, cohort, task):
    """Enumerate original/strict outer and nested purge counts without labels."""
    indices = source_indices(registry, cohort, task)
    bundles = [source[i] for i in indices]
    folds = registry[task + "_fold"][indices]
    rows = []

    def inspect(train, test, scope):
        try:
            keep, info = train_selection(train, test, "purge", scope=scope)
        except ValueError as error:
            if str(error) != "Insufficient authors after discussion control":
                raise
            threads = {c["thread_id"] for b in test for v in b["views"] for c in v}
            keep = np.asarray(
                [
                    i
                    for i, b in enumerate(train)
                    if not any(c["thread_id"] in threads for v in b["views"] for c in v)
                ],
                dtype=int,
            )
            info = {
                "train_available": len(train),
                "test_authors": len(test),
                "train_retained": len(keep),
                "train_author_commitment": material_hash(
                    [train[i]["author_id"] for i in keep]
                ),
                "test_author_commitment": material_hash([b["author_id"] for b in test]),
            }
        rows.append(
            {
                "cohort": cohort,
                "task": task,
                "scope": scope,
                **info,
                "dense_min20_supported": len(keep) >= 20,
                "labels_read": False,
            }
        )
        return [train[i] for i in keep]

    partitions = [
        (
            f"outer:{f}",
            [bundles[i] for i in np.flatnonzero(folds != f)],
            [bundles[i] for i in np.flatnonzero(folds == f)],
        )
        for f in range(5)
    ]
    bridge = [
        b
        for i, b in enumerate(source)
        if b["source_group"] == "bridge" and registry[cohort][i]
    ]
    if bridge:
        partitions.append(("bridge_source_fit", bundles, bridge))
    for scope, train, test in partitions:
        purged = inspect(train, test, scope)
        for branch, pool in (("retain", train), ("purge", purged)):
            if len(pool) < 3:
                continue
            for j, (ti, vi) in enumerate(
                KFold(3, shuffle=True, random_state=731).split(pool)
            ):
                if branch == "purge":
                    inspect(
                        [pool[i] for i in ti],
                        [pool[i] for i in vi],
                        f"{scope}:{branch}:inner:{j}",
                    )
                else:
                    rows.append(
                        {
                            "cohort": cohort,
                            "task": task,
                            "scope": f"{scope}:retain:inner:{j}",
                            "train_available": len(ti),
                            "test_authors": len(vi),
                            "train_retained": len(ti),
                            "dense_min20_supported": len(ti) >= 20,
                            "labels_read": False,
                            "train_author_commitment": material_hash(
                                [pool[i]["author_id"] for i in ti]
                            ),
                            "test_author_commitment": material_hash(
                                [pool[i]["author_id"] for i in vi]
                            ),
                        }
                    )
    return rows
