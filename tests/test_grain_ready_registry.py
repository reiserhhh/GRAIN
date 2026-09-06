#!/usr/bin/env python3
"""Same-ready population identity, source isolation and label-free feasibility."""

import copy

import numpy as np
import pytest

from grain_research.vector_mechanisms.ready_registry import (
    TASKS,
    membership_feasibility,
    registry_arrays,
    source_indices,
)


def fixture():
    """Create distinct hypothetical authors; no questionnaire values are present."""
    source = []
    for group, n in (("big5_only", 80), ("mbti_only", 80), ("bridge", 20)):
        for j in range(n):
            uid = f"{group}_{j:03d}"
            source.append(
                {
                    "author_id": uid,
                    "source_group": group,
                    "official_folds": {t: j % 5 for t in TASKS},
                    "views": [
                        [{"thread_id": uid, "body": "synthetic common material"}]
                    ],
                }
            )
    n = len(source)
    profiles = {"author_ids": np.asarray([b["author_id"] for b in source])}
    for arm in ("raw", "cue_mask", "placebo_mask"):
        profiles[arm + "_status"] = np.asarray(
            ["GEOMETRY_PROFILE_READY"] * n, dtype="U64"
        )
        profiles[arm + "_sorted_distances"] = np.tile(
            np.arange(16, dtype=float), (n, 1)
        )
    profiles["cue_mask_status"][0] = "OUTSIDE_REFERENCE_ENVELOPE"
    profiles["raw_ready"] = np.ones(n, dtype=bool)
    profiles["all_arms_ready"] = np.arange(n) != 0
    return source, profiles


def test_masks_are_same_authors_not_same_counts():
    source, profiles = fixture()
    reference = {source[80]["author_id"]}
    r = registry_arrays(source, profiles, reference)
    assert r["raw_ready"].sum() == 180 and r["all_arms_ready"].sum() == 179
    assert r["raw_ready_reference_free"].sum() == 179
    assert r["all_arms_ready_reference_free"].sum() == 178
    for task in TASKS:
        ix = source_indices(r, "raw_ready", task)
        assert len(ix) == 80 and not np.any(r["source_group"][ix] == "bridge")
        assert set(r[task + "_fold"][ix]) == set(range(5))
    swapped = copy.deepcopy(profiles)
    swapped["author_ids"][[0, 1]] = swapped["author_ids"][[1, 0]]
    with pytest.raises(ValueError, match="author order"):
        registry_arrays(source, swapped, reference)
    wrong = copy.deepcopy(profiles)
    wrong["all_arms_ready"][[0, 1]] = wrong["all_arms_ready"][[1, 0]]
    with pytest.raises(ValueError, match="actual profile statuses"):
        registry_arrays(source, wrong, reference)


def test_wrong_mask_type_fold_and_nonfinite_profile_rejected():
    source, profiles = fixture()
    for key, value in (
        ("raw_ready", profiles["raw_ready"].astype(int)),
        ("raw_sorted_distances", np.full((180, 16), np.nan)),
    ):
        with pytest.raises(ValueError):
            registry_arrays(source, profiles | {key: value}, set())
    source[0]["official_folds"]["big5"] = 5
    with pytest.raises(ValueError, match="official source fold"):
        registry_arrays(source, profiles, set())


def test_shared_threads_are_counted_including_nested_and_thin_cells():
    source, profiles = fixture()
    registry = registry_arrays(source, profiles, set())
    rows = membership_feasibility(source, registry, "raw_ready", "EI_cont")
    assert len(rows) == 42 and all(r["dense_min20_supported"] for r in rows)
    assert all(r["labels_read"] is False for r in rows)
    for b in source:
        b["views"][0][0]["thread_id"] = "shared_by_everyone"
    thin = membership_feasibility(source, registry, "raw_ready", "EI_cont")
    assert sum(not r["dense_min20_supported"] for r in thin) == 6
    assert all(r["train_retained"] == 0 for r in thin if ":inner:" not in r["scope"])


def test_independent_membership_and_splitter_replay():
    from sklearn.model_selection import KFold

    from scripts.audit_grain_same_ready_registry import (
        independent_kfold,
        reconstruct_membership,
    )

    for n in (20, 21, 22, 79):
        for actual, expected in zip(
            independent_kfold(n),
            KFold(3, shuffle=True, random_state=731).split(range(n)),
            strict=True,
        ):
            for a, e in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(a, e)
    source, profiles = fixture()
    for i, b in enumerate(source):
        if i % 11 == 0:
            b["views"][0][0]["thread_id"] = "shared_sparse"
    reg = registry_arrays(source, profiles, set())
    a = membership_feasibility(source, reg, "all_arms_ready", "big5")
    b = reconstruct_membership(source, reg["all_arms_ready"], "big5")
    assert len(a) == len(b) == 42
    for actual, expected in zip(a, b, strict=True):
        assert all(actual[k] == v for k, v in expected.items())
