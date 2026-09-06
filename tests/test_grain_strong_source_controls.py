#!/usr/bin/env python3
"""Strong baseline exposure/control metadata and complete frozen design checks."""

from collections import Counter

import pytest

from scripts.run_grain_strong_source_controls import cases, validate_case


def test_all_axes_arms_and_five_count_controls_are_retained():
    rows = cases()
    assert len(rows) == 84 and len({r["id"] for r in rows}) == 84
    assert set(Counter((r["task"], r["arm"]) for r in rows).values()) == {7}
    for row in rows:
        assert row["feature"] == "word_char" and row["exclude_reference"]
    assert Counter(r["mode"] for r in rows) == {
        "retain": 12,
        "purge": 12,
        "random_count": 60,
    }


def test_wrong_exposure_or_feature_cannot_be_reused_as_control():
    c = cases()[0]
    m = {
        "task": c["task"],
        "arm": c["arm"],
        "feature": "word_char",
        "source_group": "mbti_only",
        "common_material_sha256": "frozen",
        "exclude_reference": True,
        "thread_control": c["mode"],
        "drop_seed": c["seed"],
        "permutation_seed": None,
        "pca_applied": False,
        "bridge_labels_used": False,
    }
    validate_case(m, c, "frozen")
    for key, value in [
        ("feature", "grain_intermediate"),
        ("exclude_reference", False),
        ("bridge_labels_used", True),
        ("permutation_seed", 731),
    ]:
        with pytest.raises(ValueError):
            validate_case(m | {key: value}, c, "frozen")
    validate_case(m | {"exclude_reference": False}, c, "frozen", baseline=True)


def test_strong_audit_cannot_fall_back_to_weaker_legacy_head(tmp_path):
    from scripts.summarize_grain_vector_source_controls import reference_path

    record = {"task": "TF_cont", "arm": "raw", "path": str(tmp_path / "strong")}
    batch = {"protocol": "VM05_STRONG_SPARSE_SOURCE_CONTROLS_V1", "baselines": [record]}
    assert (
        reference_path(batch, tmp_path / "weaker", "TF_cont", "raw")
        == tmp_path / "strong"
    )
    with pytest.raises(ValueError):
        reference_path(batch, tmp_path / "weaker", "EI_cont", "raw")
    with pytest.raises(ValueError):
        reference_path(
            batch | {"baselines": [record, record]}, tmp_path, "TF_cont", "raw"
        )


def test_auditor_rejects_duplicate_or_substituted_conditions():
    from scripts.summarize_grain_vector_source_controls import verify_strong_design

    rows = cases()
    verify_strong_design({"cases": rows})
    for key, value in (
        ("feature", "dense"),
        ("seed", 999),
        ("exclude_reference", False),
    ):
        with pytest.raises(ValueError):
            verify_strong_design({"cases": [rows[0] | {key: value}, *rows[1:]]})
    with pytest.raises(ValueError):
        verify_strong_design({"cases": [rows[0], rows[0], *rows[2:]]})


def test_baseline_preflight_requires_every_packet_and_exact_users(
    tmp_path, monkeypatch
):
    import numpy as np

    from grain_research.vector_mechanisms.controls import material_hash
    from scripts import run_grain_strong_source_controls as runner

    bundles = [{"author_id": "synthetic", "source_group": "mbti_only"}]
    paths, packets = [], {}
    for task in ("EI_cont", "SN_cont", "TF_cont", "JP_cont"):
        for arm in ("raw", "cue_mask", "placebo_mask"):
            path = tmp_path / f"{task}_{arm}"
            path.mkdir()
            (path / "summary.json").write_text("{}")
            paths.append(path)
            packets[path] = {
                "manifest": {
                    "task": task,
                    "arm": arm,
                    "feature": "word_char",
                    "source_group": "mbti_only",
                    "common_material_sha256": "common",
                    "source_material_sha256": material_hash(bundles),
                    "exclude_reference": False,
                    "thread_control": "retain",
                    "drop_seed": 731,
                    "permutation_seed": None,
                    "pca_applied": False,
                    "bridge_labels_used": False,
                    "identity": path.name,
                },
                "summary": {"bridge_labels_used": False, "full_head_sha256": "test"},
                "ids": np.asarray(["synthetic"]),
                "y": np.asarray([1]),
                "folds": np.asarray([0]),
            }
    monkeypatch.setattr(runner, "load_oof", lambda path: packets[path])
    assert len(runner.baseline_index(paths, "common", bundles)) == 12
    with pytest.raises(ValueError, match="All12"):
        runner.baseline_index(paths[:-1], "common", bundles)
    with pytest.raises(ValueError, match="Duplicate"):
        runner.baseline_index([*paths, paths[0]], "common", bundles)
    packets[paths[-1]]["ids"] = np.asarray(["unexpected"])
    with pytest.raises(ValueError, match="exact complete source cohort"):
        runner.baseline_index(paths, "common", bundles)
