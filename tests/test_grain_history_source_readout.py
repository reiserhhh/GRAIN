#!/usr/bin/env python3
"""History source boundary, complete-cache, readout and saved-estimator tests."""

import json
import zipfile
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from grain_research.vector_mechanisms import history_readout_cache as cache
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_material import pack_features
from grain_research.vector_mechanisms.history_collection import observation_signature
from grain_research.vector_mechanisms.ready_comparison import digest, verified_packet
from scripts import run_grain_history_source_readout as runner
from scripts.summarize_grain_vector_external_links import load_oof


def source_fixture():
    """Include source, bridge and disjoint dictionary people with no outcomes."""
    source = [
        {
            "author_id": f"synthetic_{i:03d}",
            "source_group": "big5_only"
            if i < 10
            else "mbti_only"
            if i < 25
            else "bridge",
            "official_folds": {t: i % 5 for t in cache.TASKS},
        }
        for i in range(30)
    ]
    ids = np.array([b["author_id"] for b in source])
    reference = np.array([20 <= i < 25 for i in range(30)])
    ready = ~reference
    ready[0] = False
    reg = {
        "author_ids": ids,
        "reference_member": reference,
        "all_arms_ready_reference_free": ready,
        **{t + "_fold": np.array([i % 5 for i in range(30)]) for t in cache.TASKS},
    }
    whole = [
        b | {"views": [{"donor_author_id": ids[20]} for _ in range(3)]}
        for i, b in enumerate(source)
        if not reference[i] and i != 19
    ]
    return source, whole, reg


def test_history_cohorts_exclude_donors_bridge_and_unsupported_profiles():
    source, whole, reg = source_fixture()
    ix = cache.source_indices(source, whole, reg, "history_full", "big5")
    assert ix.tolist() == list(range(10))
    assert cache.source_indices(
        source, whole, reg, "history_ready", "big5"
    ).tolist() == list(range(1, 10))
    assert cache.source_indices(
        source, whole, reg, "history_full", "EI_cont"
    ).tolist() == list(range(10, 19))
    reg["all_arms_ready_reference_free"] = reg["all_arms_ready_reference_free"].astype(
        int
    )
    with pytest.raises(ValueError, match="booleans"):
        cache.source_indices(source, whole, reg, "history_ready", "EI_cont")
    source, whole, reg = source_fixture()
    whole[0]["views"][0]["donor_author_id"] = source[1]["author_id"]
    with pytest.raises(ValueError, match="dictionary"):
        cache.source_indices(source, whole, reg, "history_full", "big5")
    source, whole, reg = source_fixture()
    whole[0]["official_folds"] = {**whole[0]["official_folds"], "big5": 4}
    with pytest.raises(ValueError, match="roles/folds"):
        cache.source_indices(source, whole, reg, "history_full", "big5")
    source, whole, reg = source_fixture()
    source[20]["source_group"] = "bridge"
    with pytest.raises(ValueError, match="MBTI-only"):
        cache.source_indices(source, whole, reg, "history_full", "big5")


def test_every_history_shard_is_required_even_for_small_requested_task(
    tmp_path, monkeypatch
):
    whole = [{"author_id": str(i)} for i in range(128)]
    m = {
        "protocol": "VM05_HISTORY_QUEUE_V1",
        "preflight_only": False,
        "material_identity": "m",
        "audit_identity": "a",
        "qualification_identity": "q",
        "technical_identity": "t",
        "model_signature": observation_signature({"name": "synthetic"}),
        "authors": 128,
        "shards": 128,
        "collector_code_sha256": {},
        "labels_read": False,
        "bridge_outcomes_read": False,
        "strict_author_commitment": material_hash([c["author_id"] for c in whole]),
    }
    m["identity"] = material_hash(m)
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    called = []
    monkeypatch.setattr(
        cache,
        "verify_history_shard",
        lambda *args, **kw: (
            called.append(args[6]) or {"authors": 1, "real_neural_collection": True}
        ),
    )
    args = (
        tmp_path,
        whole,
        {"identity": "m"},
        {"identity": "a"},
        {},
        {"identity": "q"},
        {"identity": "t"},
        {"name": "synthetic"},
    )
    for i in range(127):
        f = tmp_path / f"shard_{i:03d}"
        f.mkdir()
        (f / "summary.json").write_text("{}")
        (f / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="missing=1"):
        cache.complete_history_index(*args)
    assert called == []
    f = tmp_path / "shard_127"
    f.mkdir()
    (f / "summary.json").write_text("{}")
    (f / "manifest.json").write_text("{}")
    index, evidence = cache.complete_history_index(*args)
    assert (
        set(index) == {str(i) for i in range(128)}
        and len(evidence) == 129
        and len(called) == 128
    )
    m["preflight_only"] = True
    m["identity"] = material_hash({k: v for k, v in m.items() if k != "identity"})
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="parent"):
        cache.complete_history_index(*args)


def test_history_feature_math_order_and_mutation_rejection(tmp_path):
    whole = [
        {"author_id": "a", "source_group": "mbti_only"},
        {"author_id": "b", "source_group": "big5_only"},
    ]
    index = {}
    expected = []
    for i, c in enumerate(whole):
        features = {}
        for task in ("behavior", "disposition"):
            for hi, h in enumerate(("own", "matched", "invisible")):
                x = np.zeros((3, 3, 1536), dtype=np.float32)
                x[:, 1, :] = (i + 1) * (hi + 1) * np.array([1, 3, 8])[:, None]
                x[:, 0, :] = (2 if task == "disposition" else 1) * x[:, 1, :]
                x[:, 2, :] = 999 + hi
                features[f"raw__{task}__{h}__block_10"] = x
        packed, aliases = pack_features(features)
        path = tmp_path / material_hash(c["author_id"])
        np.savez_compressed(path.with_suffix(".npz"), **packed)
        meta = {
            "author_id": c["author_id"],
            "source_group": c["source_group"],
            "case_commitment": material_hash(c),
            "cache_format": "lossless_history_aliases_v1",
            "feature_aliases": aliases,
            "feature_sha256": digest(path.with_suffix(".npz")),
        }
        path.with_suffix(".json").write_text(json.dumps(meta))
        index[c["author_id"]] = path
        difference = (
            features["raw__behavior__own__block_10"][:, 1].astype(float)
            - features["raw__behavior__matched__block_10"][:, 1]
        )
        expected.append(np.r_[difference.mean(0), difference.std(0)])
    actual = cache.history_matrix(
        whole[::-1], whole, index, "raw", "block_10", "current_context"
    )
    np.testing.assert_array_equal(actual, np.asarray(expected)[::-1])
    with pytest.raises(ValueError, match="unique"):
        cache.history_matrix(
            [whole[0], whole[0]], whole, index, "raw", "block_10", "current_context"
        )
    index["a"].with_suffix(".npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        cache.history_matrix(whole, whole, index, "raw", "block_10", "current_context")


def test_incomplete_history_is_rejected_before_any_targets(tmp_path, monkeypatch):
    source, whole, reg = source_fixture()
    monkeypatch.setattr(
        runner, "load_registry_inputs", lambda *a: (source, {}, reg, {}, {})
    )
    monkeypatch.setattr(
        runner,
        "registered_cases",
        lambda *a: ({}, whole, {"identity": "m"}, {"identity": "a"}),
    )
    monkeypatch.setattr(
        runner,
        "qualification_packet",
        lambda *a: ({"identity": "q"}, {}, {"name": "synthetic"}),
    )
    monkeypatch.setattr(runner, "require_technical", lambda *a: {"identity": "t"})

    def incomplete(*a):
        raise ValueError("Full128 history shards required")

    def forbidden(*a):
        raise AssertionError("No label access or fitting before full material")

    monkeypatch.setattr(runner, "complete_history_index", incomplete)
    monkeypatch.setattr(runner, "load_source_labels", forbidden)
    monkeypatch.setattr(runner, "nested_fit", forbidden)
    opts = {
        n: tmp_path
        for n in (
            "views",
            "registry_dir",
            "registry_audit_dir",
            "profile_dir",
            "preflight_dir",
            "material_dir",
            "audit_dir",
            "qualification_dir",
            "technical_parent",
            "history_dir",
        )
    }
    a = SimpleNamespace(
        **opts,
        feature="history",
        site="block_10",
        kind="current_context",
        cohort="history_full",
        task="big5",
        output_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="Full128"):
        runner.execute(a)
    assert (
        not list(tmp_path.rglob("*.joblib"))
        and not (tmp_path / "fit/source_label_audit.json").exists()
    )


@pytest.mark.parametrize("task", ["big5", "EI_cont"])
def test_source_fits_replay_and_keep_inner_transforms_local(
    tmp_path, monkeypatch, task
):
    rng = np.random.default_rng(32)
    x = rng.normal(size=(80, 7))
    folds = np.arange(80) % 5
    y = (
        rng.uniform(size=(80, 5))
        if task == "big5"
        else (np.arange(80) % 2).astype(float)
    )
    bundles = [
        {
            "author_id": f"s{i:03d}",
            "source_group": "big5_only" if task == "big5" else "mbti_only",
        }
        for i in range(80)
    ]
    monkeypatch.setattr(
        runner,
        "source_inputs",
        lambda *a: (
            bundles,
            folds,
            x,
            [],
            {},
            {"identity": "source", "source_tables": {}},
        ),
    )

    def labels(root, bs, t, tables):
        assert bs == bundles and t == task
        return y, {"bridge_label_values_converted": False}

    monkeypatch.setattr(runner, "load_source_labels", labels)

    def archive(root, path):
        with zipfile.ZipFile(path, "x"):
            pass

    monkeypatch.setattr(runner, "archive_source", archive)
    bootstrap = runner.bootstrap_score
    monkeypatch.setattr(
        runner, "bootstrap_score", lambda y, p, t: bootstrap(y, p, t, draws=20)
    )
    a = SimpleNamespace(
        output_dir=tmp_path,
        task=task,
        cohort="history_full",
        feature="history",
        site="block_10",
        kind="current_context",
        arm="raw",
        no_pca=False,
        preflight_only=False,
        prepared_root=tmp_path,
    )
    runner.execute(a)
    m, s = verified_packet(tmp_path / "fit")
    packet = load_oof(tmp_path / "fit")
    assert (
        len(packet["ids"]) == 80
        and s["saved_fold_replay_max"] < 1e-12
        and not s["bridge_labels_used"]
    )
    assert m["history_kind"] == "current_context" and s["neural_forwards"] == 0
    for f in range(5):
        model = joblib.load(tmp_path / f"fit/private_fold_{f}.joblib")
        np.testing.assert_allclose(
            model.named_steps["scale"].mean_, x[folds != f].mean(0)
        )
    (tmp_path / "fit/private_fold_0.joblib").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verified_packet(tmp_path / "fit")


def test_complete_preflight_does_not_read_labels_or_fit(tmp_path, monkeypatch):
    bundles = [{"author_id": "synthetic", "source_group": "big5_only"}]
    monkeypatch.setattr(
        runner,
        "source_inputs",
        lambda a: (
            bundles,
            np.array([0]),
            np.zeros((1, 3072)),
            [],
            {"synthetic_verified_parent": True},
            {"identity": "synthetic_source"},
        ),
    )

    def forbidden(*a, **kw):
        raise AssertionError("Preflight must not read labels or fit")

    monkeypatch.setattr(runner, "load_source_labels", forbidden)
    monkeypatch.setattr(runner, "nested_fit", forbidden)

    def archive(root, path):
        with zipfile.ZipFile(path, "x"):
            pass

    monkeypatch.setattr(runner, "archive_source", archive)
    a = SimpleNamespace(
        output_dir=tmp_path,
        task="big5",
        cohort="history_full",
        feature="history",
        site="block_10",
        kind="current_context",
        arm="raw",
        no_pca=False,
        preflight_only=True,
    )
    runner.execute(a)
    s = json.loads((tmp_path / "fit/run_status.json").read_text())
    assert s["status"] == "PREFLIGHT_COMPLETE"
    assert s["source_label_values_read"] is False and s["model_refits"] == 0
    assert not list(tmp_path.rglob("*.joblib"))
