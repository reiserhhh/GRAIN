#!/usr/bin/env python3
"""Cached task/pooling transformations preserve views, dimensions and identities."""

import copy
import hashlib
import json

import numpy as np
import pytest

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_material import (
    pack_features,
    shard_material,
    unpack_features,
)
from grain_research.vector_mechanisms.external_readout_cache import (
    cached_readout_matrix,
)
from grain_research.vector_mechanisms.external_readouts import (
    ARMS,
    SITES,
    author_channels,
    channel,
    margin_features,
    readout_table,
    variance_parts,
)
from scripts.audit_grain_cached_readouts import reconstruct


def fixture():
    """Hypothetical three-view data only; no source author or target values."""
    rng = np.random.default_rng(92)
    h, rows = {}, []
    for arm in ARMS:
        for site in SITES:
            for name in ("body_mean", "decision_behavior", "decision_disposition"):
                h[f"{arm}_{site}_{name}"] = rng.normal(size=(3, 4))
            h[f"{arm}_{site}_body_std"] = rng.uniform(0.1, 2, size=(3, 4))
        for v, n in enumerate((2, 3, 5)):
            for t in ("behavior", "disposition"):
                rows.append(
                    {
                        "arm": arm,
                        "view": v,
                        "task": t,
                        "body_tokens": n,
                        "prompt_tokens": n + 10,
                        "margin": float(v if t == "behavior" else 2 * v - 1),
                    }
                )
    return h, {"readouts": rows, "causal_prefix_max": 0, "replay_max": 0}


def test_total_variance_matches_explicit_tokens_and_view_weighting():
    groups = [
        np.asarray([[1.0, 2.0], [5.0, 8.0]]),
        np.asarray([[4.0, 2.0], [6.0, 8.0], [5.0, 3.0]]),
        np.asarray([[8.0, 1.0], [3.0, 6.0], [4.0, 9.0], [1.0, 4.0], [6.0, 7.0]]),
    ]
    m, s = np.stack([x.mean(0) for x in groups]), np.stack([x.std(0) for x in groups])
    mean, within, between = variance_parts(m, s, [len(x) for x in groups])
    np.testing.assert_allclose(mean, np.concatenate(groups).mean(0), atol=1e-12)
    np.testing.assert_allclose(
        within + between, np.concatenate(groups).var(0), atol=1e-12
    )
    mean_e, within_e, between_e = variance_parts(m, s, np.ones(3))
    np.testing.assert_allclose(mean_e, m.mean(0))
    np.testing.assert_allclose(
        within_e + between_e, (s * s + m * m).mean(0) - m.mean(0) ** 2
    )
    assert not np.allclose(mean, mean_e)
    assert not np.allclose(np.sqrt(within_e + between_e), s.mean(0))


def test_all_channels_independently_reconstruct_and_alias_losslessly():
    h, meta = fixture()
    actual, expected = author_channels(h, meta), reconstruct(h, meta)
    assert len(actual) == len(expected) == 75 and set(actual) == set(expected)
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-12)
        assert actual[key].shape == (
            (4,)
            if key.endswith("logit_margins")
            else (12,)
            if key.endswith("body_variance_parts")
            else (8,)
        )
    h["cue_mask_block_10_body_mean"] = h["raw_block_10_body_mean"].copy()
    packed, aliases = pack_features(h)
    restored = unpack_features(packed, aliases)
    assert aliases
    for key, value in author_channels(h, meta).items():
        np.testing.assert_array_equal(value, author_channels(restored, meta)[key])


def test_task_difference_is_paired_and_has_defined_polarity():
    h, meta = fixture()
    b, d = h["raw_block_10_decision_behavior"], h["raw_block_10_decision_disposition"]
    expected = np.r_[(d - b).mean(0), (d - b).std(0)]
    actual = channel(h, meta, "raw", "block_10", "decision_task_delta")
    np.testing.assert_allclose(actual, expected)
    assert not np.allclose(actual[4:], d.std(0) - b.std(0))
    changed = copy.deepcopy(h)
    changed["raw_block_10_decision_behavior"] = d
    changed["raw_block_10_decision_disposition"] = b
    flipped = channel(changed, meta, "raw", "block_10", "decision_task_delta")
    np.testing.assert_allclose(flipped[:4], -actual[:4])
    np.testing.assert_allclose(flipped[4:], actual[4:])
    np.testing.assert_allclose(
        margin_features(meta, "raw"), [1, np.sqrt(2 / 3), 1, np.sqrt(8 / 3)]
    )


def test_metadata_or_array_corruption_is_not_silently_imputed():
    h, meta = fixture()
    for modify in (
        lambda m: m["readouts"].pop(),
        lambda m: m["readouts"].append(m["readouts"][0]),
        lambda m: m["readouts"][0].update(body_tokens=0),
        lambda m: m["readouts"][0].update(body_tokens=3),
        lambda m: m.update(causal_prefix_max=0.01),
        lambda m: m["readouts"][0].update(margin=float("nan")),
    ):
        bad = copy.deepcopy(meta)
        modify(bad)
        with pytest.raises(ValueError):
            readout_table(bad)
    for values in (np.ones((2, 4)), np.ones((3, 5)), np.full((3, 4), np.nan)):
        bad = h | {"raw_block_10_decision_behavior": values}
        with pytest.raises(ValueError):
            channel(bad, meta, "raw", "block_10", "decision_task_delta")
    with pytest.raises(ValueError, match="history delta"):
        channel(h, meta, "raw", "block_10", "own_minus_none_history")
    with pytest.raises(ValueError):
        variance_parts(np.ones((3, 4)), np.full((3, 4), -1), [1, 1, 1])


def test_complete_cache_readout_preserves_order_and_rejects_unused_corruption(tmp_path):
    source = [
        {"author_id": f"toy_{i:03d}", "source_group": "mbti_only"} for i in range(24)
    ]
    common = {"material_sha256": material_hash(source)}
    queue = {
        "shards": 24,
        "material_sha256": common["material_sha256"],
        "code_sha256": "frozen_test",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(queue))
    expected, paths = {}, {}
    for i in range(24):
        shard = shard_material(source, 24, i)
        uid = shard[0]["author_id"]
        for stage in ("bge_large", "e5_large", "qwen"):
            folder = tmp_path / stage / f"shard_{i:02d}"
            folder.mkdir(parents=True)
            m = {
                "authors": [uid],
                "stage": stage,
                "shard_count": 24,
                "shard_index": i,
                "code_sha256": "frozen_test",
                "full_material_sha256": common["material_sha256"],
                "source_material_sha256": material_hash(shard),
            }
            m["identity"] = material_hash(m)
            (folder / "manifest.json").write_text(json.dumps(m))
            (folder / "run_status.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "authors": 1,
                        "parent_identity": m["identity"],
                    }
                )
            )
            if stage != "qwen":
                continue
            h, meta = fixture()
            h["raw_block_10_decision_disposition"] += i
            expected[uid] = channel(h, meta, "raw", "block_10", "decision_task_delta")
            cache = folder / "private_cache"
            cache.mkdir()
            path = cache / (uid + ".npz")
            np.savez_compressed(path, **h)
            paths[uid] = path
            meta.update(
                author_id=uid,
                feature_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            (cache / (uid + ".json")).write_text(json.dumps(meta))
    subset = [source[9], source[2], source[7]]
    got = cached_readout_matrix(
        subset, source, common, tmp_path, "raw", "block_10", "decision_task_delta"
    )
    np.testing.assert_allclose(
        got, np.stack([expected[b["author_id"]] for b in subset])
    )
    margins = cached_readout_matrix(
        subset, source, common, tmp_path, "raw", "block_10", "logit_margins"
    )
    assert margins.shape == (3, 4)
    with pytest.raises(ValueError, match="site-independent"):
        cached_readout_matrix(
            subset, source, common, tmp_path, "raw", "block_04", "logit_margins"
        )
    paths[source[0]["author_id"]].write_bytes(b"corrupted nonselected author")
    with pytest.raises(ValueError, match="checksum"):
        cached_readout_matrix(
            subset, source, common, tmp_path, "raw", "block_10", "decision_task_delta"
        )


def test_energy_partition_is_descriptive_not_a_new_noise_estimator():
    from scripts.summarize_grain_readout_qualification import energy_partition

    x = np.asarray([[1.0, 1.0], [1.0, -1.0]])
    row = energy_partition(x)
    assert row["common_energy_fraction"] == 0.5
    assert row["between_bundle_residual_fraction"] == 0.5
    assert energy_partition(x * 7)["common_energy_fraction"] == 0.5
    shifted = energy_partition(x + [10, 0])
    assert shifted["common_energy_fraction"] > 0.99
    zero = energy_partition(np.zeros((3, 2)))
    assert zero["zero_total_energy"] and zero["common_energy_fraction"] is None
