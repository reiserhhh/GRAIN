#!/usr/bin/env python3
"""Strict source cohorts and complete-history cache readouts; no label access."""

import json
from pathlib import Path

import numpy as np

from .controls import material_hash
from .external_material import shard_material, unpack_features
from .history_cache import verify_history_shard
from .history_collection import KINDS, SITES, history_channel, observation_signature
from .ready_comparison import digest
from .ready_registry import TASKS

COHORTS = ("history_full", "history_ready")


def source_indices(source, whole, registry, cohort, task):
    """Intersect strict histories and optional all-arm support, preserving original order."""
    if cohort not in COHORTS or task not in TASKS:
        raise ValueError("Unknown frozen history cohort/task")
    ids = [b["author_id"] for b in source]
    strict = {b["author_id"]: b for b in whole}
    if (
        len(strict) != len(whole)
        or len(set(ids)) != len(ids)
        or not np.array_equal(ids, registry["author_ids"])
        or not set(strict).issubset(ids)
    ):
        raise ValueError("History and common author registries differ")
    reference = np.asarray(registry["reference_member"])
    ready = np.asarray(registry["all_arms_ready_reference_free"])
    if (
        reference.dtype != bool
        or ready.dtype != bool
        or reference.shape != (len(ids),)
        or ready.shape != (len(ids),)
    ):
        raise ValueError("History eligibility masks must be aligned booleans")
    dictionary = {u for u, r in zip(ids, reference, strict=True) if r}
    if not dictionary or any(
        b["source_group"] != "mbti_only" for b in source if b["author_id"] in dictionary
    ):
        raise ValueError("History dictionary must contain MBTI-only reference authors")
    if dictionary & set(strict):
        raise ValueError("Reference donors cannot be source/evaluation targets")
    for b in source:
        if b["author_id"] not in strict:
            continue
        c = strict[b["author_id"]]
        if (
            c["source_group"] != b["source_group"]
            or c["official_folds"] != b["official_folds"]
            or len(c["views"]) != 3
            or any(v["donor_author_id"] not in dictionary for v in c["views"])
        ):
            raise ValueError(
                "History roles/folds or fixed-dictionary provenance differ"
            )
    group = "big5_only" if task == "big5" else "mbti_only"
    ix = np.asarray(
        [
            i
            for i, b in enumerate(source)
            if b["author_id"] in strict
            and b["source_group"] == group
            and (cohort == "history_full" or ready[i])
        ],
        dtype=int,
    )
    if (
        not len(ix)
        or set(registry[task + "_fold"][ix]) != set(range(5))
        or any(
            registry[task + "_fold"][i] != source[i]["official_folds"][task] for i in ix
        )
    ):
        raise ValueError("History source cohort requires unchanged official5 folds")
    return ix


def complete_history_index(
    folder, whole, material, audit, code, qualification, technical, model
):
    """Verify every one of128 shards, never fit whichever partial shards have arrived."""
    root = Path(folder)
    m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "protocol": "VM05_HISTORY_QUEUE_V1",
        "preflight_only": False,
        "material_identity": material["identity"],
        "audit_identity": audit["identity"],
        "qualification_identity": qualification["identity"],
        "technical_identity": technical["identity"],
        "model_signature": observation_signature(model),
        "authors": len(whole),
        "shards": 128,
        "collector_code_sha256": code,
        "labels_read": False,
        "bridge_outcomes_read": False,
        "strict_author_commitment": material_hash([c["author_id"] for c in whole]),
    }
    if (
        any(m.get(k) != v for k, v in expected.items())
        or material_hash({k: v for k, v in m.items() if k != "identity"})
        != m["identity"]
    ):
        raise ValueError(
            "History queue parent differs from the full qualified instrument"
        )
    absent = [
        i
        for i in range(128)
        if not (root / f"shard_{i:03d}" / "summary.json").is_file()
    ]
    if absent:
        raise ValueError(
            f"Full128 history shards required before any source fit; missing={len(absent)}"
        )
    paths, evidence = {}, {"queue_manifest_sha256": digest(root / "manifest.json")}
    for i in range(128):
        shard = root / f"shard_{i:03d}"
        verified = verify_history_shard(
            shard,
            whole,
            material,
            audit,
            code,
            128,
            i,
            qualification=qualification["identity"],
            technical=technical["identity"],
            model_signature=observation_signature(model),
        )
        cases = shard_material(whole, 128, i)
        if verified["authors"] != len(cases) or not verified["real_neural_collection"]:
            raise ValueError("History shard lacks real qualified observations")
        for c in cases:
            uid = c["author_id"]
            if uid in paths:
                raise ValueError("Duplicate history author across shards")
            paths[uid] = shard / "private_cache" / material_hash(uid)
        evidence[f"shard_{i:03d}"] = {
            "manifest_sha256": digest(shard / "manifest.json"),
            "summary_sha256": digest(shard / "summary.json"),
            **verified,
        }
    if set(paths) != {c["author_id"] for c in whole}:
        raise ValueError("History cache union differs from strict registry")
    return paths, evidence


def history_matrix(bundles, whole, index, arm, site, kind):
    """Read equal-view contrasts in requested order; token-effect covariance is unavailable."""
    from .external_material import ARMS

    if arm not in ARMS or site not in SITES or kind not in KINDS:
        raise ValueError("Unknown declared history channel")
    ids = [b["author_id"] for b in bundles]
    by_id = {c["author_id"]: c for c in whole}
    if (
        not ids
        or len(set(ids)) != len(ids)
        or set(index) != set(by_id)
        or not set(ids).issubset(index)
    ):
        raise ValueError(
            "History readout requires complete index and unique supported authors"
        )
    rows = []
    for b in bundles:
        uid = b["author_id"]
        path = index[uid]
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if (
            meta["author_id"] != uid
            or meta["source_group"] != b["source_group"]
            or meta["case_commitment"] != material_hash(by_id[uid])
            or meta["cache_format"] != "lossless_history_aliases_v1"
            or meta["feature_sha256"] != digest(path.with_suffix(".npz"))
        ):
            raise ValueError("History feature cache changed after verification")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
            h = unpack_features({k: z[k] for k in z.files}, meta["feature_aliases"])
        value = history_channel(h, arm, site, kind)
        if value.shape != (3072,) or not np.isfinite(value).all():
            raise ValueError("Expected finite pinned-Qwen3072-column history channel")
        rows.append(value)
    return np.stack(rows)
