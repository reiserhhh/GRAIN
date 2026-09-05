"""Outcome-independent source exposure and shared-thread controls for VM05."""

from __future__ import annotations

import hashlib
import io
import json

import joblib
import numpy as np

from .controls import material_hash
from .data_views import stable_key
from .external_material import raw_author_features


def exclude_reference(bundles, budget):
    """Exclude legacy discovery authors without inspecting psychological outcomes."""
    required = {"author_id", "old_reference_member"}
    if not required.issubset(budget.columns):
        raise ValueError("Missing frozen reference membership audit")
    if budget.groupby("author_id").old_reference_member.nunique().max() != 1:
        raise ValueError("Inconsistent legacy reference membership across arms")
    # CSV booleans must be actual booleans, not truthy 'False' strings.
    if not budget.old_reference_member.isin([True, False]).all():
        raise ValueError("Invalid reference membership encoding")
    old = set(budget.loc[budget.old_reference_member.eq(True), "author_id"])
    missing = {b["author_id"] for b in bundles} - set(budget.author_id)
    if missing:
        raise ValueError("Source author missing from frozen membership audit")
    return [b for b in bundles if b["author_id"] not in old]


def train_selection(
    train, test, mode="retain", seed=731, scope="outer", target_count=None
):
    """Purge shared-discussion authors or drop the same number independently."""
    if mode not in ("retain", "purge", "random_count"):
        raise ValueError("Unknown discussion control")
    ti = [b["author_id"] for b in train]
    vi = [b["author_id"] for b in test]
    if len(set(ti)) != len(ti) or len(set(vi)) != len(vi) or set(ti) & set(vi):
        raise ValueError("Source split author overlap or duplicate")
    if not train or not test:
        raise ValueError("Empty source control side")
    held_threads = {c["thread_id"] for b in test for v in b["views"] for c in v}
    if not held_threads or "" in held_threads:
        raise ValueError("Missing held-out thread IDs")
    touched = np.asarray(
        [
            any(c["thread_id"] in held_threads for v in b["views"] for c in v)
            for b in train
        ]
    )
    clean = np.flatnonzero(~touched)
    if mode == "retain":
        keep = np.arange(len(train))
    elif mode == "purge":
        keep = clean
    else:
        if seed not in range(731, 736):
            raise ValueError("Random count-control seed was not frozen")
        ranking = sorted(
            range(len(train)),
            key=lambda i: stable_key(ti[i], f"vm05_drop:{seed}:{scope}"),
        )
        target = len(clean) if target_count is None else target_count
        if not 20 <= target <= len(train):
            raise ValueError("Invalid matched training count")
        keep = np.sort(np.asarray(ranking[:target], dtype=int))
    if len(keep) < 20:
        raise ValueError("Insufficient authors after discussion control")
    record = {
        "scope": scope,
        "mode": mode,
        "seed": seed,
        "train_available": len(train),
        "test_authors": len(test),
        "shared_thread_authors": int(touched.sum()),
        "train_retained": len(keep),
        "retained_shared_thread_authors": int(touched[keep].sum()),
        "train_author_commitment": material_hash([ti[i] for i in keep]),
        "test_author_commitment": material_hash(vi),
        "label_values_used_for_selection": False,
        "matched_reference_count": target_count,
    }
    return keep, record


def inner_selector(bundles, mode, seed, scope, log, reference_bundles=None):
    """Create an inner-CV callback over exactly the outer-training author pool."""

    def select(train, valid):
        target = None
        if reference_bundles is not None:
            if mode != "random_count" or len(reference_bundles) != len(bundles):
                raise ValueError("Invalid inner matched-count reference")
            shadow, _ = train_selection(
                [reference_bundles[i] for i in train],
                [reference_bundles[i] for i in valid],
                "purge",
            )
            target = len(shadow)
        local, record = train_selection(
            [bundles[i] for i in train],
            [bundles[i] for i in valid],
            mode,
            seed,
            f"{scope}:inner:{material_hash(valid.tolist())[:12]}",
            target_count=target,
        )
        log.append(record)
        return train[local]

    return select


def frozen_masked_features(root, preflight_dir, bundles, arm, cache_dir=None):
    """Reuse only label-free frozen projections with exact source/code commitments."""
    from scripts.run_grain_vector_mechanism_external_features import bundle_arms

    pre = json.loads((preflight_dir / "manifest.json").read_text())
    path = (
        root
        / "results/v7_multiview_projection/e1_v72_full_20260715/artifacts/common_source_comment_representation.joblib"
    )
    runtime_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if runtime_sha != pre["runtime_sha256"]:
        raise ValueError("Frozen intermediate operator changed")
    identity = {
        "formula": "FROZEN_V7_LEXICAL_MEAN_STD_V1",
        "arm": arm,
        "source_material_sha256": material_hash(bundles),
        "runtime_sha256": runtime_sha,
        "feature_names": pre["feature_names"],
        "labels_read": False,
        "code": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "grain_research/vector_mechanisms/external_controls.py",
                "grain_research/vector_mechanisms/external_material.py",
                "scripts/run_grain_vector_mechanism_external_features.py",
                "suica_core/v7_psychometric.py",
            )
        },
    }
    key = material_hash(identity)
    data_path = meta_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        data_path, meta_path = cache_dir / (key + ".npz"), cache_dir / (key + ".json")
        if data_path.exists() or meta_path.exists():
            if not data_path.exists() or not meta_path.exists():
                raise ValueError("Incomplete label-free cache; preserve and inspect")
            meta = json.loads(meta_path.read_text())
            if (
                meta["identity"] != identity
                or hashlib.sha256(data_path.read_bytes()).hexdigest()
                != meta["npz_sha256"]
            ):
                raise ValueError("Label-free cache identity/checksum mismatch")
            with np.load(data_path, allow_pickle=False) as z:
                if z["author_ids"].tolist() != [b["author_id"] for b in bundles]:
                    raise ValueError("Label-free cache author order changed")
                return z["features"].copy()
    runtime = joblib.load(path)
    x = np.stack(
        [
            raw_author_features(
                runtime.transform(bundle_arms(b)[arm])[None, :, :], pre["feature_names"]
            )[0]
            for b in bundles
        ]
    )
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite frozen feature")
    if data_path is not None:
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer, features=x, author_ids=np.asarray([b["author_id"] for b in bundles])
        )
        with data_path.open("xb") as handle:
            handle.write(buffer.getvalue())
        with meta_path.open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "identity": identity,
                    "npz_sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
                },
                handle,
                indent=2,
            )
    return x
