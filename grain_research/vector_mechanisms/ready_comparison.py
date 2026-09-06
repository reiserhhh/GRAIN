#!/usr/bin/env python3
"""Exact eligible-source features and nested readouts for VM05 comparisons."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from .controls import material_hash
from .external_heads import metrics, predict_head, select_head

FEATURES = (
    "word_char",
    "sorted_distances",
    "bge_large",
    "e5_large",
    "qwen_block_04",
    "qwen_block_10",
    "qwen_block_16",
    "qwen_final_norm",
    "grain_qwen_block_10",
    "grain_bge_large",
)


def digest(path):
    """Hash an input/output without a second full in-memory copy."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verified_packet(folder):
    """Verify manifest identity and complete packet outputs before using arrays."""
    folder = Path(folder).resolve()
    m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    s = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if (
        material_hash({k: v for k, v in m.items() if k != "identity"}) != m["identity"]
        or s["status"] != "COMPLETE"
        or s["parent_identity"] != m["identity"]
        or not s["outputs_sha256"]
    ):
        raise ValueError("Incomplete or changed source packet")
    for name, sha in s["outputs_sha256"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or digest(path) != sha:
            raise ValueError("Source packet checksum mismatch")
    return m, s


def require_observer_completion(vector_dir, material_sha256):
    """Require every planned observer before selecting any ready-only neural arm."""
    root = Path(vector_dir)
    queue = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if queue["shards"] != 24 or queue["material_sha256"] != material_sha256:
        raise ValueError("Require original24-shard common observation protocol")
    for stage in ("bge_large", "e5_large", "qwen"):
        for shard in range(24):
            status = root / stage / f"shard_{shard:02d}" / "run_status.json"
            if (
                not status.exists()
                or json.loads(status.read_text(encoding="utf-8"))["status"]
                != "COMPLETE"
            ):
                raise ValueError("All72 observer shards must finish before comparison")
    return queue


def load_registry_inputs(views, registry_dir, audit_dir, profile_dir, preflight_dir):
    """Bind audited eligibility to the original profile, evidence and reference IDs."""
    import pandas as pd

    from .external_controls import exclude_reference
    from .ready_registry import registry_arrays

    source = json.loads(Path(views).read_text(encoding="utf-8"))
    common = json.loads(
        (Path(preflight_dir) / "common_material_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    m, _ = verified_packet(registry_dir)
    pm, ps = verified_packet(profile_dir)
    audit = json.loads((Path(audit_dir) / "summary.json").read_text(encoding="utf-8"))
    if (
        m["protocol"] != "VM05_SAME_READY_REGISTRY_V1"
        or m["material_sha256"] != material_hash(source)
        or m["material_sha256"] != common["material_sha256"]
        or m["profile_identity"] != pm["identity"]
        or m["profile_outputs_sha256"] != ps["outputs_sha256"]
        or m["labels_read"]
        or m["bridge_outcomes_read"]
        or pm["labels_read"]
        or audit["status"] != "COMPLETE"
        or audit["registry_identity"] != m["identity"]
        or audit["source_labels_read"]
        or audit["bridge_outcomes_read"]
    ):
        raise ValueError("Registry/profile/common material/audit mismatch")
    budget_path = Path(preflight_dir) / "budget_by_author.csv"
    if digest(budget_path) != m["budget_sha256"]:
        raise ValueError("Reference membership budget changed")
    budget = pd.read_csv(budget_path, dtype={"author_id": str})
    clean = {b["author_id"] for b in exclude_reference(source, budget)}
    reference = {b["author_id"] for b in source} - clean
    with np.load(
        Path(profile_dir) / "private_profile_features.npz", allow_pickle=False
    ) as z:
        profiles = {k: z[k] for k in z.files}
    expected = registry_arrays(source, profiles, reference)
    with np.load(
        Path(registry_dir) / "private_ready_registry.npz", allow_pickle=False
    ) as z:
        registry = {k: z[k] for k in z.files}
    if set(registry) != set(expected) or any(
        not np.array_equal(registry[k], expected[k]) for k in expected
    ):
        raise ValueError("Registry arrays no longer reproduce")
    return source, common, registry, profiles, m


def feature_matrix(
    source, registry, profiles, cohort, task, arm, feature, dense_loader
):
    """Return aligned source-only rows; concatenation never fits a transform here."""
    from scripts.run_grain_vector_mechanism_external_features import bundle_arms

    from .ready_registry import source_indices

    if feature not in FEATURES or arm not in ("raw", "cue_mask", "placebo_mask"):
        raise ValueError("Unsupported frozen feature/arm")
    if cohort.startswith("raw_ready") and arm != "raw":
        raise ValueError("Masked-arm comparison requires all-arms-ready cohort")
    ix = source_indices(registry, cohort, task)
    bundles = [source[i] for i in ix]
    ids = np.asarray([b["author_id"] for b in bundles])
    if not np.array_equal(ids, registry["author_ids"][ix]):
        raise ValueError("Source author order differs from registry")
    if feature == "word_char":
        x = np.asarray(["\n\n".join(bundle_arms(b)[arm]) for b in bundles])
        return x, bundles, ix, [{"name": "word_char", "input": "36 complete comments"}]
    g = profiles[arm + "_sorted_distances"][ix]
    if g.shape != (len(ix), 16) or not np.isfinite(g).all():
        raise ValueError("Formal profile is not a finite16-column accepted profile")
    if feature == "sorted_distances":
        if not np.any(
            registry[cohort.removesuffix("_reference_free")] != registry[cohort]
        ):
            raise ValueError(
                "Reuse completed formal profile parent; no identical refit"
            )
        blocks = [("sorted_distances", g)]
    else:
        observer = feature.removeprefix("grain_")
        h = np.asarray(dense_loader(bundles, observer, arm))
        if h.ndim != 2 or len(h) != len(ix) or not np.isfinite(h).all():
            raise ValueError("Malformed observer matrix")
        blocks = [("sorted_distances", g)] if feature.startswith("grain_") else []
        blocks.append((observer, h))
    boundaries, start = [], 0
    for name, values in blocks:
        stop = start + values.shape[1]
        boundaries.append({"name": name, "column_start": start, "column_stop": stop})
        start = stop
    return np.concatenate([v for _, v in blocks], axis=1), bundles, ix, boundaries


def support_table(y, ids, folds, task):
    """Check target support before any fit, without adapting folds to outcomes."""
    if (
        len(ids) != len(set(ids))
        or len(y) != len(ids)
        or len(folds) != len(ids)
        or set(folds) != set(range(5))
        or not np.isfinite(y).all()
    ):
        raise ValueError("Invalid source targets/authors/folds")
    if task == "big5":
        if y.shape != (len(ids), 5) or np.any((y < 0) | (y > 1)):
            raise ValueError("Invalid five-trait source targets")
    elif y.shape != (len(ids),) or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid binary source targets")
    rows = []

    def check(index, scope, fitting):
        values = y[index]
        spread = np.std(values, axis=0)
        supported = len(index) >= (20 if fitting else 3) and bool(
            np.all(spread > 1e-12)
        )
        rows.append(
            {
                "scope": scope,
                "authors": len(index),
                "fit_partition": fitting,
                "supported": supported,
                "minimum_target_std": float(np.min(spread)),
                "positive_count": int(values.sum()) if task != "big5" else None,
                "author_commitment": material_hash(ids[index].tolist()),
            }
        )

    for fold in range(6):
        train = np.flatnonzero(folds != fold) if fold < 5 else np.arange(len(ids))
        scope = f"outer:{fold}" if fold < 5 else "full_source"
        check(train, scope + ":train", True)
        if fold < 5:
            check(np.flatnonzero(folds == fold), scope + ":test", False)
        for j, (ti, vi) in enumerate(
            KFold(3, shuffle=True, random_state=731).split(train)
        ):
            check(train[ti], f"{scope}:inner:{j}:train", True)
            check(train[vi], f"{scope}:inner:{j}:valid", False)
    return rows


def nested_fit(x, y, ids, folds, task, feature_kind, pca, checkpoint):
    """Fit the existing frozen head on exact ready rows; no bridge or outer tuning."""
    support = support_table(y, ids, folds, task)
    if len(x) != len(ids) or not all(r["supported"] for r in support):
        raise ValueError("Insufficient target support; do not resample until supported")
    pred = np.full(y.shape, np.nan)
    selected, fold_metrics = [], []
    for fold in range(5):
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        model, search, parameter = select_head(
            x[train], y[train], ids[train], feature_kind, task, pca=pca
        )
        pred[test] = predict_head(model, x[test], task)
        selected.append(
            {
                "fold": fold,
                "train_n": len(train),
                "test_n": len(test),
                "parameter": parameter,
                "inner_search": search,
                "pca_components": int(model.named_steps["pca"].n_components_)
                if "pca" in model.named_steps
                else None,
                "train_author_commitment": material_hash(ids[train].tolist()),
                "test_author_commitment": material_hash(ids[test].tolist()),
            }
        )
        fold_metrics.extend(
            dict(r, fold=fold) for r in metrics(y[test], pred[test], task)
        )
        checkpoint(fold, model, selected)
    full, search, parameter = select_head(x, y, ids, feature_kind, task, pca=pca)
    return pred, selected, fold_metrics, full, search, parameter
