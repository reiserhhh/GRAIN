#!/usr/bin/env python3
"""Lossless, label-free disk fold plans compatible with the frozen null estimator."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.model_selection import KFold

from .controls import material_hash
from .external_heads import make_head
from .external_null import parameter_grid


def digest(path):
    """Stream an artifact checksum rather than duplicating it in RAM."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class DiskFoldPlans:
    """Load one outer fold at a time; clear yielded arrays before loading the next."""

    def __init__(self, root, expected):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["config"] != expected or self.manifest[
            "identity"
        ] != material_hash({k: v for k, v in self.manifest.items() if k != "identity"}):
            raise ValueError("Changed disk fold plan")
        self.stamps = {}
        self.validate_files()

    def validate_files(self):
        """Verify every saved matrix at start/resume and final acceptance."""
        for row in self.manifest["arrays"]:
            path = (self.root / row["path"]).resolve()
            if (
                not path.is_relative_to(self.root.resolve())
                or digest(path) != row["sha256"]
            ):
                raise ValueError("Disk fold matrix checksum mismatch")
            stat = path.stat()
            self.stamps[row["path"]] = (stat.st_size, stat.st_mtime_ns)

    def _load(self, row):
        path = self.root / row["path"]
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != self.stamps[row["path"]]:
            raise ValueError("Fold cache changed during execution")
        result = (
            sparse.load_npz(path)
            if row["sparse"]
            else np.load(path, allow_pickle=False)
        )
        if list(result.shape) != row["shape"] or str(result.dtype) != row["dtype"]:
            raise ValueError("Fold cache shape or dtype mismatch")
        return result

    def __iter__(self):
        """Preserve canonical fold order and eagerly release aliased inner tables."""
        for saved in self.manifest["plans"]:
            plan = {
                k: v for k, v in saved.items() if k not in ("inner", "xtrain", "xtest")
            }
            for key in ("train", "test"):
                plan[key] = np.asarray(plan[key], dtype=int)
            plan["inner"] = []
            try:
                for part in saved["inner"]:
                    plan["inner"].append(
                        {
                            "fit": np.asarray(part["fit"], dtype=int),
                            "valid": np.asarray(part["valid"], dtype=int),
                            "xfit": self._load(part["xfit"]),
                            "xvalid": self._load(part["xvalid"]),
                        }
                    )
                plan["xtrain"], plan["xtest"] = (
                    self._load(saved["xtrain"]),
                    self._load(saved["xtest"]),
                )
                yield plan
            finally:
                # The consumer retains its last `part` local; clear that alias too.
                for part in plan["inner"]:
                    part.clear()
                plan.clear()
                gc.collect()


def prepare_disk_folds(
    root,
    x,
    ids,
    folds,
    kind,
    task,
    *,
    parent_identity,
    pca=True,
    max_bytes=10_000_000_000,
):
    """Fit each label-free inner/outer transform once and save exact matrices."""
    if (
        len(set(ids)) != len(ids)
        or set(folds) != set(range(5))
        or len(x) != len(ids)
        or len(folds) != len(ids)
    ):
        raise ValueError("Unique authors and five official folds required")
    input_sha = (
        material_hash(x.tolist())
        if kind == "word_char"
        else hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    )
    root = Path(root)
    config = {
        "protocol": "VM05_DISK_FOLD_CACHE_V1",
        "parent_identity": parent_identity,
        "ids_sha256": material_hash(ids.tolist()),
        "folds_sha256": material_hash(folds.tolist()),
        "input_sha256": input_sha,
        "input_shape": list(x.shape),
        "input_dtype": str(x.dtype),
        "feature_kind": kind,
        "task": task,
        "pca": pca,
        "max_bytes": max_bytes,
        "implementation_sha256": digest(Path(__file__)),
        "estimator_sha256": digest(Path(__file__).with_name("external_heads.py")),
        "null_estimator_sha256": digest(Path(__file__).with_name("external_null.py")),
    }
    if (root / "manifest.json").exists():
        return DiskFoldPlans(root, config)
    root.mkdir(parents=True, exist_ok=True)
    if any(p for p in root.iterdir() if not p.name.startswith("._")):
        raise ValueError("Incomplete disk cache must be preserved, not overwritten")
    arrays, plans, total = [], [], 0

    def save(matrix, name):
        nonlocal total
        is_sparse = sparse.issparse(matrix)
        path = root / (name + (".npz" if is_sparse else ".npy"))
        if is_sparse:
            sparse.save_npz(path, matrix, compressed=True)
        else:
            np.save(path, matrix, allow_pickle=False)
        total += path.stat().st_size
        if total > max_bytes:
            raise ValueError("Disk fold cache exceeds frozen storage bound")
        row = {
            "path": path.name,
            "sparse": is_sparse,
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "sha256": digest(path),
            "bytes": path.stat().st_size,
        }
        arrays.append(row)
        return row

    d = None if kind == "word_char" else x.shape[1]
    for outer in range(5):
        train, test = np.flatnonzero(folds != outer), np.flatnonzero(folds == outer)
        inner = []
        for j, (ti, vi) in enumerate(
            KFold(3, shuffle=True, random_state=731).split(train)
        ):
            fit, valid = train[ti], train[vi]
            transform = make_head(
                kind, task, parameter_grid(task)[0], len(fit), d, pca=pca
            )[:-1]
            a = save(transform.fit_transform(x[fit]), f"fold_{outer}_inner_{j}_fit")
            b = save(transform.transform(x[valid]), f"fold_{outer}_inner_{j}_valid")
            inner.append(
                {"fit": fit.tolist(), "valid": valid.tolist(), "xfit": a, "xvalid": b}
            )
            del transform
            gc.collect()
        transform = make_head(
            kind, task, parameter_grid(task)[0], len(train), d, pca=pca
        )[:-1]
        a = save(transform.fit_transform(x[train]), f"fold_{outer}_train")
        b = save(transform.transform(x[test]), f"fold_{outer}_test")
        plans.append(
            {
                "outer": outer,
                "train": train.tolist(),
                "test": test.tolist(),
                "inner": inner,
                "xtrain": a,
                "xtest": b,
                "n_features": d,
                "feature_kind": kind,
                "pca": pca,
            }
        )
        del transform
        gc.collect()
    manifest = {
        "config": config,
        "plans": plans,
        "arrays": arrays,
        "bytes": total,
        "labels_used": False,
    }
    manifest["identity"] = material_hash(manifest)
    with (root / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2)
    return DiskFoldPlans(root, config)
