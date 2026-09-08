#!/usr/bin/env python3
"""Pure-CPU cache and metrics helpers; do not import neural/OpenMP runtimes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from .external_material import unpack_features

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/grain_vector_mechanisms"


def read(path):
    """Read a local cache without printing private content."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    """Persist finite JSON for reproducible reports and manifests."""
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def sha(path):
    """Hash local evidence or implementation bytes."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def key(value, salt):
    """Match the pre-existing deterministic author ordering."""
    return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()


def pooled_features(comments, query, poles, family):
    """Exactly reuse S4 mean/query pooling without importing its neural runner."""
    if family not in ("mean", "query"):
        raise ValueError("Unknown pooling family")
    poles = np.asarray(poles, float)
    poles = poles / np.maximum(np.linalg.norm(poles, axis=-1, keepdims=True), 1e-12)
    means, weighted = [], []
    for vectors in comments:
        v = np.asarray(vectors, float)
        means.append(v.mean(0))
        if family == "mean":
            continue
        units = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
        relevance = np.abs(units @ (poles[query, 1] - poles[query, 0]))
        weight = np.exp(20 * (relevance - relevance.max()))
        weighted.append((weight / weight.sum()) @ v)
    mean = np.asarray(means)
    return mean if family == "mean" else np.column_stack((mean, np.asarray(weighted)))


def feature_cache(connection, panel):
    """Load existing source authors with the same integrity checks as S4."""
    manifest, summary = (
        read(connection / "manifest.json"),
        read(connection / "summary.json"),
    )
    if sha(connection / "manifest.json") != summary["manifest_sha256"]:
        raise ValueError("Source connection changed")
    if (
        sha(panel / "private_encoder.npz")
        != read(panel / "manifest.json")["encoder_sha256"]
    ):
        raise ValueError("Query anchors changed")
    if (
        sha(BASE / "d1_author_discovery_r1/private_predictions.csv")
        != manifest["source_sha256"]
    ):
        raise ValueError("Source teacher packet changed")
    with np.load(panel / "private_encoder.npz", allow_pickle=False) as z:
        poles = z["poles"]
    ids, bags = [], []
    for name, digest in read(connection / "private_cache_hashes.json").items():
        path = ROOT / name
        if sha(path) != digest:
            raise ValueError("Encoder cache changed")
        metadata = read(path.with_suffix(".json"))
        with np.load(path, allow_pickle=False) as z:
            values = unpack_features(dict(z), metadata["feature_aliases"])[
                "cue_mask_comment_vectors"
            ]
        if values.shape != (3, 12, 1024) or not np.isfinite(values).all():
            raise ValueError("Incomplete author views")
        ids.append(path.stem)
        bags.append(values)
    membership = read(BASE / "d1_author_discovery_r1/membership_snapshot.json")
    if len(set(ids)) != len(ids) or not set(ids) <= set(
        membership["private_author_ids"]
    ):
        raise ValueError("Invalid D1 membership")
    comments = [view for bag in bags for view in bag]
    output = {}
    for q, query in enumerate(("TF_cont", "neuroticism")):
        output[query] = {
            family: pooled_features(comments, q, poles, family)
            .reshape(len(ids), 3, -1)
            .mean(1)
            for family in ("mean", "query")
        }
    return np.asarray(ids), output


def teacher_arrays(ids, matrix, targets, query):
    """Join exact author/query; reject missing and duplicate authors."""
    native = targets[targets["query"] == query].set_index("author_id")
    if not native.index.is_unique or len(set(ids)) != len(ids):
        raise ValueError("Duplicate source author")
    ix = np.array([i for i, uid in enumerate(ids) if uid in native.index])
    if len(ix) != len(native):
        raise ValueError("Missing source vectors")
    return ids[ix], matrix[ix], native.loc[ids[ix], "native_query"].to_numpy()


def corr(x, y):
    """Return finite Pearson r or None for a constant observable."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) < 1e-14 or np.ptp(y) < 1e-14:
        return None
    return float(pearsonr(x, y).statistic)


def scalar_metrics(y, p, binary=False):
    """Compute the paired associations used in this bounded packet."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if y.shape != p.shape or not np.isfinite([y, p]).all():
        raise ValueError("Invalid paired metrics")
    if binary and not np.isin(y, [0, 1]).all():
        raise ValueError("Binary source labels required")
    return {
        "pearson_r": corr(y, p),
        "auc": float(roc_auc_score(y, p))
        if binary and len(np.unique(y)) == 2
        else None,
    }


def covariance_split(y, teacher, student):
    """Keep signed covariance accounting distinct from explained variance."""
    y, teacher, student = [np.asarray(v, float) for v in (y, teacher, student)]
    cov = lambda a, b: float(np.mean((a - a.mean()) * (b - b.mean())))
    residual = teacher - student
    return {
        "teacher_label_covariance": cov(y, teacher),
        "student_label_covariance": cov(y, student),
        "residual_label_covariance": cov(y, residual),
        "identity_error": abs(cov(y, teacher) - cov(y, student) - cov(y, residual)),
        "residual_label_r": corr(y, residual),
    }


def paired_intervals(y, student, control, binary, draws=1000):
    """Use paired author bootstrap for externally linked frozen OOF predictions."""
    y, p, b = [np.asarray(v, float) for v in (y, student, control)]
    metric = roc_auc_score if binary else corr
    rng = np.random.default_rng(20260908)
    deltas = []
    for _ in range(draws):
        ix = rng.integers(len(y), size=len(y))
        if np.ptp(y[ix]) < 1e-14:
            continue
        a, c = metric(y[ix], p[ix]), metric(y[ix], b[ix])
        if a is not None and c is not None:
            deltas.append(a - c)
    a, c = metric(y, p), metric(y, b)
    return {
        "delta": float(a - c) if a is not None and c is not None else None,
        "delta_lower": float(np.quantile(deltas, 0.025)) if deltas else None,
        "delta_upper": float(np.quantile(deltas, 0.975)) if deltas else None,
    }
