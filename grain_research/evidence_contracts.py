#!/usr/bin/env python3
"""Read-only lineage and top-1 checks for frozen predecessor evidence."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def digest(path):
    """Hash a local source or cache artifact without deserializing a model."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_evidence_rows(evidence, sentences, items, threshold):
    """Require the exact ordered candidates, item keys and numeric acceptance rule."""
    if len(evidence) != len(sentences):
        raise ValueError("Evidence row count mismatch")
    if not pd.api.types.is_bool_dtype(evidence.accepted):
        raise ValueError("Accepted flags must be booleans")
    score = evidence.similarity.to_numpy(float)
    if not np.isfinite(score).all() or np.any(abs(score) > 1 + 1e-10):
        raise ValueError("Invalid cosine")
    if not np.array_equal(evidence.accepted, score >= threshold):
        raise ValueError("Accepted flag differs from cosine threshold")
    mapping = {i["id"]: i for i in items}
    for e, s in zip(evidence.itertuples(), sentences, strict=True):
        if any(
            getattr(e, k) != s[k]
            for k in ("author_index", "comment_index", "comment_id", "start", "end")
        ):
            raise ValueError("Evidence source identity mismatch")
        if e.item_id not in mapping or any(
            getattr(e, k) != mapping[e.item_id][k] for k in ("key", "facet", "trait")
        ):
            raise ValueError("Evidence item/key mismatch")


def verify_frozen_top1(material, producer, evidence, data):
    """Recompute matches from checksum-bound existing vectors; never encode again."""
    contract = json.loads((producer / "run_contract.json").read_text())
    if (
        contract["source_manifest"] != digest(material / "material_manifest.json")
        or contract["threshold"] != 0.6
    ):
        raise ValueError("Producer source or threshold changed")
    items = json.loads((material / "ipip_neo300.json").read_text())["items"]
    validate_evidence_rows(evidence, data["sentences"], items, 0.6)
    cache = producer / "paraphrase_cache"
    encoding = json.loads((cache / "contract.json").read_text())
    texts = [i["text"] for i in items] + [s["text"] for s in data["sentences"]]
    text_hash = hashlib.sha256(
        json.dumps(texts, ensure_ascii=False).encode()
    ).hexdigest()
    if (
        encoding["texts_sha256"] != text_hash
        or encoding["revision"] != contract["revision"]
        or encoding["model"] != contract["model"]
    ):
        raise ValueError("Encoder material/model differs")
    blocks, next_index = [], 0
    for path in sorted(cache.glob("batch_*.npz")):
        info = json.loads(path.with_suffix(".json").read_text())
        if info["start"] != next_index or info["sha256"] != digest(path):
            raise ValueError("Encoder batch hash/gap mismatch")
        with np.load(path, allow_pickle=False) as z:
            v = z["vectors"]
        if len(v) != info["stop"] - info["start"]:
            raise ValueError("Encoder row mismatch")
        blocks.append(v)
        next_index = info["stop"]
    if next_index != len(texts):
        raise ValueError("Incomplete encoder cache")
    vectors = np.asarray(np.concatenate(blocks), float)
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(norm < 1e-12):
        raise ValueError("Invalid encoder vectors")
    vectors /= norm
    q, h = vectors[: len(items)], vectors[len(items) :]
    maximum_error = 0.0
    with threadpool_limits(limits=2):
        for start in range(0, len(h), 1024):
            score = h[start : start + 1024] @ q.T
            best = score.argmax(1)
            expected = [items[i]["id"] for i in best]
            selected = evidence.iloc[start : start + len(score)]
            if selected.item_id.tolist() != expected:
                raise ValueError("Cached match is not frozen top-1")
            error = abs(
                score[np.arange(len(score)), best] - selected.similarity.to_numpy()
            )
            maximum_error = max(maximum_error, float(error.max()))
    if maximum_error > 1e-10:
        raise ValueError("Stored top-1 similarities differ")
    return {
        "rows_verified": len(h),
        "max_cosine_error": maximum_error,
        "producer_contract_sha256": digest(producer / "run_contract.json"),
        "encoder_contract_sha256": digest(cache / "contract.json"),
    }


def load_model_annotations(path, packet_ids, expected_sha):
    """Bind independent annotations and their allowed meanings before scoring."""
    if digest(path) != expected_sha:
        raise ValueError("Frozen annotation hash mismatch")
    annotation = json.loads(path.read_text())
    if (
        annotation.get("rater_kind") != "model"
        or annotation.get("rater_model") != "gpt-5.6-sol"
        or annotation.get("scope")
        != "independent_blinded_model_annotation_not_human_gold"
    ):
        raise ValueError("Unexpected annotation provenance")
    frame = pd.DataFrame(annotation["rows"])
    fields = {
        "audit_id",
        "self_attribution",
        "scope",
        "relation",
        "simpa_category",
        "rationale",
    }
    if (
        not fields.issubset(frame)
        or frame.audit_id.duplicated().any()
        or set(frame.audit_id) != set(packet_ids)
    ):
        raise ValueError("Annotation row identity mismatch")
    for column, choices in {
        "self_attribution": {"self", "other", "unclear"},
        "scope": {"general", "episode", "unclear"},
        "relation": {"entails", "contradicts", "neutral", "ambiguous"},
        "simpa_category": set(range(1, 8)),
    }.items():
        if not frame[column].isin(choices).all():
            raise ValueError("Unknown annotation category: " + column)
    return annotation, frame


def literal_confusion(frame):
    """Serialize explicit true/expected rows and predicted columns, never transpose."""
    table = pd.crosstab(frame.literal_expected, frame.nli)
    return {
        "row_axis": "independent_model_expected",
        "column_axis": "nli_predicted",
        "counts": table.to_dict(orient="index"),
    }
