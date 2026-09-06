#!/usr/bin/env python3
"""Independent history-cache integrity, completeness and causal-invariant checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .context import QUESTIONS
from .controls import material_hash
from .external_material import ARMS, shard_material, unpack_features
from .history_collection import HISTORIES, KINDS, SITES, observation_signature
from .ready_comparison import digest, verified_packet


def verify_history_shard(
    folder,
    whole,
    material,
    audit,
    code,
    count,
    index,
    *,
    qualification=None,
    technical=None,
    model_signature=None,
    allow_preflight=False,
):
    """Reject missing, changed, re-signed malformed or token-only neural evidence.

    Parent identities and the expected author order are supplied by the audited
    registry, not inferred from the artifact under inspection. This verifies
    saved observations; it is not a second model execution.
    """
    folder = Path(folder)
    m, s = verified_packet(folder)
    cases = shard_material(whole, count, index)
    ids = [c["author_id"] for c in cases]
    preflight = m["mode"] == "preflight"
    if (
        m["protocol"] != "VM05_HISTORY_COLLECTION_V1"
        or m["mode"] not in ("preflight", "source_shard")
        or (preflight and not allow_preflight)
        or m["material_identity"] != material["identity"]
        or m["audit_identity"] != audit["identity"]
        or m["qualification_identity"] != qualification
        or m["technical_identity"] != technical
        or m["strict_author_commitment"]
        != material_hash([c["author_id"] for c in whole])
        or m["strict_authors"] != len(whole)
        or m["authors"] != ids
        or m["case_commitment"] != material_hash(cases)
        or m["shard_count"] != count
        or m["shard_index"] != index
        or m["code_sha256"] != code
        or tuple(m["sites"]) != SITES
        or tuple(m["readout_kinds"]) != KINDS
        or m["diagnostic_margin_is_direct_trait_score"]
        or m["feature_rows"] != ["decision", "current_mean", "current_std"]
        or m["labels_read"]
        or m["bridge_outcomes_read"]
        or s["labels_read"]
        or s["bridge_outcomes_read"]
        or s["authors"] != len(ids)
        or s["strict_authors"] != len(whole)
        or s["token_layouts_replayed"] != 54 * len(ids)
        or s["real_neural_collection"] != (not preflight)
        or s["qualified_state_replay_comparison_groups"] != 0
        or s["qualified_state_replay_max"] is not None
        or s["model_signature"] != model_signature
        or digest(folder / "source_snapshot.zip") != m["source_snapshot_sha256"]
    ):
        raise ValueError("History shard protocol/parent/registry mismatch")
    status = json.loads((folder / "run_status.json").read_text(encoding="utf-8"))
    if status != s:
        raise ValueError("History completion status differs from summary")
    files = {
        p.relative_to(folder).as_posix()
        for p in folder.rglob("*")
        if p.is_file()
        and not p.name.startswith("._")
        and p.name
        not in (
            "manifest.json",
            "summary.json",
            "run_status.json",
            "source_snapshot.zip",
        )
    }
    if files != set(s["outputs_sha256"]):
        raise ValueError("History outputs missing or uncommitted")
    records = json.loads((folder / "private_records.json").read_text(encoding="utf-8"))
    if len(records) != len(cases):
        raise ValueError("Missing history author records")
    if preflight:
        if files != {"private_records.json"} or s["forwards"] or s["cache_bytes"]:
            raise ValueError("Preflight contains neural evidence")
        for c, r in zip(cases, records, strict=True):
            if r["case_commitment"] != material_hash(c) or r["layouts"] != 54:
                raise ValueError("Preflight case changed")
        unique = sum(r["unique_layouts"] for r in records)
        if unique != s["unique_layouts"]:
            raise ValueError("Preflight layout count changed")
        return {
            "authors": len(ids),
            "real_neural_collection": False,
            "unique_layouts": unique,
            "cache_bytes": 0,
            "arrays": 0,
        }
    model = json.loads((folder / "actual_observer.json").read_text(encoding="utf-8"))
    if observation_signature(model) != model_signature:
        raise ValueError("Actual history model signature changed")
    expected_files = {"private_records.json", "actual_observer.json"}
    expected_names = {
        f"{a}__{t}__{h}__{site}"
        for a in ARMS
        for t in QUESTIONS
        for h in HISTORIES
        for site in SITES
    }
    unique = calls = cache_bytes = matrices = 0
    for ci, (case, record) in enumerate(zip(cases, records, strict=True)):
        key = material_hash(case["author_id"])
        npz_path = folder / "private_cache" / (key + ".npz")
        meta_path = folder / "private_cache" / (key + ".json")
        expected_files.update(
            p.relative_to(folder).as_posix() for p in (npz_path, meta_path)
        )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            record["cache_key"] != key
            or record["author_id"] != case["author_id"]
            or meta["author_id"] != case["author_id"]
            or meta["source_group"] != case["source_group"]
            or meta["case_commitment"] != material_hash(case)
            or digest(npz_path) != record["npz_sha256"]
            or digest(npz_path) != meta["feature_sha256"]
            or digest(meta_path) != record["json_sha256"]
            or meta["cache_format"] != "lossless_history_aliases_v1"
            or meta["logical_layouts"] != 54
            or any(
                meta[k] != 0
                for k in ("replay_max", "causal_prefix_max", "input_embedding_max")
            )
            or meta["unique_layouts"] != record["unique_layouts"]
            or meta["forwards"] != meta["unique_layouts"] + int(ci == 0)
        ):
            raise ValueError("History author provenance or instrument mismatch")
        rows = meta["readouts"]
        row_keys = {
            (r["view"], r["text_arm"], r["task"], r["history_arm"]) for r in rows
        }
        if (
            len(rows) != 54
            or row_keys
            != {
                (v, a, t, h)
                for v in range(3)
                for a in ARMS
                for t in QUESTIONS
                for h in HISTORIES
            }
            or len({r["layout_sha256"] for r in rows}) != meta["unique_layouts"]
        ):
            raise ValueError("Incomplete history logical readouts")
        for r in rows:
            if (
                not np.isfinite([r["yes_logit"], r["no_logit"], r["margin"]]).all()
                or r["margin"] != r["yes_logit"] - r["no_logit"]
                or not 0
                < r["current_tokens"]
                <= r["visible_tokens"]
                <= r["prompt_tokens"]
                <= 3056
            ):
                raise ValueError("History diagnostic readout changed")
        with np.load(npz_path, allow_pickle=False) as z:
            arrays = unpack_features(
                {k: z[k] for k in z.files}, meta["feature_aliases"]
            )
        if set(arrays) != expected_names or any(
            x.shape != (3, 3, 1536)
            or x.dtype != np.float32
            or not np.isfinite(x).all()
            or np.any(x[:, 2] < 0)
            for x in arrays.values()
        ):
            raise ValueError("History arrays are not complete pinned-Qwen states")
        for arm in ARMS:
            for task in QUESTIONS:
                ref = arrays[f"{arm}__{task}__own__input_embedding"]
                if any(
                    not np.array_equal(
                        ref, arrays[f"{arm}__{task}__{h}__input_embedding"]
                    )
                    for h in HISTORIES
                ):
                    raise ValueError(
                        "Saved current/decision embedding changed with history"
                    )
            for h in HISTORIES:
                for site in SITES:
                    if not np.array_equal(
                        arrays[f"{arm}__behavior__{h}__{site}"][:, 1:],
                        arrays[f"{arm}__disposition__{h}__{site}"][:, 1:],
                    ):
                        raise ValueError(
                            "Saved causal prefix changed with later question"
                        )
        unique += meta["unique_layouts"]
        calls += meta["forwards"]
        cache_bytes += npz_path.stat().st_size + meta_path.stat().st_size
        matrices += len(arrays)
    if (
        files != expected_files
        or unique != s["unique_layouts"]
        or calls != s["forwards"]
        or cache_bytes != s["cache_bytes"]
        or cache_bytes > m["max_cache_bytes"]
    ):
        raise ValueError("Incomplete history cache/accounting")
    return {
        "authors": len(ids),
        "real_neural_collection": True,
        "arrays": matrices,
        "unique_layouts": unique,
        "forwards": calls,
        "cache_bytes": cache_bytes,
    }
