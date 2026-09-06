#!/usr/bin/env python3
"""Full direct-readout shard and semantic-parent checks without fitting any target."""

import json
from pathlib import Path

import numpy as np

from .controls import material_hash
from .direct_cache import verify_collection
from .direct_semantics import panel, prepare, summarize
from .external_material import shard_material
from .history_collection import observation_signature
from .ready_comparison import digest, verified_packet


def semantic_parent(folder, material, qualification, signature, code, tokenizer):
    """Require a complete real panel, not positive outcomes or a chosen prompt variant."""
    if folder is None:
        raise ValueError("Real direct queue requires complete semantic panel")
    folder = Path(folder)
    m, s = verified_packet(folder)
    cases = panel()
    if (
        m["protocol"] != "VM05_DIRECT_SEMANTICS_V1"
        or m["preflight_only"]
        or m["panel_sha256"] != material_hash(cases)
        or m["material_identity"] != material["identity"]
        or m["qualification_identity"] != qualification
        or m["qualified_model_signature"] != signature
        or m["code_sha256"] != code
        or not s["real_neural_collection"]
        or s["cases"] != 90
        or s["forwards"] != 180
        or m["labels_read"]
        or m["bridge_outcomes_read"]
        or s["labels_read"]
        or s["bridge_outcomes_read"]
        or digest(folder / "source_snapshot.zip") != m["source_snapshot_sha256"]
    ):
        raise ValueError("Semantic panel is incomplete or belongs to another observer")
    model = json.loads((folder / "actual_observer.json").read_text(encoding="utf-8"))
    if observation_signature(model) != signature:
        raise ValueError("Semantic observer changed")
    rows = json.loads((folder / "observations.json").read_text(encoding="utf-8"))
    contrasts, tasks = summarize(rows)
    maximum = 0.0
    for c, r in zip(cases, rows, strict=True):
        ids, alphabet = prepare(c, tokenizer)
        if (
            r["prompt_sha256"] != material_hash(ids)
            or r["prompt_tokens"] != len(ids)
            or r["alphabet"] != alphabet
        ):
            raise ValueError("Semantic native prompt or code meaning changed")
        z = np.asarray(r["candidate_logits"], dtype=float)
        normalizer = r["full_vocab_log_normalizer"]
        if (
            z.shape != (len(alphabet["values"]),)
            or not np.isfinite(z).all()
            or not np.isfinite(normalizer)
        ):
            raise ValueError("Invalid semantic native logits")
        weights = np.exp(z - z.max())
        q = weights / weights.sum()
        log_mass = z.max() + np.log(weights.sum()) - normalizer
        expected = {
            "conditional_score": q @ alphabet["values"],
            "conditional_probabilities": q,
            "conditional_entropy": -np.sum(q[q > 0] * np.log(q[q > 0])),
            "candidate_log_mass": log_mass,
            "candidate_mass": np.exp(log_mass),
        }
        if log_mass > 1e-10:
            raise ValueError("Semantic candidate mass exceeds total mass")
        for name, value in expected.items():
            actual = np.asarray(r[name], dtype=float)
            if actual.shape != np.asarray(value).shape or not np.isfinite(actual).all():
                raise ValueError("Malformed semantic readout")
            maximum = max(maximum, float(np.max(abs(actual - value))))
    if maximum > 1e-10:
        raise ValueError("Semantic readout arithmetic changed")
    if (
        json.loads((folder / "semantic_contrasts.json").read_text(encoding="utf-8"))
        != contrasts
        or json.loads((folder / "task_sensitivity.json").read_text(encoding="utf-8"))
        != tasks
    ):
        raise ValueError("Semantic summary changed")
    return {
        "identity": m["identity"],
        "max_arithmetic_error": maximum,
        "task_sensitivity": tasks,
        "positive_pairs_not_a_gate": True,
    }


def verify_shard(
    folder,
    whole,
    material,
    code,
    count,
    index,
    *,
    preflight,
    qualification=None,
    model_signature=None,
):
    """Verify exact shard membership, qualified observer and every saved direct readout."""
    folder = Path(folder)
    m, s = verified_packet(folder)
    cases = shard_material(whole, count, index)
    if (
        m["mode"] != ("preflight" if preflight else "source_shard")
        or m["protocol"] != "VM05_DIRECT_TRAIT_COLLECTION_V1"
        or m["material_identity"] != material["identity"]
        or m["case_commitment"] != material_hash(cases)
        or m["authors"] != [c["author_id"] for c in cases]
        or m["supported_authors"] != len(whole)
        or m["shard_count"] != count
        or m["shard_index"] != index
        or m["qualification_identity"] != qualification
        or m["code_sha256"] != code
        or s["authors"] != len(cases)
        or s["real_neural_collection"] != (not preflight)
        or s["model_signature"] != model_signature
        or s["all_qualification_rows_replayed"]
        or s["replay_max"] != 0
        or m["labels_read"]
        or m["bridge_outcomes_read"]
        or s["labels_read"]
        or s["bridge_outcomes_read"]
        or digest(folder / "source_snapshot.zip") != m["source_snapshot_sha256"]
        or json.loads((folder / "run_status.json").read_text(encoding="utf-8")) != s
    ):
        raise ValueError("Direct shard mode, parent, support or observer mismatch")
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
        raise ValueError("Uncommitted or missing direct shard outputs")
    records = json.loads((folder / "private_records.json").read_text(encoding="utf-8"))
    logical = sum(len(c["layouts"]) for c in cases)
    if s["logical_prompts"] != logical:
        raise ValueError("Direct logical grid count changed")
    if preflight:
        expected = [
            {
                "case_commitment": material_hash(c),
                "logical_prompts": len(c["layouts"]),
                "token_replay": True,
            }
            for c in cases
        ]
        if (
            records != expected
            or files != {"private_records.json"}
            or s["forwards"]
            or s["cache_bytes"]
        ):
            raise ValueError("Preflight must be complete token-only evidence")
        return {
            "authors": len(cases),
            "logical_prompts": logical,
            "forwards": 0,
            "cache_bytes": 0,
            "max_arithmetic_error": 0.0,
            "real_neural_collection": False,
        }
    result = verify_collection(folder, cases, material, code)
    model = json.loads((folder / "actual_observer.json").read_text(encoding="utf-8"))
    if observation_signature(model) != model_signature:
        raise ValueError("Direct actual observer differs from real qualification")
    cache_paths = [
        folder / "private_cache" / (material_hash(c["author_id"]) + ".json")
        for c in cases
    ]
    expected_files = {p.relative_to(folder).as_posix() for p in cache_paths} | {
        "private_records.json",
        "actual_observer.json",
    }
    if files != expected_files or s["cache_bytes"] != sum(
        p.stat().st_size for p in cache_paths
    ):
        raise ValueError("Direct cache size or file set changed")
    unique = 0
    for i, path in enumerate(cache_paths):
        row = json.loads(path.read_text(encoding="utf-8"))
        meta = row["metrics"]
        if (
            meta["forwards"] != meta["unique_prompts"] + int(i == 0)
            or meta["replay_max"] != 0
        ):
            raise ValueError("Direct native forward/replay count changed")
        unique += meta["unique_prompts"]
    if s["forwards"] != unique + 1 or not 0 < s["cache_bytes"] <= m["max_cache_bytes"]:
        raise ValueError("Direct total native work or disk bound violated")
    return {
        "authors": len(cases),
        "logical_prompts": logical,
        "forwards": s["forwards"],
        "cache_bytes": s["cache_bytes"],
        "max_arithmetic_error": result["max_error"],
        "real_neural_collection": True,
    }
