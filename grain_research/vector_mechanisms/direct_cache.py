#!/usr/bin/env python3
"""Independent reconstruction of saved direct readouts; no model or target values."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .controls import material_hash
from .ready_comparison import verified_packet


def verify_author(data, case, alphabets):
    """Recompute output arithmetic from raw candidate logits and vocabulary normalizer.

    This checks saved calculations, not the truth of the model logits; native
    repeated forwards are a separate instrument test. Full-vocabulary probability
    mass needs the saved log normalizer and is never inferred from conditional q.
    """
    rows = data["readouts"]
    expected = case["layouts"]
    if (
        data["author_id"] != case["author_id"]
        or data["source_group"] != case["source_group"]
        or data["case_commitment"] != material_hash(case)
        or len(rows) != len(expected)
        or data["metrics"]["logical_prompts"] != len(expected)
        or data["metrics"]["alphabets"] != {t: alphabets[t] for t in case["tasks"]}
    ):
        raise ValueError("Direct author/task/grid identity changed")
    errors, unique = [], set()
    for row, layout in zip(rows, expected, strict=True):
        if any(
            row[k] != layout[k]
            for k in ("arm", "view", "task", "prompt_tokens", "prompt_sha256")
        ):
            raise ValueError("Direct registered row changed")
        unique.add(row["prompt_sha256"])
        a = alphabets[row["task"]]
        z = np.asarray(row["candidate_logits"], dtype=float)
        v = np.asarray(a["values"], dtype=float)
        normalizer = row["full_vocab_log_normalizer"]
        if (
            z.shape != v.shape
            or not np.isfinite(z).all()
            or not np.isfinite(normalizer)
        ):
            raise ValueError("Invalid saved direct logits")
        weights = np.exp(z - z.max())
        q = weights / weights.sum()
        local_norm = z.max() + np.log(weights.sum())
        log_mass = float(local_norm - normalizer)
        if log_mass > 1e-10:
            raise ValueError("Candidate mass exceeds full vocabulary mass")
        computed = {
            "conditional_score": float(q @ v),
            "conditional_probabilities": q,
            "conditional_entropy": float(-np.sum(q[q > 0] * np.log(q[q > 0]))),
            "candidate_log_mass": log_mass,
            "candidate_mass": float(np.exp(log_mass)),
        }
        for key, value in computed.items():
            actual = np.asarray(row[key], dtype=float)
            reference = np.asarray(value)
            if actual.shape != reference.shape or not np.isfinite(actual).all():
                raise ValueError("Invalid saved direct result")
            error = float(np.max(np.abs(actual - reference)))
            if error > 1e-10:
                raise ValueError("Saved direct result cannot be reconstructed")
            errors.append(error)
        top = int(z.argmax())
        greedy = row["greedy_first_token_id"]
        if (
            row["candidate_argmax_label"] != a["labels"][top]
            or row["candidate_argmax_value"] != v[top]
            or row["greedy_first_token_is_candidate"] != (greedy in a["token_ids"])
            or (greedy in a["token_ids"] and z[a["token_ids"].index(greedy)] != z[top])
            or row["complete_response_parse_rate_measured"] is not False
        ):
            raise ValueError("Saved direct answer/compliance changed")
    if data["metrics"]["unique_prompts"] != len(unique):
        raise ValueError("Wrong exact-input reuse count")
    means = {
        f"{arm}__{task}": float(
            np.mean(
                [
                    r["conditional_score"]
                    for r in rows
                    if r["arm"] == arm and r["task"] == task
                ]
            )
        )
        for arm in ("raw", "cue_mask", "placebo_mask")
        for task in case["tasks"]
    }
    if data["view_means"] != means:
        raise ValueError("Direct view aggregation differs")
    return {"rows": len(rows), "unique_prompts": len(unique), "max_error": max(errors)}


def verify_collection(folder, cases, material, code):
    """Reject omitted cases, output mutation and arithmetically inconsistent caches."""
    folder = Path(folder)
    m, s = verified_packet(folder)
    expected_ids = [c["author_id"] for c in cases]
    if (
        m["protocol"] != "VM05_DIRECT_TRAIT_COLLECTION_V1"
        or m["material_identity"] != material["identity"]
        or m["case_commitment"] != material_hash(cases)
        or m["authors"] != expected_ids
        or m["code_sha256"] != code
        or not s["real_neural_collection"]
        or s["authors"] != len(cases)
        or m["labels_read"]
        or m["bridge_outcomes_read"]
        or s["labels_read"]
        or s["bridge_outcomes_read"]
    ):
        raise ValueError("Unqualified or changed direct collection")
    keys = [material_hash(c["author_id"]) for c in cases]
    records = json.loads((folder / "private_records.json").read_text(encoding="utf-8"))
    if {
        p.stem
        for p in (folder / "private_cache").glob("*.json")
        if not p.name.startswith("._")
    } != set(keys) or [r["cache_key"] for r in records] != keys:
        raise ValueError("Direct cache membership incomplete")
    results, forward_sum = [], 0
    for c, key, record in zip(cases, keys, records, strict=True):
        rel = f"private_cache/{key}.json"
        if s["outputs_sha256"].get(rel) != record["sha256"]:
            raise ValueError("Direct record not bound to complete cache")
        data = json.loads((folder / rel).read_text(encoding="utf-8"))
        result = verify_author(data, c, material["alphabets"])
        if record["logical_prompts"] != result["rows"]:
            raise ValueError("Direct record count differs")
        results.append(result)
        forward_sum += data["metrics"]["forwards"]
    if s["logical_prompts"] != sum(r["rows"] for r in results) or s[
        "forwards"
    ] != forward_sum * (2 if m["mode"] == "qualification" else 1):
        raise ValueError("Direct total forward/grid count differs")
    return {
        "authors": len(results),
        "rows": s["logical_prompts"],
        "max_error": max(r["max_error"] for r in results),
    }
