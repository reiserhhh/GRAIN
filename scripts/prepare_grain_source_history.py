#!/usr/bin/env python3
"""Bounded, label-free full-source history support and token-layout registration."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.context import QUESTIONS
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_material import ARMS, masked_comment
from grain_research.vector_mechanisms.inventory import archive_source
from grain_research.vector_mechanisms.ready_comparison import digest, verified_packet
from grain_research.vector_mechanisms.source_history import (
    aligned_triplet,
    profile,
    select_donor,
    split_view,
    tokenize_parts,
)
from scripts.run_grain_vector_mechanism_controls import write_json


def reference_ids(source, budget):
    """Require the original complete99-author dictionary, never infer it from labels."""
    ids = {b["author_id"] for b in source}
    if not budget.old_reference_member.isin([True, False]).all():
        raise ValueError("Reference membership is not boolean")
    if budget.groupby("author_id").old_reference_member.nunique().max() != 1:
        raise ValueError("Inconsistent reference membership across text arms")
    reference = set(budget.loc[budget.old_reference_member, "author_id"]) & ids
    if len(reference) != 99 or any(
        b["source_group"] != "mbti_only" for b in source if b["author_id"] in reference
    ):
        raise ValueError("Expected original99 MBTI-only reference authors")
    return reference


def check_layouts(tokenizer, own, donor, current, *, enclose_separator=False):
    """Build every arm/task without rematching; save commitments, not token arrays."""
    masked = {
        c["comment_id"]: masked_comment(c["body"], c["comment_id"])[0]
        for c in own + donor + [current]
    }
    records = []
    for arm in ARMS:
        h = [c | {"body": masked[c["comment_id"]][arm]} for c in own]
        d = [c | {"body": masked[c["comment_id"]][arm]} for c in donor]
        x = masked[current["comment_id"]][arm]
        previous = None
        for task in QUESTIONS:
            a = tokenize_parts(
                tokenizer, h, x, task, enclose_separator=enclose_separator
            )
            b = tokenize_parts(
                tokenizer, d, x, task, enclose_separator=enclose_separator
            )
            triplet = aligned_triplet(a, b, tokenizer.pad_token_id or 0)
            cur = {
                name: (r["current_positions"], r["current_token_sha256"])
                for name, r in triplet.items()
            }
            if previous is not None and previous != cur:
                raise ValueError("Current prefix layout changes across tasks")
            previous = cur
            for name, row in triplet.items():
                records.append(
                    {
                        "text_arm": arm,
                        "task": task,
                        "history_arm": name,
                        "layout_sha256": material_hash(row),
                        "prompt_tokens": len(row["ids"]),
                        "visible_tokens": row["visible_tokens"],
                        "history_slot_tokens": row["slot"][1] - row["slot"][0],
                        "current_tokens": len(row["current_positions"]),
                        "current_position_sha256": material_hash(
                            row["current_positions"]
                        ),
                        "current_token_sha256": row["current_token_sha256"],
                    }
                )
    return records


def main():
    """Freeze source-only dictionary and all support failures without reading targets."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "views",
        "preflight-dir",
        "profile-dir",
        "qwen-tokenizer",
        "output-dir",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--max-seconds", type=int, default=1800)
    p.add_argument(
        "--history-envelope",
        choices=("body_only", "body_and_separator"),
        default="body_only",
        help="History interval only; does not change native text or tokenization.",
    )
    a = p.parse_args()
    if not 60 <= a.max_seconds <= 1800 or not a.output_dir.resolve().is_relative_to(
        ROOT / "results"
    ):
        raise ValueError("Require bounded private result directory")
    started = time.monotonic()
    source = json.loads(a.views.read_text())
    common = json.loads((a.preflight_dir / "common_material_manifest.json").read_text())
    preflight = json.loads((a.preflight_dir / "manifest.json").read_text())
    if material_hash(source) != common["material_sha256"] or len(
        {b["author_id"] for b in source}
    ) != len(source):
        raise ValueError("Changed common material")
    for name, sha in preflight["qwen_tokenizer_files"].items():
        if digest(a.qwen_tokenizer / name) != sha:
            raise ValueError("Pinned tokenizer file changed")
    tokenizer = AutoTokenizer.from_pretrained(
        str(a.qwen_tokenizer), local_files_only=True, trust_remote_code=False
    )
    if material_hash(tokenizer.chat_template) != preflight["qwen_template_sha256"]:
        raise ValueError("Tokenizer template differs")
    budget = pd.read_csv(
        a.preflight_dir / "budget_by_author.csv", dtype={"author_id": str}
    )
    reference = reference_ids(source, budget)
    pm, _ = verified_packet(a.profile_dir)
    if pm["material_sha256"] != common["material_sha256"]:
        raise ValueError("Profile material differs")
    with np.load(
        a.profile_dir / "private_profile_features.npz", allow_pickle=False
    ) as z:
        profile_ready = {
            uid: bool(ready) for uid, ready in zip(z["author_ids"], z["all_arms_ready"])
        }
    token_count = lambda s: len(tokenizer.encode(s, add_special_tokens=False))
    donors = []
    rejected_donors = []
    for b in source:
        if b["author_id"] not in reference:
            continue
        for vi, view in enumerate(b["views"]):
            try:
                h, _ = split_view(view)
                donors.append(
                    {
                        "author_id": b["author_id"],
                        "view": vi,
                        "history": h,
                        "profile": profile(h, token_count),
                    }
                )
            except ValueError as e:
                rejected_donors.append(
                    {"author_id": b["author_id"], "view": vi, "reason": str(e)}
                )
    targets = [b for b in source if b["author_id"] not in reference]
    a.output_dir.mkdir(parents=True, exist_ok=False)
    fields = {
        "protocol": "VM05_SOURCE_HISTORY_MATERIAL_V1",
        "common_material_sha256": common["material_sha256"],
        "preflight_identity": preflight["identity"],
        "profile_identity": pm["identity"],
        "budget_sha256": digest(a.preflight_dir / "budget_by_author.csv"),
        "reference_authors": len(reference),
        "reference_commitment": material_hash(sorted(reference)),
        "dictionary_views": len(donors),
        "target_authors": len(targets),
        "matching_rule": "raw:log_length+.15log_gap+.5(1-jaccard)",
        "history_envelope": a.history_envelope,
        "qwen_revision": preflight["qwen_revision"],
        "qwen_tokenizer_files": preflight["qwen_tokenizer_files"],
        "pad_length": 3072,
        "labels_read": False,
        "bridge_outcomes_read": False,
        "model_forwards": 0,
        "max_seconds": a.max_seconds,
        "scoring_code": {
            name: digest(ROOT / name)
            for name in (
                "scripts/prepare_grain_source_history.py",
                "grain_research/vector_mechanisms/source_history.py",
                "grain_research/vector_mechanisms/context.py",
                "grain_research/vector_mechanisms/external_material.py",
            )
        },
    }
    fields["identity"] = material_hash(fields)
    write_json(a.output_dir / "manifest.json", fields)
    archive_source(ROOT, a.output_dir / "source_snapshot.zip")
    records = []
    statuses = []
    complete = True
    for i, b in enumerate(targets):
        if time.monotonic() - started > a.max_seconds:
            complete = False
            break
        rows = []
        try:
            all_comments = [c for v in b["views"] for c in v]
            if len(all_comments) != 36 or len(b["views"]) != 3:
                raise ValueError("Not original36 comments")
            for vi, view in enumerate(b["views"]):
                history, current = split_view(view)
                choice = select_donor(
                    profile(history, token_count), current, all_comments, donors
                )
                d = choice["donor"]
                layouts = check_layouts(
                    tokenizer,
                    history,
                    d["history"],
                    current,
                    enclose_separator=a.history_envelope == "body_and_separator",
                )
                rows.append(
                    {
                        "view": vi,
                        "current_id": current["comment_id"],
                        "own_history_ids": [c["comment_id"] for c in history],
                        "donor_author_id": d["author_id"],
                        "donor_view": d["view"],
                        "donor_history_ids": [c["comment_id"] for c in d["history"]],
                        "match": {k: v for k, v in choice.items() if k != "donor"},
                        "layouts": layouts,
                    }
                )
            records.append(
                {
                    "author_id": b["author_id"],
                    "source_group": b["source_group"],
                    "official_folds": b["official_folds"],
                    "views": rows,
                }
            )
            status, reason = "SUPPORTED", ""
        except ValueError as e:
            status, reason = "UNSUPPORTED", str(e)
        statuses.append(
            {
                "author_id": b["author_id"],
                "source_group": b["source_group"],
                "status": status,
                "reason": reason,
                "original_all_arms_ready": profile_ready[b["author_id"]],
            }
        )
        if (i + 1) % 100 == 0:
            progress = {
                "status": "RUNNING",
                "authors_checked": i + 1,
                "supported": len(records),
                "planned": len(targets),
                "seconds": time.monotonic() - started,
                "labels_read": False,
            }
            write_json(a.output_dir / "run_status.json", progress)
            print(json.dumps(progress), flush=True)
    write_json(a.output_dir / "private_history_cases.json", records)
    write_json(a.output_dir / "private_rejected_dictionary_views.json", rejected_donors)
    frame = pd.DataFrame(statuses)
    frame.to_csv(a.output_dir / "private_support_by_author.csv", index=False)
    frame.groupby(["source_group", "status", "reason"], dropna=False).size().rename(
        "authors"
    ).reset_index().to_csv(a.output_dir / "support_summary.csv", index=False)
    reused = Counter(v["donor_author_id"] for r in records for v in r["views"])
    write_json(a.output_dir / "private_donor_reuse.json", dict(reused))
    summary = {
        "status": "COMPLETE" if complete else "PARTIAL_TIME_BOUND",
        "parent_identity": fields["identity"],
        "target_authors": len(targets),
        "authors_checked": len(statuses),
        "supported_authors": len(records),
        "supported_by_group": dict(Counter(r["source_group"] for r in records)),
        "matched_views": 3 * len(records),
        "dictionary_authors_used": len(reused),
        "maximum_dictionary_author_reuse": max(reused.values(), default=0),
        "same_ready_supported_by_group": frame.loc[
            frame.status.eq("SUPPORTED") & frame.original_all_arms_ready
        ]
        .groupby("source_group")
        .size()
        .to_dict(),
        "labels_read": False,
        "bridge_outcomes_read": False,
        "model_forwards": 0,
        "model_weights_loaded": False,
        "seconds": time.monotonic() - started,
        "target_values_read_in_this_run": False,
        "outputs_sha256": {
            f.name: digest(f)
            for f in a.output_dir.iterdir()
            if f.is_file()
            and f.name
            not in ("manifest.json", "run_status.json", "source_snapshot.zip")
            and not f.name.startswith("._")
        },
    }
    write_json(a.output_dir / "summary.json", summary)
    write_json(a.output_dir / "run_status.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
