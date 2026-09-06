#!/usr/bin/env python3
"""Prepare role-separated MEPS material and audit local tokenizer coverage only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.meps_material import (
    load_text_material,
    token_audit,
)
from scripts.run_grain_vector_mechanism_controls import write_json

MODELS = (
    (
        "bge_large",
        "BAAI/bge-large-en-v1.5",
        "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        "",
    ),
    (
        "e5_large",
        "intfloat/e5-large-v2",
        "f169b11e22de13617baa190a028a32f3493550b6",
        "query: ",
    ),
    (
        "multilingual_e5_large",
        "intfloat/multilingual-e5-large",
        "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
        "query: ",
    ),
)


def tokenizer_setup(cache_root):
    """Require all frozen tokenizer files locally before reading private text."""
    from transformers import AutoTokenizer

    loaded, records = {}, []
    for name, model, revision, prefix in MODELS:
        snapshot = (
            cache_root
            / ("models--" + model.replace("/", "--"))
            / "snapshots"
            / revision
        )
        if not snapshot.is_dir():
            raise ValueError(
                "Frozen tokenizer snapshot is absent; no download attempted"
            )
        files = sorted(
            p
            for p in snapshot.iterdir()
            if p.is_file()
            and p.name
            in {
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "sentencepiece.bpe.model",
                "vocab.txt",
                "config.json",
                "added_tokens.json",
            }
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False
        )
        loaded[name] = (tokenizer, prefix)
        records.append(
            {
                "name": name,
                "model": model,
                "revision": revision,
                "prefix": prefix,
                "files": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files
                },
            }
        )
    return loaded, records


def coverage_tables(views, token_rows):
    """Reduce private row-level audits to aggregate coverage, not author scores."""
    metadata = pd.DataFrame([v["metadata"] for v in views])
    coverage = (
        metadata.groupby(["view_kind", "condition"], sort=True)
        .agg(
            views=("participant_code", "size"),
            participants=("participant_code", "nunique"),
            total_user_chars=("user_chars", "sum"),
            median_user_chars=("user_chars", "median"),
            total_assistant_chars=("assistant_chars", "sum"),
            median_user_turns=("user_turns", "median"),
            median_assistant_turns=("assistant_turns", "median"),
            ai_assistance_possible=("ai_assistance_possible", "sum"),
        )
        .reset_index()
    )
    tokens = pd.DataFrame(token_rows)
    token_summary = (
        tokens.groupby(["model", "view_kind", "condition", "arm"], sort=True)
        .agg(
            views=("participant_code", "size"),
            min_tokens=("tokens", "min"),
            median_tokens=("tokens", "median"),
            max_tokens=("tokens", "max"),
            complete_under_512=("within_complete_view_budget", "sum"),
            unknown_tokens=("unknown_tokens", "sum"),
            content_tokens=("content_tokens", "sum"),
        )
        .reset_index()
    )
    token_summary["unknown_token_fraction"] = (
        token_summary.unknown_tokens / token_summary.content_tokens
    )
    paired = []
    for model in sorted(tokens.model.unique()):
        for arm in ("user_only", "assistant_only", "full_dialogue"):
            eligible = tokens.loc[
                (tokens.model == model)
                & (tokens.arm == arm)
                & tokens.within_complete_view_budget
            ]
            direct = {
                q: set(
                    eligible.loc[
                        (eligible.view_kind == "direct_answer")
                        & (eligible.condition == q),
                        "participant_code",
                    ]
                )
                for q in ("meps_q1", "meps_q2", "meps_q3")
            }
            free = set(
                eligible.loc[eligible.view_kind == "free_chat", "participant_code"]
            )
            split = {
                q: set(
                    eligible.loc[
                        (eligible.view_kind == "split_chat")
                        & (eligible.condition == q),
                        "participant_code",
                    ]
                )
                for q in direct
            }
            paired.append(
                {
                    "model": model,
                    "arm": arm,
                    "complete_three_direct": len(set.intersection(*direct.values())),
                    "complete_direct_plus_free": len(
                        set.intersection(*direct.values()) & free
                    ),
                    "complete_three_assistance_chats": len(
                        set.intersection(*split.values())
                    ),
                    "complete_assistance_plus_free": len(
                        set.intersection(*split.values()) & free
                    ),
                }
            )
    return metadata, coverage, tokens, token_summary, pd.DataFrame(paired)


def main():
    """Create a fresh, content-free preflight; never fit an observer or read labels."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument(
        "--cache-root", type=Path, default=Path.home() / ".cache/huggingface/hub"
    )
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    start = time.monotonic()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(p for p in out.iterdir() if not p.name.startswith("._")):
        raise ValueError("Use a fresh MEPS preflight directory")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    tokenizers, model_records = tokenizer_setup(a.cache_root)
    views, integrity = load_text_material(a.input_root)
    token_rows = []
    for name, (tokenizer, prefix) in tokenizers.items():
        token_rows.extend(token_audit(views, tokenizer, model_name=name, prefix=prefix))
    metadata, coverage, tokens, token_summary, paired = coverage_tables(
        views, token_rows
    )
    metadata.to_csv(out / "private_view_metadata.csv", index=False)
    pd.DataFrame(integrity["chat_coverage"]).to_csv(
        out / "private_chat_integrity.csv", index=False
    )
    tokens.to_csv(out / "private_token_metadata.csv", index=False)
    coverage.to_csv(out / "view_coverage.csv", index=False)
    role = coverage.loc[coverage.view_kind != "direct_answer"].copy()
    role["assistant_share_of_role_chars"] = role.total_assistant_chars / (
        role.total_assistant_chars + role.total_user_chars
    )
    role.to_csv(out / "role_coverage.csv", index=False)
    token_summary.to_csv(out / "token_coverage.csv", index=False)
    paired.to_csv(out / "paired_support.csv", index=False)
    write_json(out / "source_integrity.json", integrity)
    code_paths = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/meps_material.py",
        ROOT / "suica_core/meps_ai_conv.py",
        ROOT / "suica_core/meps_layout.py",
        ROOT / "grain_research/vector_mechanisms/controls.py",
        ROOT / "scripts/run_grain_vector_mechanism_controls.py",
        ROOT / "tests/test_grain_meps_material.py",
        ROOT / "tests/test_meps_ai_conv.py",
    ]
    with zipfile.ZipFile(
        out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in code_paths:
            archive.write(path, path.relative_to(ROOT))
    manifest = {
        "protocol": "GRAIN_VM06_ROLE_MATERIAL_PREFLIGHT_V1",
        "participants": integrity["participants"],
        "text_views": len(views),
        "tokenizer_conditions": len(tokens),
        "source_identity": material_hash(integrity["source_files"]),
        "tokenizers": model_records,
        "access_policy": integrity["access_policy"],
        "encoder_weights_loaded": False,
        "views_truncated": 0,
        "psychological_outcomes_used": False,
        "code": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in code_paths
        },
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    result = {
        "status": "COMPLETE_MATERIAL_PREFLIGHT_ONLY",
        "parent_identity": manifest["identity"],
        "participants": integrity["participants"],
        "views": len(views),
        "tokenizer_conditions": len(tokens),
        "seconds": time.monotonic() - start,
        "model_forwards": 0,
        "labels_read": False,
        "vm06_frozen_mechanism_validation_complete": False,
        "outputs_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in out.iterdir()
            if p.is_file() and not p.name.startswith("._")
        },
    }
    write_json(out / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
