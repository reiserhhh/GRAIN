#!/usr/bin/env python3
"""Recover HM150 token preparation without editing its sealed failed runner."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_joint_persona_style_hm150 as R

FAILED = R.DEFAULT
DEFAULT = ROOT / "results/grain_math_response_v1/joint_persona_style_r2"


def flat_ids(encoded):
    """Normalize supported tokenizer return types; never convert keys to tokens."""
    values = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if not values or not all(isinstance(v, int) for v in values):
        raise ValueError("Actual flat integer token IDs required")
    return list(values)


def prepare(out):
    """Use the identical full experiment with corrected cached tokenizer IDs."""
    from transformers import AutoTokenizer

    failed_manifest = R.check(FAILED)
    if R.H.load(FAILED / "generation_status.json")["status"] != "FAILED":
        raise ValueError("Expected terminal preparation-format failure")
    if list((FAILED / "batches").glob("*.json")):
        raise ValueError("Recovery requires no scientific generation in failed run")
    old = R.H.load(R.H.SOURCE / "manifest.json")
    rows = R.M.joint_materials(old)
    tokenizer = AutoTokenizer.from_pretrained(
        str(R.MODEL), local_files_only=True, trust_remote_code=False
    )
    for row in rows:
        messages = [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": row["user"]},
        ]
        row["prompt_ids"] = flat_ids(
            tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        )
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if row["prompt_ids"] != tokenizer.encode(rendered, add_special_tokens=False):
            raise ValueError("Prompt IDs disagree with actual rendered template")
        if len(row["prompt_ids"]) > 384:
            raise ValueError("Prompt exceeds unchanged fixed support")
    out.mkdir(parents=True, exist_ok=False)
    (out / "batches").mkdir()
    (out / "observations").mkdir()
    R.H.write_json(out / "materials.json", rows)
    added = (
        "scripts/prepare_grain_joint_persona_style_hm150_recovery.py",
        "tests/test_grain_joint_persona_style_recovery.py",
        "docs/GRAIN_JOINT_PERSONA_STYLE_HM150_RECOVERY.md",
    )
    hashes = dict(failed_manifest["hashes"])
    hashes.update({name: R.H.digest(ROOT / name) for name in added})
    R.H.write_json(
        out / "manifest.json",
        {
            **failed_manifest,
            "hashes": hashes,
            "materials_sha256": R.H.digest(out / "materials.json"),
            "recovery": {
                "failed_directory": str(FAILED.relative_to(ROOT)),
                "failed_manifest_sha256": R.H.digest(FAILED / "manifest.json"),
                "reason": "Tokenizer Mapping was converted to string keys;correct flat input_ids and template replay.0 prior generated articles",
                "scientific_design_changed": False,
            },
        },
    )
    print(
        f"Recovered {len(rows)} actual integer-token prompts;no scientific design change",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    prepare(parser.parse_args().output)
