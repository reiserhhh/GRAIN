#!/usr/bin/env python3
"""Verify a lossless speaker-preserving window design against frozen MEPS material."""

import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.meps_material import (
    load_text_material,
    split_complete_turn,
)
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_meps_preflight import tokenizer_setup


def main():
    """Emit span metadata only; preserve every turn, speaker and source character."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--preflight-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--cache-root", type=Path, default=Path.home() / ".cache/huggingface/hub"
    )
    a = p.parse_args()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(p for p in out.iterdir() if not p.name.startswith("._")):
        raise ValueError("Use a fresh window design directory")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    preflight = json.loads((a.preflight_dir / "manifest.json").read_text())
    complete = json.loads((a.preflight_dir / "summary.json").read_text())
    if (
        preflight["identity"]
        != material_hash({k: v for k, v in preflight.items() if k != "identity"})
        or complete["parent_identity"] != preflight["identity"]
        or complete["status"] != "COMPLETE_MATERIAL_PREFLIGHT_ONLY"
    ):
        raise ValueError("Unsealed source preflight")
    tokenizers, specs = tokenizer_setup(a.cache_root)
    spec = next(s for s in specs if s["name"] == "multilingual_e5_large")
    if spec not in preflight["tokenizers"]:
        raise ValueError("Changed fixed tokenizer")
    tokenizer, prefix = tokenizers["multilingual_e5_large"]
    views, integrity = load_text_material(a.input_root)
    if material_hash(integrity["source_files"]) != preflight["source_identity"]:
        raise ValueError("Changed private MEPS source")
    rows, turn_rows = [], []
    for view in views:
        m = view["metadata"]
        for turn in view["turns"]:
            spans = split_complete_turn(turn["text"], tokenizer, prefix=prefix)
            shared = {k: m[k] for k in ("participant_code", "view_kind", "condition")}
            shared.update(role=turn["role"], turn_sequence=turn["source_sequence"])
            for index, span in enumerate(spans):
                rows.append(
                    {
                        **shared,
                        "window": index,
                        "start": span["start"],
                        "end": span["end"],
                        "tokens": span["tokens"],
                        "text_sha256": hashlib.sha256(
                            span["text"].encode()
                        ).hexdigest(),
                    }
                )
            turn_rows.append(
                {
                    **shared,
                    "source_chars": len(turn["text"]),
                    "covered_chars": sum(s["end"] - s["start"] for s in spans),
                    "windows": len(spans),
                    "source_text_sha256": hashlib.sha256(
                        turn["text"].encode()
                    ).hexdigest(),
                }
            )
    frame, turns = pd.DataFrame(rows), pd.DataFrame(turn_rows)
    frame.to_csv(out / "private_window_metadata.csv", index=False)
    turns.to_csv(out / "private_turn_integrity.csv", index=False)
    table = (
        turns.groupby(["view_kind", "condition", "role"])
        .agg(
            participants=("participant_code", "nunique"),
            turns=("turn_sequence", "size"),
            windows=("windows", "sum"),
            max_windows_per_turn=("windows", "max"),
            source_chars=("source_chars", "sum"),
            covered_chars=("covered_chars", "sum"),
        )
        .reset_index()
    )
    table.to_csv(out / "window_coverage.csv", index=False)
    paths = [
        Path(__file__),
        ROOT / "scripts/run_grain_vector_mechanism_meps_preflight.py",
        ROOT / "grain_research/vector_mechanisms/meps_material.py",
        ROOT / "grain_research/vector_mechanisms/controls.py",
        ROOT / "suica_core/meps_ai_conv.py",
        ROOT / "suica_core/meps_layout.py",
        ROOT / "scripts/run_grain_vector_mechanism_controls.py",
        ROOT / "tests/test_grain_meps_material.py",
    ]
    with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            z.write(path, path.relative_to(ROOT))
    result = {
        "status": "COMPLETE_WINDOW_DESIGN_ONLY",
        "preflight_identity": preflight["identity"],
        "tokenizer": spec,
        "participants": integrity["participants"],
        "views": len(views),
        "turns": len(turns),
        "windows": len(frame),
        "max_window_tokens": int(frame.tokens.max()),
        "source_chars": int(turns.source_chars.sum()),
        "covered_chars": int(turns.covered_chars.sum()),
        "text_loss": int((turns.source_chars - turns.covered_chars).sum()),
        "speaker_boundaries_crossed": 0,
        "labels_read": False,
        "model_forwards": 0,
        "equivalent_to_full_dialogue_context": False,
        "raw_text_or_token_ids_persisted": False,
        "code": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
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
