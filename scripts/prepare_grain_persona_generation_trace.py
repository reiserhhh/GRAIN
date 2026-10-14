#!/usr/bin/env python3
"""Freeze materials, aligned persona pairs, splits and budgets for one design."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.persona_trace import (
    DESIGNS,
    NEUTRAL,
    NONPERSONA,
    PERSONAS,
    SHARED_SUFFIX,
    SITUATIONS,
    build_materials,
    digest,
    split_roles,
)

MODEL_DIR = ROOT / "results/grain_vector_mechanisms/local_observers/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
BASE = ROOT / "results/grain_vector_mechanisms/persona_generation_trace_v1"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", choices=list(DESIGNS), default="pilot")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out = (args.output or BASE / args.design).resolve()
    if out.exists():
        raise FileExistsError(out)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, trust_remote_code=False)
    rows, aligned = build_materials(tokenizer, args.design)
    out.mkdir(parents=True)
    manifest = {
        "protocol": "GRAIN_PERSONA_GENERATION_TRACE_V1",
        "design": args.design,
        "spec": DESIGNS[args.design],
        "rows": len(rows),
        "conditions": sorted({r["condition"] for r in rows}),
        "aligned_pairs": aligned,
        "shared_suffix": SHARED_SUFFIX,
        "neutral": NEUTRAL,
        "nonpersona": NONPERSONA,
        "situations": SITUATIONS[: DESIGNS[args.design]["situations"]],
        "split_roles": split_roles(args.design),
        "generation": {"max_new_tokens": 192, "temperature": 0.8, "top_p": 0.95, "do_sample": True},
        "model_dir": str(MODEL_DIR.relative_to(ROOT)),
        "model_files": {n: sha(MODEL_DIR / n) for n in ("config.json", "tokenizer.json", "tokenizer_config.json")},
        "model_weights_sha256_pinned": "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee",
        "code": {p: sha(ROOT / p) for p in ("grain_research/vector_mechanisms/persona_trace.py", "scripts/prepare_grain_persona_generation_trace.py")},
        "materials_identity": digest(rows),
        "persona_alternates": {f"{a}:{p}:{f}": v for (a, p, f), v in PERSONAS.items()},
        "prepared_utc": time.time(),
        "no_outcome_filtering": True,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (out / "materials.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"design": args.design, "rows": len(rows), "aligned": {k: v["span_tokens"] for k, v in aligned.items()}, "output": str(out)}, indent=1))


if __name__ == "__main__":
    main()
