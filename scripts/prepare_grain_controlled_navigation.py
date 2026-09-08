#!/usr/bin/env python3
"""Freeze NAV-01 synthetic worlds, query coordinates and reference provenance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import (
    BASE,
    KEYS,
    MODEL,
    OUTPUT,
    QUERIES,
    build_worlds,
    digest,
    read_json,
    real_reference_sources,
    reference_world_ids,
    tokenize_world,
    write_json,
)


def prepare(out):
    """Use only synthetic materials and frozen coefficient metadata, never human labels."""
    from transformers import AutoTokenizer

    if not out.resolve().is_relative_to((BASE / "navigation_v1").resolve()):
        raise ValueError("Use the ignored navigation output root")
    if out.exists():
        raise FileExistsError("Choose a fresh run ID; do not overwrite frozen design")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    sources = real_reference_sources()
    core = BASE / "all_factor_function_analysis_m28_v2"
    narrow = BASE / "narrow_functions_m30b_v1"
    if (
        digest(sources["extraversion"])
        != read_json(core / "summary.json")["functions_sha256"]
    ):
        raise ValueError("Core reference archive changed")
    if (
        digest(sources["adapted_dejected5"])
        != read_json(narrow / "reference_manifest.json")["sha256"]
    ):
        raise ValueError("Narrow reference archive changed")
    manifests = {
        "extraversion": BASE / "all_factor_functions_m28_v1/manifest.json",
        "TF_cont": BASE / "complete_level_response_v1/manifest.json",
        "adapted_dejected5": narrow / "manifest.json",
    }
    expected = {q: read_json(p)["question_tokens"][q] for q, p in manifests.items()}
    model_sha = digest(MODEL / "model.safetensors")
    if any(read_json(p)["model_sha256"] != model_sha for p in manifests.values()):
        raise ValueError("Frozen readers use a different model")
    worlds, design, coefficients = build_worlds(), [], {}
    for world in worlds:
        for q in QUERIES:
            _, _, encoding, mask, alphabet = tokenize_world(
                tokenizer, world["texts"], q
            )
            if np.asarray(encoding["input_ids"])[mask].tolist() != expected[q]:
                raise ValueError(
                    "Question-token identity differs from completed functions"
                )
            design.append(
                {
                    "index": len(design),
                    "world_id": world["world_id"],
                    "query": q,
                    "input_ids": encoding["input_ids"],
                    "mask": mask.tolist(),
                    "alphabet": alphabet,
                }
            )
    for q, source in sources.items():
        with np.load(source, allow_pickle=False) as z:
            coefficients.update({f"{q}__{k}": z[f"{q}__{k}"] for k in KEYS})
    out.mkdir(parents=True)
    (out / "packets").mkdir()
    (out / "synthetic_reference_units").mkdir()
    write_json(out / "worlds.json", worlds)
    write_json(out / "design.json", design)
    np.savez_compressed(out / "real_reference.npz", **coefficients)
    code = (
        "grain_research/vector_mechanisms/navigation.py",
        "scripts/prepare_grain_controlled_navigation.py",
        "scripts/run_grain_navigation_baseline.py",
        "grain_research/vector_mechanisms/factor_functions.py",
        "grain_research/vector_mechanisms/query_functional.py",
        "grain_research/vector_mechanisms/full_state_edit.py",
    )
    provenance_paths = (
        set(sources.values()) | set(manifests.values()) | {narrow / "queries.json"}
    )
    write_json(
        out / "manifest.json",
        {
            "protocol": "GRAIN_CONTROLLED_NAVIGATION_V1",
            "worlds": 96,
            "native_units": 288,
            "queries": list(QUERIES),
            "question_tokens": expected,
            "model_sha256": model_sha,
            "model_path": str(MODEL.relative_to(ROOT)),
            "worlds_sha256": digest(out / "worlds.json"),
            "design_sha256": digest(out / "design.json"),
            "real_reference_sha256": digest(out / "real_reference.npz"),
            "synthetic_references": reference_world_ids(),
            "reference_rule": "two construction scenes per target, self/voluntary/renderer0; remove action excerpt",
            "source_sha256": {
                str(p.relative_to(ROOT)): digest(p) for p in provenance_paths
            },
            "code_sha256": {p: digest(ROOT / p) for p in code},
            "plan_sha256": digest(ROOT / "docs/GRAIN_NAVIGATION_RESEARCH_GOAL_V6.md"),
            "neural_calls_planned": 342,
            "per_packet_seconds": 1200,
            "meps_construction": False,
            "human_label_fit": False,
            "device": "mps",
            "dtype": "float32",
            "block": 16,
            "split": "6construction/3development/3held_scene; renderer vocabulary shared",
        },
    )
    write_json(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "calls": 0,
            "seconds": 0,
            "completed_units": 0,
            "completed_references": 0,
        },
    )
    print(
        {
            "status": "PREPARED",
            "worlds": 96,
            "native_units": 288,
            "synthetic_references": 6,
            "max_tokens": max(len(d["input_ids"]) for d in design),
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    prepare(parser.parse_args().output_dir.resolve())
