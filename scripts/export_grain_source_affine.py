#!/usr/bin/env python3
"""Compile and package the new linear state model as a small affine early-state reader."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.affine_program import compile_affine, score_affine
from grain_research.vector_mechanisms.navigation import (
    BASE,
    digest,
    read_json,
    write_json,
)
from scripts.export_grain_navigation_runtime import replay
from scripts.run_grain_navigation_baseline import checked_arrays
from scripts.run_grain_source_span_extension import DEFAULT as SOURCE
from scripts.run_grain_source_span_extension import OLD, load_extension
from scripts.run_grain_state_transport import load_operators

DEFAULT = BASE / "navigation_v1/NAV-11/source_affine_r1"


def prepare(out):
    """Verify exact affine compilation and export no snapshot matrices or teacher."""
    if out.exists() or not out.resolve().is_relative_to(
        (BASE / "navigation_v1").resolve()
    ):
        raise ValueError("Choose a fresh ignored export directory")
    models = load_extension(SOURCE)
    _p, old, functions = load_operators(OLD)
    programs = {q: compile_affine(m, functions[q]) for q, m in models.items()}
    out.mkdir(parents=True)
    checks = []
    with threadpool_limits(limits=2):
        for mode in ("worlds", "natural"):
            source = SOURCE / mode
            freeze = read_json(source / "prediction_freeze.json")
            if digest(source / "predictions.csv") != freeze["sha256"]:
                raise ValueError("Generated source scores changed")
            for d in pd.read_csv(source / "predictions.csv").to_dict("records"):
                index, q = d["index"], d["query"]
                meta, z = checked_arrays(
                    BASE
                    / f"navigation_v1/NAV-10/state_{mode}_r1/private_units/unit_{index:04d}.json"
                )
                if any(
                    meta["identity"][k] != d[k] for k in ("index", "query", "world_id")
                ):
                    raise ValueError("Compiled replay identity changed")
                x = z["residual_mean"] - old[q]["mean_r"]
                value = score_affine(programs[q], x)
                checks.append(
                    {
                        "mode": mode,
                        "query": q,
                        "world_id": d["world_id"],
                        **{k + "_abs_error": abs(value[k] - d[k]) for k in value},
                    }
                )
    frame = pd.DataFrame(checks)
    error = float(frame[[c for c in frame if c.endswith("_abs_error")]].max().max())
    if error > 1e-10:
        raise ValueError("Affine compilation failed its algebraic replay")
    frame.to_csv(out / "affine_compilation_replay.csv", index=False)
    bundle = out / "bundle"
    bundle.mkdir()
    (bundle / "compiled").mkdir()
    (bundle / "grain_runtime").mkdir()
    original = BASE / "navigation_v1/NAV-11/runtime_export_r1"
    original_manifest = read_json(original / "bundle/manifest.json")
    for name in (
        "context.safetensors",
        "config.json",
        "tokenizer.json",
        "queries.json",
    ):
        src = original / "bundle" / name
        if digest(src) != original_manifest["files"][name]:
            raise ValueError("Runtime scaffold changed")
        shutil.copy2(src, bundle / name)
    modules = [
        "__init__.py",
        "navigation_runtime.py",
        "projection_program.py",
        "context_program.py",
        "state_geometry.py",
        "state_transport.py",
        "affine_program.py",
    ]
    for name in modules:
        shutil.copy2(
            ROOT / "grain_research/vector_mechanisms" / name,
            bundle / "grain_runtime" / name,
        )
    shutil.copy2(ROOT / "scripts/run_grain_navigation_runtime.py", bundle / "run.py")
    for q, program in programs.items():
        np.savez_compressed(bundle / "compiled" / f"{q}.npz", **program)
    files = {
        str(f.relative_to(bundle)): digest(f) for f in bundle.rglob("*") if f.is_file()
    }
    write_json(
        bundle / "manifest.json",
        {
            "protocol": "NAV11_SOURCE_AFFINE_V1",
            "files": files,
            "program_dependencies": {"source_affine": list(files)},
            "runtime_classes": {
                "source_affine": "four extracted Transformer blocks plus one fixed affine reader; no kernel anchors or full P/Q"
            },
            "construction_freeze_sha256": digest(SOURCE / "construction_freeze.json"),
            "construction": "48 synthetic snapshots and12 disjoint unlabelled reference-author average states per query",
            "human_labels_fit": False,
            "MEPS_fit": False,
            "installed_versions": {
                n: importlib.metadata.version(n)
                for n in (
                    "numpy",
                    "scipy",
                    "tokenizers",
                    "threadpoolctl",
                    "torch",
                    "transformers",
                    "safetensors",
                )
            },
        },
    )
    # Same raw replay materials, not stored in the transferable bundle.
    shutil.copy2(original / "private_inputs.jsonl", out / "private_inputs.jsonl")
    natural_worlds = [
        w
        for w in read_json(
            BASE / "navigation_v1/NAV-04/natural_evidence_r1/private_worlds.json"
        )
        if w["condition"] == "original"
    ][:2]
    worlds = [
        w
        for w in read_json(BASE / "navigation_v1/NAV-01/controlled_r1/worlds.json")
        if w["split"] == "held_scene"
        and w["actor"] == "self"
        and w["opportunity"] == "voluntary"
        and w["realization"] == 0
    ]
    tables = {
        mode: pd.read_csv(SOURCE / mode / "predictions.csv")
        for mode in ("worlds", "natural")
    }
    expected = []
    for i, (w, mode) in enumerate(
        [(w, "natural") for w in natural_worlds] + [(w, "worlds") for w in worlds]
    ):
        for q in programs:
            r = tables[mode][
                (tables[mode].world_id == w["world_id"]) & (tables[mode]["query"] == q)
            ]
            if len(r) != 1:
                raise ValueError("Expected affine replay score missing")
            expected.append(
                {
                    "id": f"case{i}_{q}",
                    "program": "source_affine",
                    "score": float(r.iloc[0].score),
                }
            )
    inputs = [
        json.loads(line)
        for line in (out / "private_inputs.jsonl").read_text().splitlines()
    ]
    if {x["id"] for x in inputs} != {x["id"] for x in expected}:
        raise ValueError("Raw replay membership differs")
    write_json(out / "expected.json", expected)
    write_json(
        out / "export.json",
        {
            "status": "EXPORTED",
            "bundle_manifest_sha256": digest(bundle / "manifest.json"),
            "input_sha256": digest(out / "private_inputs.jsonl"),
            "expected_sha256": digest(out / "expected.json"),
            "replay_rows_per_program": 15,
            "absolute_score_tolerance": 1e-5,
            "affine_compilation_rows": len(frame),
            "affine_max_abs_error": error,
            "affine_parameter_bytes": sum(
                v.nbytes for c in programs.values() for v in c.values()
            ),
            "source_operator_parameter_bytes": sum(
                v.nbytes for m in models.values() for v in m.values()
            ),
            "reader_depends_on_current_author_or_batch": False,
            "full_LLM_required": False,
            "retained_transformer_layers": 4,
        },
    )
    print(read_json(out / "export.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "replay"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    a = p.parse_args()
    with threadpool_limits(limits=2):
        prepare(a.out) if a.action == "prepare" else replay(a.out, "source_affine")
