#!/usr/bin/env python3
"""NAV-03 first explicit token-to-projection approximation and held-scene test."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.functional_profiles import FunctionalQuery
from grain_research.vector_mechanisms.navigation import (
    BASE,
    MODEL,
    OUTPUT,
    QUERIES,
    digest,
    load_functions,
    quality,
    read_json,
    write_json,
)
from grain_research.vector_mechanisms.projection_program import (
    METHODS,
    pool_pairings,
    score_token_ids,
)
from scripts.run_grain_navigation_baseline import checked_arrays, save_unit


def construct(reference, out):
    """Differentiate two frozen projections at six synthetic construction origins."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver
    from grain_research.vector_mechanisms.upstream_routes import (
        capture_differentiable_prefix,
    )

    if out.exists():
        raise FileExistsError("Use a new program run ID")
    manifest = read_json(reference / "manifest.json")
    ref_path = reference / "synthetic_reference.npz"
    if (
        digest(ref_path)
        != read_json(reference / "reference_source_manifest.json")["R_synth"]
    ):
        raise ValueError("Synthetic reader changed")
    design = {
        (d["world_id"], d["query"]): d for d in read_json(reference / "design.json")
    }
    fs = load_functions(ref_path)
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    if digest(MODEL / "model.safetensors") != manifest["model_sha256"]:
        raise ValueError("Frozen model changed")
    out.mkdir(parents=True)
    (out / "charts").mkdir()
    write_json(
        out / "manifest.json",
        {
            "status": "CONSTRUCTING",
            "reference_sha256": digest(ref_path),
            "reference_manifest_sha256": digest(reference / "manifest.json"),
            "methods": METHODS,
            "synthetic_references": manifest["synthetic_references"],
            "source_model_sha256": manifest["model_sha256"],
            "human_label_fit": False,
            "runtime": "input token embedding table + explicit frozen alignment + two derivative projections",
            "candidate_scope": "local first-order approximation; query token positions and contextual composition omitted",
            "code_sha256": {
                p: digest(ROOT / p)
                for p in (
                    "scripts/run_grain_projection_program.py",
                    "grain_research/vector_mechanisms/projection_program.py",
                    "grain_research/vector_mechanisms/upstream_routes.py",
                )
            },
        },
    )
    start = time.monotonic()
    for q, ids in manifest["synthetic_references"].items():
        f = fs[q]
        G = f.config_coef + f.mean_coef / len(f.p_ref)
        pieces = (
            FunctionalQuery(f.p_ref, f.q_ref, G, np.zeros_like(f.g_final), 0),
            FunctionalQuery(f.p_ref, f.q_ref, np.zeros_like(G), f.g_final, 0),
        )
        for i, wid in enumerate(ids):
            d = design[(wid, q)]
            select = ~np.asarray(d["mask"], bool)
            select[-1] = False
            gradients, values = [], []
            for part in pieces:
                capture = capture_differentiable_prefix(
                    observer, d["input_ids"], d["mask"], part
                )
                gradients.append(capture["input_gradient"][select])
                values.append(capture["score"])
            device = next(observer.model.parameters()).device
            tokens = torch.tensor(np.asarray(d["input_ids"])[select], device=device)
            with torch.no_grad():
                embeddings = (
                    observer.model.model.embed_tokens(tokens).float().cpu().numpy()
                )
            save_unit(
                out / "charts" / f"{q}_{i}.json",
                {"query": q, "world_id": wid, "source_split": "construction"},
                embeddings=embeddings,
                gradients=np.asarray(gradients),
                projection_origin=np.asarray(values),
            )
    # Only this tensor, not Transformer blocks or LM head, is retained at runtime.
    table = observer.model.model.embed_tokens.weight.detach().float().cpu().numpy()
    np.save(out / "embedding_table.npy", table, allow_pickle=False)
    write_json(
        out / "program.json",
        {
            "queries": list(QUERIES),
            "anchors": {q: fs[q].anchor for q in QUERIES},
            "question_tokens": manifest["question_tokens"],
            "embedding_sha256": digest(out / "embedding_table.npy"),
            "embedding_shape": list(table.shape),
            "embedding_bytes": table.nbytes,
            "neural_calls_construction": observer.calls,
            "seconds": time.monotonic() - start,
            "neural_calls_runtime": 0,
            "full_transformer_required_runtime": False,
            "pretrained_embedding_required": True,
        },
    )
    write_json(
        out / "run_status.json",
        {"status": "CONSTRUCTION_COMPLETE", "calls": observer.calls},
    )
    print(
        {
            "status": "CONSTRUCTION_COMPLETE",
            "calls": observer.calls,
            "seconds": time.monotonic() - start,
        },
        flush=True,
    )


def load_program(out):
    """Load only NumPy parameters; no Torch/model inference is imported here."""
    program = read_json(out / "program.json")
    if digest(out / "embedding_table.npy") != program["embedding_sha256"]:
        raise ValueError("Embedding table changed")
    table = np.load(out / "embedding_table.npy", mmap_mode="r", allow_pickle=False)
    charts = {}
    for q in program["queries"]:
        charts[q] = []
        for i in range(2):
            meta, z = checked_arrays(out / "charts" / f"{q}_{i}.json")
            if meta["source_split"] != "construction" or meta["query"] != q:
                raise ValueError("Chart provenance changed")
            z["offsets"] = {
                method: pool_pairings(
                    z["embeddings"], z["embeddings"], z["gradients"], method
                )
                for method in METHODS
            }
            charts[q].append(z)
    return program, table, charts


def evaluate(reference, out):
    """Run the actual independent candidate before comparing model-derived targets."""
    from threadpoolctl import threadpool_limits

    start = time.monotonic()
    with threadpool_limits(limits=2):
        program, table, charts = load_program(out)
        rows = []
        worlds = {w["world_id"]: w for w in read_json(reference / "worlds.json")}
        for d in read_json(reference / "design.json"):
            q = d["query"]
            if (
                np.asarray(d["input_ids"])[d["mask"]].tolist()
                != program["question_tokens"][q]
            ):
                raise ValueError("Query identity differs from the program")
            for method in METHODS:
                value = score_token_ids(
                    d["input_ids"],
                    d["mask"],
                    table,
                    charts[q],
                    program["anchors"][q],
                    method,
                )
                rows.append(
                    {
                        "world_id": d["world_id"],
                        "query": q,
                        "method": method,
                        "split": worlds[d["world_id"]]["split"],
                        **{k: v for k, v in value.items() if k != "charts"},
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "predictions.csv", index=False)
    targets = pd.read_csv(reference / "native_and_function.csv").query(
        "reference == 'R_synth'"
    )
    joined = frame.merge(
        targets[["world_id", "query", "score", "native_score"]].rename(
            columns={"score": "function_target"}
        ),
        on=["world_id", "query"],
        validate="many_to_one",
    )
    if len(joined) != len(frame):
        raise ValueError("Missing target states")
    metrics = []
    for (q, method, split), g in joined.groupby(["query", "method", "split"]):
        metrics.append(
            {
                "query": q,
                "method": method,
                "split": split,
                **quality(g.function_target, g.score),
                "native_r": quality(g.native_score, g.score)["r"],
            }
        )
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(out / "metrics.csv", index=False)
    write_json(
        out / "evaluation.json",
        {
            "status": "COMPLETE_FIRST_CANDIDATE_EVAL",
            "runtime_neural_calls": 0,
            "texts_scored": 288,
            "method_conditions": 3,
            "seconds": time.monotonic() - start,
            "reference": "R_synth",
            "raw_token_input_only": True,
            "human_outcome_fit": False,
        },
    )
    print(
        metrics[metrics.split == "held_scene"].round(5).to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("construct", "evaluate"))
    p.add_argument("--reference-dir", type=Path, default=OUTPUT)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=BASE / "navigation_v1/NAV-03/input_pullback_r1",
    )
    a = p.parse_args()
    if not a.output_dir.resolve().is_relative_to((BASE / "navigation_v1").resolve()):
        p.error("Use ignored navigation outputs")
    (construct if a.mode == "construct" else evaluate)(
        a.reference_dir.resolve(), a.output_dir.resolve()
    )
