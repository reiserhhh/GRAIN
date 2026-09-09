#!/usr/bin/env python3
"""NAV-03/12: shallow-context pullback and local charts, without full-prefix runtime."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.context_program import (
    CUTS,
    CroppedContext,
    cropped_config_dict,
    score_context_charts,
)
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
from grain_research.vector_mechanisms.projection_program import pool_pairings
from scripts.run_grain_navigation_baseline import checked_arrays, save_unit
from scripts.run_grain_upstream_routes import collector_guard


def code_hashes():
    """Pin construction, arithmetic and inference implementation for this packet."""
    return {
        p: digest(ROOT / p)
        for p in (
            "scripts/run_grain_context_program.py",
            "grain_research/vector_mechanisms/context_program.py",
            "grain_research/vector_mechanisms/projection_program.py",
        )
    }


def construct(reference, out):
    """Differentiate six construction texts once each; export only eight early blocks."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from grain_research.vector_mechanisms.context_program import (
        capture_context_jacobians,
    )
    from grain_research.vector_mechanisms.trace import DecoderObserver

    if out.exists():
        raise FileExistsError("Choose a fresh contextual program ID")
    parent = read_json(reference / "manifest.json")
    ref = reference / "synthetic_reference.npz"
    if (
        digest(ref)
        != read_json(reference / "reference_source_manifest.json")["R_synth"]
    ):
        raise ValueError("Reference changed")
    if digest(MODEL / "model.safetensors") != parent["model_sha256"]:
        raise ValueError("Model changed")
    torch.set_num_threads(2)
    obs = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    obs.model.requires_grad_(False)
    out.mkdir(parents=True)
    (out / "charts").mkdir()
    write_json(
        out / "manifest.json",
        {
            "protocol": "NAV03_CONTEXT_PULLBACK_V1",
            "cuts": CUTS,
            "synthetic_references": parent["synthetic_references"],
            "reference_sha256": digest(ref),
            "source_model_sha256": parent["model_sha256"],
            "code_sha256": code_hashes(),
            "pooling": ["chart0", "chart1", "mean", "affinity"],
            "alignment": "exact P/Q slots plus semantic_order contextual evidence alignment",
            "gate": "softmax(8*cos(mean_evidence_hidden, mean_reference_hidden))",
            "human_label_fit": False,
            "runtime_class": "extracted-neural-program",
        },
    )
    fs = load_functions(ref)
    design = {
        (d["world_id"], d["query"]): d for d in read_json(reference / "design.json")
    }
    start = time.monotonic()
    for q, worlds in parent["synthetic_references"].items():
        for i, world in enumerate(worlds):
            d = design[(world, q)]
            charts = capture_context_jacobians(obs, d["input_ids"], d["mask"], fs[q])
            for cut, values in charts.items():
                save_unit(
                    out / "charts" / f"{q}_{i}_{cut}.json",
                    {
                        "query": q,
                        "world_id": world,
                        "cut": cut,
                        "source_split": "construction",
                    },
                    **values,
                )
    cfg = cropped_config_dict(obs.model.config.to_dict(), max(CUTS) + 1)
    write_json(out / "config.json", cfg)
    selected, ledger = {}, []
    with safe_open(MODEL / "model.safetensors", framework="pt", device="cpu") as file:
        for k in file.keys():  # noqa: SIM118 - safe_open is not a mapping iterator.
            if not k.startswith("model."):
                continue
            key = k.removeprefix("model.")
            if key.startswith("layers.") and int(key.split(".")[1]) > max(CUTS):
                continue
            value = file.get_tensor(k).contiguous()
            selected[key] = value
            ledger.append(
                {
                    "source_key": k,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "bytes": value.numel() * value.element_size(),
                }
            )
    save_file(selected, out / "context.safetensors")
    write_json(
        out / "program.json",
        {
            "cuts": CUTS,
            "question_tokens": parent["question_tokens"],
            "anchors": {q: fs[q].anchor for q in QUERIES},
            "weights_sha256": digest(out / "context.safetensors"),
            "config_sha256": digest(out / "config.json"),
            "retained_layers": 8,
            "original_layers": len(obs.model.model.layers),
            "full_PQ_boundary_layers": 17,
            "runtime_lm_head": False,
            "runtime_full_PQ_inputs": False,
            "runtime_transformer_blocks": True,
            "weight_bytes": sum(v["bytes"] for v in ledger),
            "weights": ledger,
            "calls_construction": obs.calls,
            "seconds_construction": time.monotonic() - start,
        },
    )
    write_json(
        out / "run_status.json", {"status": "CONSTRUCTION_COMPLETE", "calls": obs.calls}
    )
    print(
        {
            "status": "CONSTRUCTION_COMPLETE",
            "calls": obs.calls,
            "seconds": time.monotonic() - start,
        },
        flush=True,
    )


def load_program(program_dir):
    """Load only the extracted decoder; the full original model is not accessed."""
    import torch
    from safetensors.torch import load_file
    from transformers import Qwen2Config, Qwen2Model

    p = read_json(program_dir / "program.json")
    if (
        digest(program_dir / "context.safetensors") != p["weights_sha256"]
        or digest(program_dir / "config.json") != p["config_sha256"]
    ):
        raise ValueError("Extracted program changed")
    torch.set_num_threads(2)
    cfg = Qwen2Config.from_dict(read_json(program_dir / "config.json"))
    cfg._attn_implementation = "eager"
    model = Qwen2Model(cfg).float()
    model.load_state_dict(load_file(program_dir / "context.safetensors"), strict=True)
    model.to("mps")
    runtime = CroppedContext(model, cuts=p["cuts"])
    charts = {}
    for q in QUERIES:
        charts[q] = {}
        for cut in p["cuts"]:
            charts[q][cut] = []
            for i in range(2):
                meta, z = checked_arrays(program_dir / "charts" / f"{q}_{i}_{cut}.json")
                if (
                    meta["query"] != q
                    or meta["cut"] != cut
                    or meta["source_split"] != "construction"
                ):
                    raise ValueError("Chart provenance changed")
                e = ~z["mask"]
                e[-1] = False
                z["offset"] = pool_pairings(
                    z["hidden"][e],
                    z["hidden"][e],
                    z["gradients"][:, e],
                    "semantic_order",
                )
                charts[q][cut].append(z)
    return p, runtime, charts


def external_units():
    """Read existing author inputs only; human scores are deliberately excluded."""
    paths = {
        "source": BASE / "all_factor_source_m29_v1/private_design.json",
        "mood": BASE / "counseling_mood_m30c_v1/private_design.json",
    }
    units = []
    for d in read_json(paths["source"]):
        if d["query"] in ("extraversion", "TF_cont"):
            units.append(
                {
                    "dataset": "PANDORA_source",
                    "author_id": d["author_id"],
                    "query": d["query"],
                    "views": d["views"],
                }
            )
    for d in read_json(paths["mood"]):
        if d["query"] == "adapted_dejected5":
            units.append(
                {
                    "dataset": d["view"],
                    "author_id": d["participant_code"],
                    "query": d["query"],
                    "views": [
                        {
                            "input_ids": d["input_ids"],
                            "question_mask": d["question_mask"],
                        }
                    ],
                }
            )
    if len(units) != 369:
        raise ValueError("Real input membership changed")
    return units, paths


def evaluate(program_dir, reference, out, external=False, seconds=1200):
    """Score every fixed local chart before any model target or human criterion join."""
    from threadpoolctl import threadpool_limits

    program, runtime, charts = load_program(program_dir)
    paths = {"design": reference / "design.json", "worlds": reference / "worlds.json"}
    if external:
        units, paths = external_units()
    else:
        worlds = {w["world_id"]: w for w in read_json(paths["worlds"])}
        units = [
            {
                "world_id": d["world_id"],
                "query": d["query"],
                "split": worlds[d["world_id"]]["split"],
                "views": [{"input_ids": d["input_ids"], "question_mask": d["mask"]}],
            }
            for d in read_json(paths["design"])
        ]
    expected = {
        "protocol": "NAV03_CONTEXT_EXTERNAL_V1"
        if external
        else "NAV03_CONTEXT_WORLDS_V1",
        "program_sha256": digest(program_dir / "program.json"),
        "code_sha256": code_hashes(),
        "sources": {k: digest(v) for k, v in paths.items()},
        "units": len(units),
        "human_label_fit": False,
        "retained_layers": program["retained_layers"],
    }
    if out.exists():
        if read_json(out / "manifest.json") != expected:
            raise ValueError("Frozen evaluation changed")
    else:
        out.mkdir(parents=True)
        (out / "private_units").mkdir()
        write_json(out / "manifest.json", expected)
        write_json(
            out / "run_status.json", {"status": "PREPARED", "calls": 0, "seconds": 0}
        )
    prior = read_json(out / "run_status.json")
    start, rows = time.monotonic(), []
    with threadpool_limits(limits=2):
        for i, unit in enumerate(units):
            q = unit["query"]
            identity = {k: v for k, v in unit.items() if k != "views"}
            file = out / "private_units" / f"{i:04d}.json"
            if file.exists():
                saved = read_json(file)
                if saved["identity"] != identity:
                    raise ValueError("Unit identity changed")
                rows.extend(saved["rows"])
                continue
            if time.monotonic() - start >= seconds:
                write_json(
                    out / "run_status.json",
                    {
                        "status": "PAUSED_RESOURCE_BOUNDARY",
                        "calls": prior["calls"] + runtime.calls,
                        "seconds": prior["seconds"] + time.monotonic() - start,
                    },
                )
                return
            view_rows = []
            for view in unit["views"]:
                ids, mask = view["input_ids"], view["question_mask"]
                if np.asarray(ids)[mask].tolist() != program["question_tokens"][q]:
                    raise ValueError("Query identity differs from construction")
                states = runtime.capture(ids)
                for cut, h in states.items():
                    values = score_context_charts(
                        charts[q][cut], h, mask, program["anchors"][q]
                    )
                    view_rows.extend(
                        {"cut": cut, "pooling": pooling, **value}
                        for pooling, value in values.items()
                    )
            aggregate = (
                pd.DataFrame(view_rows)
                .groupby(["cut", "pooling"], as_index=False)
                .mean()
            )
            result = [
                {
                    **identity,
                    **r,
                    "method": f"context{int(r['cut']) + 1}_{r['pooling']}",
                }
                for r in aggregate.to_dict("records")
            ]
            write_json(file, {"identity": identity, "rows": result})
            rows.extend(result)
            if (i + 1) % 30 == 0:
                print(
                    {
                        "status": "CONTEXT_SCORING",
                        "units": i + 1,
                        "total": len(units),
                        "calls": runtime.calls,
                        "seconds": time.monotonic() - start,
                    },
                    flush=True,
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "private_predictions.csv", index=False)
    write_json(
        out / "prediction_freeze.json",
        {
            "sha256": digest(out / "private_predictions.csv"),
            "cached_full_PQ_read": False,
            "human_criteria_read_before_freeze": False,
            "full_model_loaded": False,
            "executed_blocks_per_input": list(range(max(CUTS) + 1)),
            "comparison_cuts": [c + 1 for c in CUTS],
            "runtime_class": "extracted-neural-program",
        },
    )
    if external:
        summarize_external(frame, out)
    else:
        target = pd.read_csv(reference / "native_and_function.csv").query(
            "reference=='R_synth'"
        )
        merged = frame.merge(
            target[["world_id", "query", "score", "native_score"]].rename(
                columns={"score": "full_function"}
            ),
            on=["world_id", "query"],
            validate="many_to_one",
        )
        if len(merged) != len(frame):
            raise ValueError("Missing full-function target")
        rows = []
        for (q, method, split), g in merged.groupby(["query", "method", "split"]):
            rows.append(
                {
                    "query": q,
                    "method": method,
                    "split": split,
                    **quality(g.full_function, g.score),
                    "native_r": quality(g.native_score, g.score)["r"],
                }
            )
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
        merged.to_csv(out / "private_joined.csv", index=False)
    write_json(
        out / "run_status.json",
        {
            "status": "COMPLETE",
            "calls": prior["calls"] + runtime.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "units": len(units),
        },
    )
    print(read_json(out / "run_status.json"), flush=True)


def summarize_external(frame, out):
    """Join verified author-level endpoints after all candidate predictions are frozen."""
    from scripts.analyze_grain_all_factor_source import paired_quality

    ref = BASE / "navigation_v1/NAV-10/reference_transfer_r1"
    source = pd.read_csv(
        ref / "private_source_external.csv", dtype={"author_id": str}
    ).query("reference=='R_synth'")
    source = source[["author_id", "query", "score", "y_true", "prediction"]].rename(
        columns={"score": "full_function"}
    )
    joined = frame[frame.dataset == "PANDORA_source"].merge(
        source, on=["author_id", "query"], validate="many_to_one"
    )
    if len(joined) != 277 * 12:
        raise ValueError("Missing source endpoints")
    mood = (
        pd.read_csv(ref / "private_counseling_predictions.csv")
        .query("reference=='R_synth'")
        .rename(
            columns={
                "participant_code": "author_id",
                "view": "dataset",
                "score": "full_function",
            }
        )
    )
    matched = frame[frame.dataset != "PANDORA_source"].merge(
        mood[["author_id", "dataset", "query", "full_function", "native_score"]],
        on=["author_id", "dataset", "query"],
        validate="many_to_one",
    )
    if len(matched) != 92 * 12:
        raise ValueError("Missing mood endpoints")
    criterion = BASE / "counseling_mood_m30c_v1/private_mood_scores.csv"
    if (
        digest(criterion)
        != read_json(criterion.parent / "manifest.json")["criterion_cache_sha256"]
    ):
        raise ValueError("Mood criterion changed")
    criteria = pd.read_csv(criterion).pivot(
        index="participant_code", columns="scale", values="score"
    )
    rows = []
    for (q, method), g in joined.groupby(["query", "method"]):
        rows.append(
            {
                "dataset": "PANDORA_source",
                "query": q,
                "method": method,
                "full_function_r": quality(g.full_function, g.score)["r"],
                **paired_quality(
                    g.score, g.prediction, g.y_true, binary=q == "TF_cont"
                ),
            }
        )
    for (dataset, method), g in matched.groupby(["dataset", "method"]):
        key = f"{'post' if dataset == 'free_chat' else 'mid'}_adapted_dejected5"
        y = (criteria.loc[g.author_id, key].to_numpy() - 1) / 5
        rows.append(
            {
                "dataset": dataset,
                "query": "adapted_dejected5",
                "method": method,
                "full_function_r": quality(g.full_function, g.score)["r"],
                **paired_quality(g.score, g.native_score, y),
            }
        )
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    joined.to_csv(out / "private_source_joined.csv", index=False)
    matched.to_csv(out / "private_mood_joined.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["construct", "worlds", "external"])
    p.add_argument("--reference", type=Path, default=OUTPUT)
    p.add_argument(
        "--program",
        type=Path,
        default=BASE / "navigation_v1/NAV-03/context_pullback_r2",
    )
    p.add_argument("--out", type=Path)
    p.add_argument("--seconds", type=int, default=1200)
    a = p.parse_args()
    out = a.out or (
        a.program
        if a.mode == "construct"
        else BASE / f"navigation_v1/NAV-10/context_{a.mode}_r2"
    )
    with collector_guard(out):
        if a.mode == "construct":
            construct(a.reference, a.program)
        else:
            evaluate(a.program, a.reference, out, a.mode == "external", a.seconds)
