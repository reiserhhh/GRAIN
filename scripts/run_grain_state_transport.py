#!/usr/bin/env python3
"""NAV-02 route A: construction-only early-to-P/Q residual state transport."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.context_program import (
    CroppedContext,
    cropped_config_dict,
)
from grain_research.vector_mechanisms.navigation import (
    BASE,
    OUTPUT,
    QUERIES,
    digest,
    load_functions,
    read_json,
    write_json,
)
from grain_research.vector_mechanisms.state_transport import (
    fit_transport,
    pack_state,
    reconstruct,
    unpack_state,
)
from scripts.run_grain_navigation_baseline import checked_arrays, save_unit
from scripts.run_grain_upstream_routes import collector_guard

DEFAULT = BASE / "navigation_v1/NAV-02/state_transport_r1"
CONTEXT = BASE / "navigation_v1/NAV-03/context_pullback_r2"


def code_hashes():
    """Pin state construction and inference implementations."""
    return {
        p: digest(ROOT / p)
        for p in (
            "scripts/run_grain_state_transport.py",
            "grain_research/vector_mechanisms/state_transport.py",
            "grain_research/vector_mechanisms/context_program.py",
        )
    }


def load_runtime(folder):
    """Load four extracted blocks, with neither the full model nor its LM head."""
    import torch
    from safetensors.torch import load_file
    from transformers import Qwen2Config, Qwen2Model

    p = read_json(folder / "program.json")
    for name in ("context.safetensors", "config.json"):
        if digest(folder / name) != p["files"][name]:
            raise ValueError("State runtime bytes changed")
    torch.set_num_threads(2)
    cfg = Qwen2Config.from_dict(read_json(folder / "config.json"))
    cfg._attn_implementation = "eager"
    model = Qwen2Model(cfg).float()
    model.load_state_dict(load_file(folder / "context.safetensors"), strict=True)
    model.to("mps")
    return p, CroppedContext(model, cuts=(3,))


def load_operators(folder):
    """Load frozen state operators and fixed readers without later-layer inputs."""
    p = read_json(folder / "program.json")
    if p["status"] != "CONSTRUCTION_COMPLETE":
        raise ValueError("Finish state construction")
    for name, sha in p["files"].items():
        if digest(folder / name) != sha:
            raise ValueError("Independent program artifact changed: " + name)
    operators = {}
    for q in QUERIES:
        with np.load(folder / f"{q}.npz", allow_pickle=False) as arrays:
            operators[q] = dict(arrays)
    return p, operators, load_functions(folder / "functions.npz")


def construction(out, seconds=1200):
    """Fit paired states on 48 construction worlds per query; never read human data."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    m = read_json(OUTPUT / "manifest.json")
    if digest(OUTPUT / "design.json") != m["design_sha256"]:
        raise ValueError("Controlled design changed")
    source = read_json(CONTEXT / "program.json")
    if digest(CONTEXT / "context.safetensors") != source["weights_sha256"]:
        raise ValueError("Extracted source changed")
    expected = {
        "protocol": "NAV02_REDUCED_STATE_TRANSPORT_V1",
        "input_sha256": digest(OUTPUT / "design.json"),
        "source_program_sha256": digest(CONTEXT / "program.json"),
        "reference_sha256": digest(OUTPUT / "synthetic_reference.npz"),
        "code_sha256": code_hashes(),
        "alpha": 1.0,
        "kernels": [1, 2],
        "construction_only": True,
        "human_data_fit": False,
        "MEPS_fit": False,
        "predeclared_methods": [
            "upper_mean",
            "residual_mean",
            "transport1",
            "transport2",
        ],
    }
    if out.exists():
        if read_json(out / "manifest.json") != expected:
            raise ValueError("Construction manifest changed")
        if read_json(out / "program.json")["status"] == "CONSTRUCTION_COMPLETE":
            print({"status": "CONSTRUCTION_COMPLETE", "new_calls": 0}, flush=True)
            return
    else:
        out.mkdir(parents=True)
        (out / "private_early").mkdir()
        write_json(out / "manifest.json", expected)
        cfg = cropped_config_dict(read_json(CONTEXT / "config.json"), 4)
        write_json(out / "config.json", cfg)
        weights = {}
        with safe_open(
            CONTEXT / "context.safetensors", framework="pt", device="cpu"
        ) as f:
            for key in f.keys():  # noqa: SIM118
                if key.startswith("layers.") and int(key.split(".")[1]) >= 4:
                    continue
                weights[key] = f.get_tensor(key).contiguous()
        save_file(weights, out / "context.safetensors")
        from grain_research.vector_mechanisms.navigation import MODEL

        for name in ("tokenizer.json", "tokenizer_config.json"):
            shutil.copy2(MODEL / name, out / name)
        shutil.copy2(OUTPUT / "synthetic_reference.npz", out / "functions.npz")
        write_json(
            out / "program.json",
            {
                "status": "PREPARED",
                "question_tokens": m["question_tokens"],
                "files": {
                    name: digest(out / name)
                    for name in (
                        "config.json",
                        "context.safetensors",
                        "functions.npz",
                        "tokenizer.json",
                        "tokenizer_config.json",
                    )
                },
                "runtime_class": "extracted-neural-plus-reduced-state-operator",
                "runtime_layers": 4,
                "full_PQ_boundary_layers": 17,
                "original_full_layers": 28,
                "runtime_lm_head": False,
                "full_teacher_or_PQ_inputs_at_runtime": False,
                "weight_bytes": (out / "context.safetensors").stat().st_size,
            },
        )
        write_json(
            out / "run_status.json", {"status": "PREPARED", "calls": 0, "seconds": 0}
        )
    p, runtime = load_runtime(out)
    prior, start = read_json(out / "run_status.json"), time.monotonic()
    design = read_json(OUTPUT / "design.json")
    roles = {w["world_id"]: w["split"] for w in read_json(OUTPUT / "worlds.json")}
    with threadpool_limits(limits=2):
        for d in design:
            path = out / "private_early" / f"unit_{d['index']:04d}.json"
            if path.exists():
                meta, _ = checked_arrays(path)
                if any(meta[k] != d[k] for k in ("index", "query", "world_id")):
                    raise ValueError("Early snapshot identity changed")
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
            ids, mask = d["input_ids"], np.asarray(d["mask"], bool)
            if np.asarray(ids)[mask].tolist() != p["question_tokens"][d["query"]]:
                raise ValueError("Query coordinates changed")
            h = runtime.capture(ids)[3]
            save_unit(
                path,
                {k: d[k] for k in ("index", "query", "world_id")},
                P=h[mask],
                Q=h[-1],
            )
        diagnostics = []
        for q in QUERIES:
            ds = [
                d
                for d in design
                if d["query"] == q and roles[d["world_id"]] == "construction"
            ]
            if len(ds) != 48:
                raise ValueError("Expected 48 construction snapshots per query")
            xx, yy = [], []
            for d in ds:
                _, early = checked_arrays(
                    out / "private_early" / f"unit_{d['index']:04d}.json"
                )
                meta, upper = checked_arrays(
                    OUTPUT / "packets" / f"unit_{d['index']:04d}.json"
                )
                if any(meta[k] != d[k] for k in ("index", "query", "world_id")):
                    raise ValueError("Teacher snapshot identity mismatch")
                xx.append(pack_state(early["P"], early["Q"]))
                yy.append(pack_state(upper["P"], upper["Q"]))
            model = fit_transport(np.stack(xx), np.stack(yy), alpha=1.0)
            np.savez_compressed(out / f"{q}.npz", **model)
            p["files"][f"{q}.npz"] = digest(out / f"{q}.npz")
            p.setdefault("state_shapes", {})[q] = list(early["P"].shape)
            diagnostics.append(
                {
                    "query": q,
                    "construction_worlds": len(ds),
                    "construction_scenes": 6,
                    "state_dimensions": len(xx[0]),
                    "residual_span_upper_rank": len(ds) - 1,
                    "effective_df_linear": float(model["effective_df1"]),
                    "effective_df_quadratic": float(model["effective_df2"]),
                }
            )
        pd.DataFrame(diagnostics).to_csv(
            out / "construction_diagnostics.csv", index=False
        )
    p["status"] = "CONSTRUCTION_COMPLETE"
    p["construction_source"] = "six synthetic scenes; no human text/labels"
    p["neural_calls"] = prior["calls"] + runtime.calls
    p["seconds"] = prior["seconds"] + time.monotonic() - start
    write_json(out / "program.json", p)
    write_json(
        out / "run_status.json",
        {
            "status": "CONSTRUCTION_COMPLETE",
            "calls": p["neural_calls"],
            "seconds": p["seconds"],
        },
    )
    print(read_json(out / "run_status.json"), flush=True)


def input_units(mode):
    """Read input coordinates only; target states and human criteria are separate."""
    if mode in ("worlds", "natural"):
        folder = (
            OUTPUT
            if mode == "worlds"
            else BASE / "navigation_v1/NAV-04/natural_evidence_r1"
        )
        path = folder / ("design.json" if mode == "worlds" else "private_design.json")
        m = read_json(folder / "manifest.json")
        expected = (
            m["design_sha256"]
            if mode == "worlds"
            else m["input_sha256"]["private_design.json"]
        )
        if digest(path) != expected:
            raise ValueError("Input design changed")
        return [
            {
                **{k: d[k] for k in ("index", "query", "world_id")},
                "views": [{"input_ids": d["input_ids"], "question_mask": d["mask"]}],
            }
            for d in read_json(path)
        ], {str(path.relative_to(ROOT)): digest(path)}
    source = BASE / "all_factor_source_m29_v1/private_design.json"
    mood = BASE / "counseling_mood_m30c_v1/private_design.json"
    rows = []
    for d in read_json(source):
        if d["query"] in ("extraversion", "TF_cont"):
            rows.append(
                {
                    "index": len(rows),
                    "dataset": "PANDORA_source",
                    "author_id": d["author_id"],
                    "query": d["query"],
                    "views": d["views"],
                }
            )
    for d in read_json(mood):
        if d["query"] == "adapted_dejected5":
            rows.append(
                {
                    "index": len(rows),
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
    if len(rows) != 369:
        raise ValueError("External packet membership changed")
    for path in (source, mood):
        if digest(path) != read_json(path.parent / "manifest.json")["input_sha256"]:
            raise ValueError("External input source changed")
    return rows, {str(p.relative_to(ROOT)): digest(p) for p in (source, mood)}


def evaluate(program_dir, out, mode, seconds=1200):
    """Generate states and scores first, saving a freeze before teacher/label joins."""
    p, operators, functions = load_operators(program_dir)
    units, sources = input_units(mode)
    expected = {
        "protocol": "NAV02_RAW_INPUT_STATE_EVALUATION_V1",
        "mode": mode,
        "program_sha256": digest(program_dir / "program.json"),
        "input_sha256": sources,
        "code_sha256": code_hashes(),
        "full_teacher_inputs": False,
        "human_labels_before_freeze": False,
        "methods": ["upper_mean", "residual_mean", "transport1", "transport2"],
    }
    if out.exists():
        if read_json(out / "manifest.json") != expected:
            raise ValueError("Evaluation identity changed")
        if read_json(out / "run_status.json")["status"] == "COMPLETE":
            print(read_json(out / "run_status.json"), flush=True)
            return
    else:
        out.mkdir(parents=True)
        (out / "private_units").mkdir()
        write_json(out / "manifest.json", expected)
        write_json(
            out / "run_status.json", {"status": "PREPARED", "calls": 0, "seconds": 0}
        )
    _, runtime = load_runtime(program_dir)
    prior, start = read_json(out / "run_status.json"), time.monotonic()
    rows = []
    with threadpool_limits(limits=2):
        for u in units:
            path = out / "private_units" / f"unit_{u['index']:04d}.json"
            ident = {k: v for k, v in u.items() if k != "views"}
            if path.exists():
                meta = read_json(path)
                if meta["identity"] != ident:
                    raise ValueError("Prediction identity changed")
                rows.extend(meta["rows"])
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
            q, predictions, generated = u["query"], [], []
            for view in u["views"]:
                ids, mask = view["input_ids"], np.asarray(view["question_mask"], bool)
                if np.asarray(ids)[mask].tolist() != p["question_tokens"][q]:
                    raise ValueError("Early query tokens changed")
                early_path = (
                    program_dir / "private_early" / f"unit_{u['index']:04d}.json"
                )
                if mode == "worlds" and early_path.exists():
                    meta, state = checked_arrays(early_path)
                    if any(meta[k] != u[k] for k in ("index", "query", "world_id")):
                        raise ValueError("Early cache mismatch")
                    x = pack_state(state["P"], state["Q"])
                else:
                    h = runtime.capture(ids)[3]
                    x = pack_state(h[mask], h[-1])
                reconstruction = reconstruct(operators[q], x)
                generated.append(reconstruction)
                predictions.append(
                    {
                        name: functions[q].parts(*unpack_state(y, p["state_shapes"][q]))
                        for name, y in reconstruction.items()
                    }
                )
            current = [
                ident
                | {"readout": name}
                | {
                    k: float(np.mean([v[name][k] for v in predictions]))
                    for k in predictions[0][name]
                }
                for name in expected["methods"]
            ]
            meta = {"identity": ident, "rows": current}
            if mode == "external":
                write_json(path, meta)
            else:
                save_unit(path, meta, **generated[0])
            rows.extend(current)
            if (u["index"] + 1) % 30 == 0:
                print(
                    {
                        "status": "SCORING",
                        "mode": mode,
                        "units": u["index"] + 1,
                        "total": len(units),
                        "seconds": time.monotonic() - start,
                    },
                    flush=True,
                )
    pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)
    write_json(
        out / "prediction_freeze.json",
        {
            "sha256": digest(out / "predictions.csv"),
            "full_teacher_inputs": False,
            "human_labels_before_freeze": False,
        },
    )
    write_json(
        out / "run_status.json",
        {
            "status": "COMPLETE",
            "units": len(units),
            "calls": prior["calls"] + runtime.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
        },
    )
    print(read_json(out / "run_status.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("construct", "evaluate"))
    parser.add_argument("--program", type=Path, default=DEFAULT)
    parser.add_argument(
        "--mode", choices=("worlds", "natural", "external"), default="worlds"
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seconds", type=int, default=1200)
    args = parser.parse_args()
    out = args.out or BASE / f"navigation_v1/NAV-10/state_{args.mode}_r1"
    lock = args.program if args.action == "construct" else out
    with collector_guard(lock):
        if args.action == "construct":
            construction(args.program, args.seconds)
        else:
            evaluate(args.program, out, args.mode, args.seconds)
