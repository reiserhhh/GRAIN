#!/usr/bin/env python3
"""Execute a bounded four-query, three-carrier interchange experiment."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.shared_query_actions import (
    cube_terms,
    exchange,
    make_edges,
    region_masks,
    worlds,
)
from grain_research.vector_mechanisms.navigation import (
    BASE,
    MODEL,
    digest,
    load_functions,
    read_json,
    write_json,
)
from scripts.run_grain_narrow_functions import reference_view
from scripts.run_grain_navigation_baseline import checked_arrays, save_unit

REFERENCE = BASE / "narrow_functions_m30b_v1"
CONFIG = ROOT / "configs/grain_shared_query_actions.json"
PLAN = ROOT / "docs/GRAIN_SHARED_QUERY_ACTION_PLAN_V1.md"
DEFAULT = BASE / "mechanism_todo_v1/shared_query_actions_r1"
CODE = [
    "grain_research/shared_query_actions.py",
    "scripts/run_grain_shared_query_actions.py",
    "scripts/audit_grain_shared_query_actions.py",
    "grain_research/vector_mechanisms/full_state_edit.py",
    "grain_research/vector_mechanisms/context_carrier.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "grain_research/vector_mechanisms/functional_profiles.py",
    "grain_research/vector_mechanisms/trace.py",
    "scripts/run_grain_narrow_functions.py",
]


def array_hash(x):
    """Hash the exact contiguous intervention bytes, not a rounded statistic."""
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def verify(out):
    """Keep the finite question/source/implementation specification unchanged."""
    m = read_json(out / "manifest.json")
    for name, expected in m["files"].items():
        if digest(out / name) != expected:
            raise ValueError("Packet input changed: " + name)
    for name, expected in m["sources"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen source changed: " + name)
    return m


def prepare(out):
    """Freeze raw public materials and exact matching query/position geometry."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Use a new output directory")
    cfg = read_json(CONFIG)
    definitions = read_json(REFERENCE / "queries.json")
    old = read_json(REFERENCE / "manifest.json")
    frozen = read_json(REFERENCE / "reference_manifest.json")
    if digest(REFERENCE / "reference.npz") != frozen["sha256"]:
        raise ValueError("Old function parameters changed")
    if digest(MODEL / "model.safetensors") != old["model_sha256"]:
        raise ValueError("Pinned observer changed")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    raw, design = worlds(cfg), []
    for world in raw:
        for query in cfg["queries"]:
            text, spans, encoded, mask = reference_view(
                tokenizer, [world["body"]], definitions[query]["question"]
            )
            if (
                np.asarray(encoded["input_ids"])[mask].tolist()
                != old["question_tokens"][query]
            ):
                raise ValueError("Old function query coordinates differ")
            design.append(
                {
                    "index": len(design),
                    "world_id": world["id"],
                    **{k: world[k] for k in ("frame", "owner", "arousal", "dejection")},
                    "query": query,
                    "ids": encoded["input_ids"],
                    "question_mask": mask.tolist(),
                    "full_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "body_span": spans[0],
                }
            )
    edges = make_edges(design)
    if len(design) != 64 or len(edges) != 64:
        raise ValueError("Incomplete balanced factorial packet")
    out.mkdir(parents=True)
    for folder in ("states", "actions", "cross_query"):
        (out / folder).mkdir()
    for name, value in [
        ("config.json", cfg),
        ("worlds.json", raw),
        ("design.json", design),
        ("edges.json", edges),
        ("queries.json", {q: definitions[q] for q in cfg["queries"]}),
    ]:
        write_json(out / name, value)
    sources = [
        CONFIG,
        PLAN,
        REFERENCE / "queries.json",
        REFERENCE / "reference.npz",
        REFERENCE / "manifest.json",
        REFERENCE / "reference_manifest.json",
        *[ROOT / n for n in CODE],
    ]
    write_json(
        out / "manifest.json",
        {
            "protocol": cfg["protocol"],
            "files": {
                n: digest(out / n)
                for n in (
                    "config.json",
                    "worlds.json",
                    "design.json",
                    "edges.json",
                    "queries.json",
                )
            },
            "sources": {str(p.relative_to(ROOT)): digest(p) for p in sources},
            "model_sha256": old["model_sha256"],
            "alphabet": old["alphabet"],
            "units": 64,
            "edges": 64,
            "planned_calls": 560,
            "no_human_labels": True,
            "no_fit": True,
            "frozen_time_ns": time.time_ns(),
        },
    )
    write_json(
        out / "run_status.json", {"status": "PREPARED", "calls": 0, "seconds": 0.0}
    )
    print(
        {
            "stage": "prepare",
            "units": 64,
            "edges": 64,
            "calls": 560,
            "max_tokens": max(len(d["ids"]) for d in design),
        },
        flush=True,
    )


def collect(out):
    """Capture native states and replay exact donor cubes with per-cell checkpoints."""
    import torch

    from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
    from grain_research.vector_mechanisms.full_state_edit import (
        capture_boundary,
        replay_suffix,
    )
    from grain_research.vector_mechanisms.trace import DecoderObserver

    m = verify(out)
    prior = read_json(out / "run_status.json")
    if prior["status"] == "COMPLETE":
        print(prior, flush=True)
        return
    cfg = read_json(out / "config.json")
    if digest(MODEL / "model.safetensors") != m["model_sha256"]:
        raise ValueError("Observer weights changed")
    torch.set_num_threads(2)
    start = time.monotonic()
    observer = DecoderObserver.load(
        str(MODEL), device=cfg["device"], dtype="float32", max_tokens=1024
    )
    observer.model.requires_grad_(False)
    load_time = time.monotonic() - start
    write_json(out / "observer.json", observer.manifest())
    designs = {d["index"]: d for d in read_json(out / "design.json")}

    def status(label):
        value = {
            "status": label,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "load_seconds": load_time,
        }
        write_json(out / "run_status.json", value)
        return value

    def expired():
        if prior["seconds"] + time.monotonic() - start > cfg["max_seconds"]:
            print(status("PAUSED_RESOURCE_BOUNDARY"), flush=True)
            return True
        return False

    try:
        for i, d in designs.items():
            path = out / "states" / f"unit_{i:03d}.json"
            if path.exists():
                meta, _ = checked_arrays(path)
                if meta["index"] != i:
                    raise ValueError("Resume input identity mismatch")
                continue
            if expired():
                return
            state, kwargs, native = capture_boundary(
                observer, d["ids"], m["alphabet"], cfg["block"]
            )
            replay = replay_suffix(
                observer,
                state,
                cached_suffix_kwargs(observer.model, state),
                m["alphabet"],
                cfg["block"],
            )
            if abs(native["conditional_score"] - replay["conditional_score"]) > 1e-5:
                raise ValueError("Native identity replay failed")
            save_unit(
                path,
                {"index": i, "native": native, "identity_replay": replay},
                state=state[0].cpu().numpy(),
            )
            if (i + 1) % 16 == 0:
                print(status("CAPTURING"), flush=True)
        for edge in read_json(out / "edges.json"):
            _bm, bz = checked_arrays(out / "states" / f"unit_{edge['base']:03d}.json")
            _dm, dz = checked_arrays(out / "states" / f"unit_{edge['donor']:03d}.json")
            d = designs[edge["base"]]
            base = torch.tensor(bz["state"][None], device=cfg["device"])
            kwargs = cached_suffix_kwargs(observer.model, base)
            for mask in range(1, 7):
                path = out / "actions" / f"edge_{edge['index']:03d}_mask_{mask}.json"
                if path.exists():
                    old = read_json(path)
                    if old["edge"] != edge["index"] or old["mask"] != mask:
                        raise ValueError("Resume action identity mismatch")
                    continue
                if expired():
                    return
                mixed = exchange(bz["state"], dz["state"], d["question_mask"], mask)
                result = replay_suffix(
                    observer,
                    torch.tensor(mixed[None], device=cfg["device"]),
                    kwargs,
                    m["alphabet"],
                    cfg["block"],
                )
                write_json(
                    path,
                    {
                        "edge": edge["index"],
                        "mask": mask,
                        "state_sha256": array_hash(mixed),
                        "native": result,
                    },
                )
            if (edge["index"] + 1) % 8 == 0:
                print(status("INTERCHANGING"), flush=True)
        distant = cfg["queries"][-1]
        for edge in read_json(out / "edges.json"):
            if edge["query"] == distant:
                continue
            path = out / "cross_query" / f"edge_{edge['index']:03d}.json"
            if path.exists():
                if read_json(path)["edge"] != edge["index"]:
                    raise ValueError("Cross-query resume identity differs")
                continue
            if expired():
                return
            source = designs[edge["base"]]
            recipient = next(
                d
                for d in designs.values()
                if d["world_id"] == source["world_id"] and d["query"] == distant
            )
            _, rz = checked_arrays(
                out / "states" / f"unit_{recipient['index']:03d}.json"
            )
            _, bz = checked_arrays(out / "states" / f"unit_{edge['base']:03d}.json")
            _, dz = checked_arrays(out / "states" / f"unit_{edge['donor']:03d}.json")
            h = rz["state"].copy()
            h[-1] += dz["state"][-1] - bz["state"][-1]
            state = torch.tensor(h[None], device=cfg["device"])
            native = replay_suffix(
                observer,
                state,
                cached_suffix_kwargs(observer.model, state),
                m["alphabet"],
                cfg["block"],
            )
            write_json(
                path,
                {
                    "edge": edge["index"],
                    "recipient": recipient["index"],
                    "state_sha256": array_hash(h),
                    "native": native,
                },
            )
        print(status("COMPLETE"), flush=True)
    except Exception as exc:
        status("ERROR")
        write_json(
            out / "error.json", {"type": type(exc).__name__, "message": str(exc)}
        )
        raise


def evaluate(out):
    """Produce an operation map, not a person-correlation leaderboard."""
    verify(out)
    if read_json(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Finish the bounded packet before interpretation")
    cfg = read_json(out / "config.json")
    functions = load_functions(REFERENCE / "reference.npz", queries=cfg["queries"])
    designs = {d["index"]: d for d in read_json(out / "design.json")}
    data = {i: checked_arrays(out / "states" / f"unit_{i:03d}.json") for i in designs}
    levels, cells, terms = [], [], []
    for i, d in designs.items():
        meta, arr = data[i]
        mask = np.asarray(d["question_mask"], bool)
        value = functions[d["query"]].parts(arr["state"][mask], arr["state"][-1])
        levels.append(
            {
                **{
                    k: d[k]
                    for k in (
                        "index",
                        "world_id",
                        "frame",
                        "owner",
                        "query",
                        "arousal",
                        "dejection",
                    )
                },
                "native": meta["native"]["conditional_score"],
                "affine": value["score"],
                "candidate_mass": meta["native"]["candidate_mass"],
            }
        )
    for edge in read_json(out / "edges.json"):
        d = designs[edge["base"]]
        bm, bz = data[edge["base"]]
        dm, dz = data[edge["donor"]]
        values = {"native": [], "affine": []}
        for mask in range(8):
            if mask in (0, 7):
                native = (bm if mask == 0 else dm)["native"]
            else:
                action = read_json(
                    out / "actions" / f"edge_{edge['index']:03d}_mask_{mask}.json"
                )
                native = action["native"]
            mixed = exchange(bz["state"], dz["state"], d["question_mask"], mask)
            if mask not in (0, 7) and array_hash(mixed) != action["state_sha256"]:
                raise ValueError("Intervention bytes changed")
            p = np.asarray(d["question_mask"], bool)
            affine = functions[d["query"]].parts(mixed[p], mixed[-1])["score"]
            values["native"].append(native["conditional_score"])
            values["affine"].append(affine)
            cells.append(
                {
                    **edge,
                    "mask": mask,
                    "native": native["conditional_score"],
                    "affine": affine,
                    "candidate_mass": native["candidate_mass"],
                }
            )
        for reader, vector in values.items():
            terms.append({**edge, "reader": reader, **cube_terms(vector)})
    tables = {
        "levels.csv": pd.DataFrame(levels),
        "cube_cells.csv": pd.DataFrame(cells),
        "component_actions.csv": pd.DataFrame(terms),
    }
    f = tables["component_actions.csv"]
    numeric = [c for c in f if c.startswith(("I_", "phi_"))] + ["total"]
    tables["operation_map_by_frame.csv"] = (
        f.groupby(["frame", "owner", "factor", "query", "reader"])[numeric]
        .mean()
        .reset_index()
    )
    tables["operation_map.csv"] = (
        f.groupby(["owner", "factor", "query", "reader"])[numeric].mean().reset_index()
    )
    comparisons = []
    for (factor, query, reader), group in tables["operation_map.csv"].groupby(
        ["factor", "query", "reader"]
    ):
        pair = group.set_index("owner")
        comparisons.append(
            {
                "factor": factor,
                "query": query,
                "reader": reader,
                **{n: pair.loc["self", n] - pair.loc["other", n] for n in numeric},
            }
        )
    tables["ownership_interactions.csv"] = pd.DataFrame(comparisons)
    cross = []
    for edge in read_json(out / "edges.json"):
        path = out / "cross_query" / f"edge_{edge['index']:03d}.json"
        if edge["query"] == cfg["queries"][-1]:
            continue
        result = read_json(path)
        recipient = designs[result["recipient"]]
        donor_recipient = next(
            d
            for d in designs.values()
            if d["world_id"] == designs[edge["donor"]]["world_id"]
            and d["query"] == recipient["query"]
        )
        base_score = data[recipient["index"]][0]["native"]["conditional_score"]
        cross.append(
            {
                **edge,
                "recipient_query": recipient["query"],
                "cross_Q_delta": result["native"]["conditional_score"] - base_score,
                "recipient_natural_delta": data[donor_recipient["index"]][0]["native"][
                    "conditional_score"
                ]
                - base_score,
                "source_natural_delta": data[edge["donor"]][0]["native"][
                    "conditional_score"
                ]
                - data[edge["base"]][0]["native"]["conditional_score"],
            }
        )
    tables["cross_query_Q.csv"] = pd.DataFrame(cross)
    common = []
    for wid, group in tables["levels.csv"].groupby("world_id"):
        ids = group["index"].tolist()
        reference = designs[ids[0]]
        cmask = region_masks(reference["question_mask"])["C"]
        x = data[ids[0]][1]["state"][cmask]
        for i in ids[1:]:
            mask = region_masks(designs[i]["question_mask"])["C"]
            if not np.array_equal(
                np.array(reference["ids"])[cmask], np.array(designs[i]["ids"])[mask]
            ):
                raise ValueError("Queries do not share the same pre-question tokens")
            y = data[i][1]["state"][mask]
            common.append(
                {
                    "world_id": wid,
                    "query": designs[i]["query"],
                    "max_absolute_state_difference": float(abs(x - y).max()),
                    "relative_rms": float(
                        np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-12)
                    ),
                }
            )
    tables["common_context_check.csv"] = pd.DataFrame(common)
    for name, frame in tables.items():
        frame.to_csv(out / name, index=False)
    summary = {
        "status": "COMPLETE",
        "calls": read_json(out / "run_status.json"),
        "units": len(levels),
        "edges": len(f) // 2,
        "cube_rows": len(cells),
        "common_C_max_difference": max(
            r["max_absolute_state_difference"] for r in common
        ),
        "max_cube_sum_error": float(f.sum_error.max()),
        "min_candidate_mass": float(tables["cube_cells.csv"].candidate_mass.min()),
        "new_human_labels": False,
        "neural_training": False,
        "method": "three node-group interchange at one boundary, not edge-specific path identification",
    }
    write_json(out / "summary.json", summary)
    write_json(
        out / "evaluation_manifest.json",
        {"files": {n: digest(out / n) for n in [*tables, "summary.json"]}},
    )
    print(summary, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "collect", "evaluate"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    a = p.parse_args()
    globals()[a.stage](a.out.resolve())
