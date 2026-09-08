#!/usr/bin/env python3
"""Collect and analyze a bounded controlled-world packet with two reference sources."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import (
    BASE,
    KEYS,
    MODEL,
    OUTPUT,
    digest,
    load_functions,
    quality,
    read_json,
    tokenize_world,
    write_json,
)


def checked_arrays(path):
    """Check the committed unit hash before treating it as complete."""
    meta = read_json(path)
    if digest(path.with_suffix(".npz")) != meta["array_sha256"]:
        raise ValueError("Saved navigation array changed")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
        return meta, dict(z)


def save_unit(path, metadata, **arrays):
    """Commit array first and JSON marker last so interrupted units are replayable."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path.with_suffix(".npz"))
    write_json(path, {**metadata, "array_sha256": digest(path.with_suffix(".npz"))})


def construct_reference(observer, world, query):
    """Reuse the old two-context/three-node rule on a synthetic action-removal pair."""
    from grain_research.vector_mechanisms.context_carrier import (
        cached_suffix_kwargs,
        split_context_state,
    )
    from grain_research.vector_mechanisms.factor_functions import joint_gradient
    from grain_research.vector_mechanisms.full_state_edit import (
        align_excerpts,
        align_states,
        capture_boundary,
    )
    from grain_research.vector_mechanisms.query_functional import (
        integrated_matrix,
        path_state,
        region_gradient,
    )

    ft, fs, fe, fm, alphabet = tokenize_world(observer.tokenizer, world["texts"], query)
    et, es, ee, em, _ = tokenize_world(observer.tokenizer, world["texts"][1:], query)
    if not np.array_equal(
        np.asarray(fe["input_ids"])[fm], np.asarray(ee["input_ids"])[em]
    ):
        raise ValueError("Reference edit changed the fixed question")
    mapping, audit = align_excerpts(ft, et, fs, es, [1], fe, ee)
    full, full_kwargs, native = capture_boundary(observer, fe["input_ids"], alphabet)
    _, _, g = joint_gradient(observer, full, fm, full_kwargs, alphabet)
    edited, _, _ = capture_boundary(observer, ee["input_ids"], alphabet)
    aligned = align_states(full, edited, mapping)
    kwargs, gradients = cached_suffix_kwargs(observer.model, edited), []
    for e in (0, 1):
        context = aligned if e == 0 else split_context_state(aligned, edited, em, "E")
        nodes = []
        for t in (0.0, 0.5, 1.0):
            _, G = region_gradient(
                observer,
                context,
                path_state(aligned[:, em], edited[:, em], t),
                em,
                kwargs,
                alphabet,
            )
            nodes.append(G)
        gradients.append(nodes)
    arrays = {
        "Z": full[0, fm].cpu().numpy(),
        "h": full[0, -1].cpu().numpy(),
        "G": integrated_matrix(gradients),
        "g": g,
        "b": native["conditional_score"],
    }
    return arrays, audit


def collect(out, seconds=None):
    """Resume whole units only, preserving original reader/model/design identities."""
    import torch

    from grain_research.vector_mechanisms.full_state_edit import capture_boundary
    from grain_research.vector_mechanisms.trace import DecoderObserver

    manifest, prior = (
        read_json(out / "manifest.json"),
        read_json(out / "run_status.json"),
    )
    for name in ("worlds", "design"):
        if digest(out / f"{name}.json") != manifest[f"{name}_sha256"]:
            raise ValueError("Frozen controlled design changed")
    if digest(out / "real_reference.npz") != manifest["real_reference_sha256"]:
        raise ValueError("Frozen real reader changed")
    for p, sha in {**manifest["source_sha256"], **manifest["code_sha256"]}.items():
        if digest(ROOT / p) != sha:
            raise ValueError(f"Frozen code/source changed: {p}")
    if digest(MODEL / "model.safetensors") != manifest["model_sha256"]:
        raise ValueError("Observer weights changed")
    if prior["status"] == "COMPLETE":
        print({"status": "COMPLETE", "new_calls": 0}, flush=True)
        return
    lock = out / "collector.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    observer, start = None, time.monotonic()
    completed, refs = 0, 0

    def status(state):
        """Record a resumable boundary and actual logical neural evaluations."""
        record = {
            "status": state,
            "calls": prior["calls"] + (observer.calls if observer else 0),
            "seconds": prior["seconds"] + time.monotonic() - start,
            "completed_units": completed,
            "completed_references": refs,
        }
        write_json(out / "run_status.json", record)
        return record

    limit = (
        manifest["per_packet_seconds"]
        if seconds is None
        else min(seconds, manifest["per_packet_seconds"])
    )
    try:
        torch.set_num_threads(2)
        observer = DecoderObserver.load(
            str(MODEL), device="mps", dtype="float32", max_tokens=3072
        )
        observer.model.requires_grad_(False)
        observed = observer.manifest()
        expected = read_json(BASE / "complete_level_response_v1/observer.json")
        for k in (
            "torch",
            "transformers",
            "dtype",
            "device",
            "attention",
            "template_sha256",
            "pad_to_length",
        ):
            if observed[k] != expected[k]:
                raise ValueError(f"Observer changed: {k}")
        write_json(out / "observer.json", observed)
        for d in read_json(out / "design.json"):
            path = out / "packets" / f"unit_{d['index']:04d}.json"
            if path.exists():
                meta, _ = checked_arrays(path)
                if any(meta[k] != d[k] for k in ("index", "query", "world_id")):
                    raise ValueError("Cached unit identity changed")
                completed += 1
                continue
            if time.monotonic() - start >= limit:
                print(status("PAUSED_RESOURCE_BOUNDARY"), flush=True)
                return
            h, _, native = capture_boundary(observer, d["input_ids"], d["alphabet"])
            mask = np.asarray(d["mask"], bool)
            save_unit(
                path,
                {k: d[k] for k in ("index", "query", "world_id")} | {"native": native},
                P=h[0, mask].cpu().numpy(),
                Q=h[0, -1].cpu().numpy(),
            )
            completed += 1
            del h
            print(status("COLLECTING_NATIVE"), flush=True)
        worlds = {w["world_id"]: w for w in read_json(out / "worlds.json")}
        arrays = {}
        for q, ids in manifest["synthetic_references"].items():
            values = []
            for i, wid in enumerate(ids):
                world = worlds[wid]
                if world["split"] != "construction":
                    raise ValueError("Synthetic reference outside construction split")
                path = out / "synthetic_reference_units" / f"{q}_{i}.json"
                if path.exists():
                    meta, z = checked_arrays(path)
                    if meta["query"] != q or meta["world_id"] != wid:
                        raise ValueError("Synthetic reference identity changed")
                else:
                    if time.monotonic() - start >= limit:
                        print(status("PAUSED_RESOURCE_BOUNDARY"), flush=True)
                        return
                    z, audit = construct_reference(observer, world, q)
                    save_unit(
                        path, {"query": q, "world_id": wid, "alignment": audit}, **z
                    )
                values.append(z)
                refs += 1
                print(status("CONSTRUCTING_REFERENCE"), flush=True)
            arrays.update(
                {
                    f"{q}__{k}": np.mean(
                        [z[k] for z in values], axis=0, dtype=np.float64
                    )
                    for k in KEYS
                }
            )
        np.savez_compressed(out / "synthetic_reference.npz", **arrays)
        write_json(
            out / "reference_source_manifest.json",
            {
                "R_real": manifest["real_reference_sha256"],
                "R_synth": digest(out / "synthetic_reference.npz"),
                "synthetic_reference_worlds": manifest["synthetic_references"],
                "human_label_fit": False,
                "meps_construction": False,
            },
        )
        print(status("COMPLETE"), flush=True)
    except Exception as exc:
        status("ERROR")
        write_json(
            out / "error.json", {"type": type(exc).__name__, "message": str(exc)}
        )
        raise
    finally:
        lock.unlink(missing_ok=True)


def paired_effects(frame):
    """Compare exact factorial partners rather than unpaired score differences."""
    axes = {
        "actor": ("other", "self"),
        "opportunity": ("assigned", "voluntary"),
        "realization": (0, 1),
    }
    keys = ["query", "reference", "scene_id", "actor", "opportunity", "realization"]
    rows = []
    for axis, (low, high) in axes.items():
        group_keys = [k for k in keys if k != axis]
        for identity, group in frame.groupby(group_keys):
            if len(group) != 2 or set(group[axis]) != {low, high}:
                raise ValueError("Incomplete factorial response pair")
            pair = group.set_index(axis)
            row = dict(zip(group_keys, identity, strict=True))
            row.update(axis=axis, split=group.split.iloc[0])
            for col in ("native_score", "score", "mean", "configuration", "final"):
                row[col + "_delta"] = float(pair.loc[high, col] - pair.loc[low, col])
            rows.append(row)
    return pd.DataFrame(rows)


def analyze(out):
    """Describe held-scene fidelity and role/opportunity responses; no synthetic accuracy."""
    if read_json(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Finish or explicitly scope the packet before final analysis")
    refs = read_json(out / "reference_source_manifest.json")
    functions = {}
    for arm, filename in (
        ("R_real", "real_reference.npz"),
        ("R_synth", "synthetic_reference.npz"),
    ):
        if digest(out / filename) != refs[arm]:
            raise ValueError("Reference changed after freeze")
        functions[arm] = load_functions(out / filename)
    worlds = {w["world_id"]: w for w in read_json(out / "worlds.json")}
    rows = []
    for d in read_json(out / "design.json"):
        meta, z = checked_arrays(out / "packets" / f"unit_{d['index']:04d}.json")
        if any(meta[k] != d[k] for k in ("index", "query", "world_id")):
            raise ValueError("Native row mismatch")
        world = worlds[d["world_id"]]
        for arm, fs in functions.items():
            rows.append(
                {
                    k: world[k]
                    for k in (
                        "world_id",
                        "scene_id",
                        "split",
                        "semantic_family",
                        "actor",
                        "opportunity",
                        "realization",
                    )
                }
                | {
                    "query": d["query"],
                    "reference": arm,
                    "native_score": meta["native"]["conditional_score"],
                    "candidate_mass": meta["native"]["candidate_mass"],
                    **fs[d["query"]].parts(z["P"], z["Q"]),
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame) != 576 or frame.duplicated(["world_id", "query", "reference"]).any():
        raise ValueError("Incomplete or duplicate navigation scores")
    frame.to_csv(out / "native_and_function.csv", index=False)
    metrics = []
    for (q, arm, split), group in frame.groupby(["query", "reference", "split"]):
        metrics.append(
            {
                "query": q,
                "reference": arm,
                "split": split,
                "scenes": group.scene_id.nunique(),
                "candidate_mass_mean": group.candidate_mass.mean(),
                **quality(group.native_score, group.score),
            }
        )
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(out / "level_metrics.csv", index=False)
    effects = paired_effects(frame)
    effects.to_csv(out / "response_by_condition.csv", index=False)
    response = []
    for (q, arm, split, axis), group in effects.groupby(
        ["query", "reference", "split", "axis"]
    ):
        response.append(
            {
                "query": q,
                "reference": arm,
                "split": split,
                "axis": axis,
                "scenes": group.scene_id.nunique(),
                "mean_native_delta": group.native_score_delta.mean(),
                "mean_function_delta": group.score_delta.mean(),
                **quality(group.native_score_delta, group.score_delta),
            }
        )
    response = pd.DataFrame(response)
    response.to_csv(out / "response_metrics.csv", index=False)
    write_json(
        out / "summary.json",
        {
            "status": "COMPLETE_CONTROLLED_PACKET",
            "worlds": 96,
            "independent_scene_skeletons": 12,
            "held_scene_skeletons": 3,
            "queries": 3,
            "reference_sources": 2,
            "human_label_fit": False,
            "upstream_independence_tested": False,
            "native_calls_and_gradients": read_json(out / "run_status.json")["calls"],
        },
    )
    report = [
        "# NAV-01/04 controlled-world packet",
        "",
        "Synthetic event facts, not ground-truth personality scores. No human labels fitted.",
        "",
        "## Held scene levels",
        "",
        "```text",
        metrics[metrics.split == "held_scene"].round(5).to_string(index=False),
        "```",
        "",
        "## Held scene matched responses",
        "",
        "```text",
        response[response.split == "held_scene"].round(5).to_string(index=False),
        "```",
        "",
        "Twelve scene skeletons, three held. Repeated renderings are not independent people.",
        "Scores are unclipped affine readouts. Report level, response and human linkage separately.",
        "Real-text transport and upstream reconstruction have their own packets; this is not an independent raw-text runtime.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(
        metrics[metrics.split == "held_scene"].round(4).to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "analyze"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to((BASE / "navigation_v1").resolve()):
        parser.error("Use the local ignored navigation root")
    if args.seconds is not None and args.seconds <= 0:
        parser.error("Positive packet seconds required")
    collect(out, args.seconds) if args.mode == "collect" else analyze(out)
