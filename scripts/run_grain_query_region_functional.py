#!/usr/bin/env python3
"""Bounded question-region functional matrices on the frozen four-author pilot."""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.context_carrier import (
    cached_suffix_kwargs,
    split_context_state,
)
from grain_research.vector_mechanisms.full_state_edit import align_states
from grain_research.vector_mechanisms.query_functional import (
    frozen_reference_matrix,
    hidden_subspace_overlap,
    integrated_matrix,
    matrix_response,
    matrix_variants,
    path_state,
    region_gradient,
    response_quality,
)
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write

PARENT = BASE / "context_carrier_split_v1"
FULL = BASE / "full_state_edit_connection_v1"
NODES = (0.0, 0.5, 1.0)


def prepare(out):
    """Freeze cached state lineage and endpoint targets without reading author text."""
    if read(PARENT / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete carrier split required")
    design = read(PARENT / "private_design.json")
    if (
        sha(PARENT / "private_design.json")
        != read(PARENT / "manifest.json")["input_sha256"]
    ):
        raise ValueError("Carrier design changed")
    if len(design) != 12 or len({d["author_id"] for d in design}) != 4:
        raise ValueError("Use the fixed four-author panel only")
    for d in design:
        source = ROOT / d["parent_packet"]
        target = PARENT / "private_packets" / f"unit_{d['index']:02d}_{d['query']}.json"
        if (
            sha(source) != d["packet_sha256"]
            or sha(source.with_suffix(".npz")) != d["arrays_sha256"]
        ):
            raise ValueError("Full-state source changed")
        t = read(target)
        if (
            t["author_id"] != d["author_id"]
            or t["parent_packet_sha256"] != d["packet_sha256"]
        ):
            raise ValueError("Target linkage changed")
        d["target_packet"] = str(target.relative_to(ROOT))
        d["target_sha256"] = sha(target)
    out.mkdir(parents=True, exist_ok=False)
    for sub in ("private_nodes", "private_packets"):
        (out / sub).mkdir()
    write(out / "private_design.json", design)
    files = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/query_functional.py",
        ROOT / "grain_research/vector_mechanisms/context_carrier.py",
        ROOT / "grain_research/vector_mechanisms/full_state_edit.py",
        ROOT / "docs/GRAIN_QUERY_REGION_FUNCTIONAL_PLAN_V1.md",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_QUERY_REGION_FUNCTIONAL_V1",
            "authors": 4,
            "units": 12,
            "reference_authors": 2,
            "evaluation_authors": 2,
            "nodes": list(NODES),
            "evidence_conditions": [0, 1],
            "max_forward_backward_pairs": 72,
            "max_seconds": 900,
            "prefix_calls": 0,
            "labels_read": False,
            "target": "symmetric P allocation conditional on Q0, not full natural response",
            "input_sha256": sha(out / "private_design.json"),
            "parent_manifest_sha256": sha(PARENT / "manifest.json"),
            "model_sha256": read(PARENT / "manifest.json")["model_sha256"],
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "controls": ["zero", "reference_mean", "token_mean", "rank1", "rank4"],
            "cross_query_alignment": "right singular subspaces in same hidden coordinates, rank4; no token matching across queries",
        },
    )
    write(
        out / "run_status.json",
        {"status": "PREPARED", "calls": 0, "seconds": 0.0, "completed_units": 0},
    )
    print({"status": "PREPARED", "units": 12, "max_pairs": 72}, flush=True)


def verify_inputs(out):
    """Validate immutable inputs; completed analysis does not require current code hashes."""
    manifest = read(out / "manifest.json")
    if sha(out / "private_design.json") != manifest["input_sha256"]:
        raise ValueError("Frozen design changed")
    for d in read(out / "private_design.json"):
        for name, digest in (
            (d["parent_packet"], d["packet_sha256"]),
            (d["target_packet"], d["target_sha256"]),
        ):
            if sha(ROOT / name) != digest:
                raise ValueError("Endpoint packet changed")
        if sha((ROOT / d["parent_packet"]).with_suffix(".npz")) != d["arrays_sha256"]:
            raise ValueError("State array changed")
    return manifest


def run(out):
    """Collect at most 72 suffix forward/backward pairs, checkpointing every node."""
    manifest = verify_inputs(out)
    prior = read(out / "run_status.json")
    if prior["status"] == "COMPLETE":
        analyze(out)
        return
    for name, digest in manifest["code_hashes"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Frozen implementation changed")
    if sha(MODEL / "model.safetensors") != manifest["model_sha256"]:
        raise ValueError("Model changed")
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    current, old = observer.manifest(), read(FULL / "observer.json")
    for name in (
        "torch",
        "transformers",
        "dtype",
        "device",
        "attention",
        "template_sha256",
        "pad_to_length",
    ):
        if current[name] != old[name]:
            raise ValueError(f"Observer changed: {name}")
    write(out / "observer.json", current)
    device = next(observer.model.parameters()).device
    start, complete = time.monotonic(), 0

    def status(state):
        value = {
            "status": state,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "completed_units": complete,
            "prefix_calls": 0,
            "labels_read": False,
            "call_unit": "suffix forward/backward pair",
        }
        write(out / "run_status.json", value)
        return value

    try:
        for d in read(out / "private_design.json"):
            stem = f"unit_{d['index']:02d}_{d['query']}"
            dst = out / "private_packets" / f"{stem}.json"
            if dst.exists():
                r = read(dst)
                if (
                    r["arrays_sha256"] != sha(dst.with_suffix(".npz"))
                    or r["source_sha256"] != d["packet_sha256"]
                ):
                    raise ValueError("Completed packet changed")
                complete += 1
                continue
            with np.load(
                (ROOT / d["parent_packet"]).with_suffix(".npz"), allow_pickle=False
            ) as z:
                full = torch.tensor(z["full_state"][None], device=device)
                edited = torch.tensor(z["edited_state"][None], device=device)
                aligned = align_states(full, edited, z["mapping"])
            mask = np.asarray(d["question_mask"], bool)
            z0, z1 = aligned[:, mask].detach(), edited[:, mask].detach()
            displacement = (z1 - z0)[0].cpu().numpy().copy()
            kwargs = cached_suffix_kwargs(observer.model, edited)
            scores, gradients = np.zeros((2, 3)), []
            for e in (0, 1):
                context = (
                    aligned
                    if e == 0
                    else split_context_state(aligned, edited, mask, "E")
                )
                gs = []
                for n, t in enumerate(NODES):
                    node = out / "private_nodes" / f"{stem}_e{e}_n{n}.json"
                    if node.exists():
                        cached = read(node)
                        if (
                            cached["arrays_sha256"] != sha(node.with_suffix(".npz"))
                            or cached["source_sha256"] != d["packet_sha256"]
                        ):
                            raise ValueError("Cached node changed")
                        with np.load(node.with_suffix(".npz"), allow_pickle=False) as a:
                            g = a["gradient"].copy()
                        value = cached["score"]
                    else:
                        if (
                            prior["calls"] + observer.calls
                            >= manifest["max_forward_backward_pairs"]
                            or prior["seconds"] + time.monotonic() - start
                            >= manifest["max_seconds"]
                        ):
                            print(status("PAUSED_RESOURCE_BOUNDARY"), flush=True)
                            return
                        value, g = region_gradient(
                            observer,
                            context,
                            path_state(z0, z1, t),
                            mask,
                            kwargs,
                            d["alphabet"],
                        )
                        np.savez_compressed(node.with_suffix(".npz"), gradient=g)
                        write(
                            node,
                            {
                                "score": value,
                                "source_sha256": d["packet_sha256"],
                                "arrays_sha256": sha(node.with_suffix(".npz")),
                            },
                        )
                        status("RUNNING")
                    scores[e, n] = value
                    gs.append(g)
                gradients.append(gs)
            carrier = read(ROOT / d["target_packet"])
            expected = np.array(
                [
                    [carrier["origin"], carrier["new_cells"]["P"]["conditional_score"]],
                    [
                        carrier["new_cells"]["E"]["conditional_score"],
                        carrier["joint_context_endpoint"],
                    ],
                ]
            )
            error = float(np.max(np.abs(scores[:, [0, 2]] - expected)))
            if error > 2e-6:
                raise ValueError(
                    "Differentiable endpoints do not replay observed E/P cells"
                )
            g = integrated_matrix(gradients)
            np.savez_compressed(
                dst.with_suffix(".npz"),
                gradients=np.asarray(gradients),
                integrated=g,
                displacement=displacement,
                node_scores=scores,
            )
            write(
                dst,
                {
                    "index": d["index"],
                    "author_id": d["author_id"],
                    "role": d["role"],
                    "query": d["query"],
                    "source_sha256": d["packet_sha256"],
                    "arrays_sha256": sha(dst.with_suffix(".npz")),
                    "endpoint_error": error,
                    "actual": carrier["components"]["query_allocated"],
                    "simpson": matrix_response(g, displacement),
                    "parameter_gradients": any(
                        p.grad is not None for p in observer.model.parameters()
                    ),
                },
            )
            complete += 1
            print(status("RUNNING"), flush=True)
            del full, edited, aligned, z0, z1, context, kwargs
        print(status("COMPLETE"), flush=True)
    except BaseException as exc:
        print(status("FAILED_RETAINED"), flush=True)
        write(out / "error_log.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    analyze(out)


def analyze(out):
    """Compare fixed readouts, token contributions and hidden subspaces without labels."""
    verify_inputs(out)
    if read(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Only analyze the complete frozen packet")
    records, arrays = [], []
    for d in read(out / "private_design.json"):
        path = out / "private_packets" / f"unit_{d['index']:02d}_{d['query']}.json"
        r = read(path)
        if sha(path.with_suffix(".npz")) != r["arrays_sha256"]:
            raise ValueError("Output arrays changed")
        records.append(r)
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as a:
            arrays.append({k: a[k].copy() for k in a.files})
    frame = pd.DataFrame(records)
    matrices = [a["integrated"] for a in arrays]
    rows, tokens, compression, references = [], [], [], {}
    for q in sorted(frame["query"].unique()):
        ids = np.flatnonzero(frame["query"].to_numpy() == q)
        group = frame.iloc[ids]
        g = frozen_reference_matrix(
            np.stack([matrices[i] for i in ids]).astype(float), group.role
        )
        references[q] = g
        variants, info = matrix_variants(g)
        compression.append({"query": q, **info, "tokens": len(g)})
        ref_mean = float(group.loc[group.role == "reference", "actual"].mean())
        for i in ids:
            r, a = records[i], arrays[i]
            predictions = {
                k: matrix_response(v, a["displacement"]) for k, v in variants.items()
            }
            predictions.update(
                zero=0.0, reference_mean=ref_mean, local_simpson=r["simpson"]
            )
            for method, pred in predictions.items():
                rows.append(
                    {
                        "index": r["index"],
                        "role": r["role"],
                        "query": q,
                        "method": method,
                        "actual": r["actual"],
                        "predicted": pred,
                        "error": pred - r["actual"],
                    }
                )
            local = np.sum(a["integrated"] * a["displacement"], axis=1)
            transported = np.sum(g * a["displacement"], axis=1)
            for j, (v, t) in enumerate(zip(local, transported, strict=True)):
                tokens.append(
                    {
                        "index": r["index"],
                        "role": r["role"],
                        "query": q,
                        "question_token_position": j,
                        "local_contribution": v,
                        "frozen_contribution": t,
                    }
                )
    pred = pd.DataFrame(rows)
    pred.to_csv(out / "private_predictions.csv", index=False)
    pd.DataFrame(tokens).to_csv(out / "private_token_contributions.csv", index=False)
    metrics = []
    for (q, role, method), group in pred.groupby(["query", "role", "method"]):
        metrics.append(
            {
                "query": q,
                "role": role,
                "method": method,
                **response_quality(group.actual, group.predicted),
            }
        )
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(compression).to_csv(out / "compression.csv", index=False)
    sharing = [
        {
            "first": a,
            "second": b,
            "rank4_hidden_overlap": hidden_subspace_overlap(
                references[a], references[b]
            ),
        }
        for a, b in itertools.combinations(sorted(references), 2)
    ]
    pd.DataFrame(sharing).to_csv(out / "query_hidden_overlap.csv", index=False)
    np.savez_compressed(out / "reference_matrices.npz", **references)
    summary = {
        "status": "COMPLETE",
        "authors": 4,
        "evaluation_authors": 2,
        "scope": "Four-author mechanism panel; error/sign results, no population correlations",
        "target": "P allocation conditional on Q0, not natural Delta or questionnaire scores",
        "cost": read(out / "run_status.json"),
        "metrics": metrics,
        "compression": compression,
        "query_hidden_overlap": sharing,
        "max_endpoint_replay_error": float(frame.endpoint_error.max()),
        "max_simpson_error": float(np.max(np.abs(frame.simpson - frame.actual))),
        "parameter_gradients": bool(frame.parameter_gradients.any()),
    }
    write(out / "summary.json", summary)
    print(
        pd.DataFrame(metrics).query("role == 'evaluation'").to_string(index=False),
        flush=True,
    )


def main():
    """Expose preparation, bounded execution and cache-only analysis separately."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "analyze"))
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "query_region_functional_v1"
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError("Private output must remain inside ignored results")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    {"prepare": prepare, "run": run, "analyze": analyze}[args.phase](out)


if __name__ == "__main__":
    main()
