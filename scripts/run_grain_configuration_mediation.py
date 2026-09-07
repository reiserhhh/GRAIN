#!/usr/bin/env python3
"""S24: transport the actual source body-configuration contrast through one head."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.configuration_mediation import (
    endpoint_errors,
    factorial_allocation,
    projected_body,
)
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.cross_query_functional import effect_quality
from grain_research.vector_mechanisms.direct_traits import candidates, tokenize_prompt
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_teacher_student_bridge import paired_intervals
from scripts.run_grain_attention_pairing import capture_head, patch_head_score
from scripts.run_grain_functional_transfer import verify as verify_s20
from scripts.run_grain_multiquery_sensitivity import token_groups
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write
from scripts.run_grain_source_remainder import verify_prepared as verify_s9


def sources(s20, s9, device):
    """Match the frozen transfer panel to complete S9 material by unique author ID."""
    m, _s14, _s12, material, _scores = verify_s20(s20, device)
    _source_manifest, source = verify_s9(s9, device)
    by_id = {a["author_id"]: a for a in source}
    if len(by_id) != len(source) or m["intervention_indices"] != list(range(24, 48)):
        raise ValueError("Unique S9 author IDs and fixed S20 panel required")
    selected = []
    for i in m["intervention_indices"]:
        author = material[i]
        old = by_id[author["author_id"]]
        if old["query"] != "neuroticism" or old["views"] != author["views"]:
            raise ValueError("S9 and S20 exact source text views differ")
        path = s9 / "private_packets" / f"author_{old['index']:03d}.json"
        packet = read(path)
        if (
            packet["author_id"] != author["author_id"]
            or packet["query"] != old["query"]
        ):
            raise ValueError("S9 endpoint packet identity differs")
        view = [r for r in packet["views"] if r["view"] == 0]
        if len(view) != 1:
            raise ValueError("Unique S9 view0 required")
        selected.append((i, author, path, view[0]))
    return selected


def prepare(args):
    """Freeze the no-outcome four-cell protocol and source replay endpoints."""
    panel = sources(args.source, args.s9, args.device)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    files = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/configuration_mediation.py",
        ROOT / "docs/GRAIN_CONFIGURATION_MEDIATION_S24_PLAN.md",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_S24_CONFIGURATION_MEDIATION_V1",
            "s20": str(args.source),
            "s9": str(args.s9),
            "s20_manifest_sha256": sha(args.source / "manifest.json"),
            "s20_summary_sha256": sha(args.source / "summary.json"),
            "s9_manifest_sha256": sha(args.s9 / "manifest.json"),
            "s9_packet_hashes": {str(p): sha(p) for _, _, p, _ in panel},
            "reference_sha256": sha(args.source / "private_reference.npz"),
            "membership_sha256": material_hash(
                [a["author_id"] for _, a, _, _ in panel]
            ),
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "device": args.device,
            "authors": 24,
            "view": 0,
            "indices": list(range(24, 48)),
            "body_layer": 16,
            "head_layer": 17,
            "cell_order": ["F11", "F00", "F10", "F01"],
            "max_forwards": 96,
            "max_seconds": 300,
            "source_outcomes_read": False,
            "gradient_fitted": False,
        },
    )
    print({"status": "PREPARED", "authors": 24, "max_forwards": 96}, flush=True)


def verify(out, device):
    """Recheck frozen sources, functional reader, code and author membership."""
    m = read(out / "manifest.json")
    s20, s9 = Path(m["s20"]), Path(m["s9"])
    checks = [
        (s20 / "manifest.json", m["s20_manifest_sha256"]),
        (s20 / "summary.json", m["s20_summary_sha256"]),
        (s9 / "manifest.json", m["s9_manifest_sha256"]),
        (s20 / "private_reference.npz", m["reference_sha256"]),
        *[(Path(p), h) for p, h in m["s9_packet_hashes"].items()],
        *[(ROOT / p, h) for p, h in m["code_hashes"].items()],
    ]
    if device != m["device"] or any(sha(p) != h for p, h in checks):
        raise ValueError("Frozen S24 source or protocol changed")
    panel = sources(s20, s9, device)
    if (
        material_hash([a["author_id"] for _, a, _, _ in panel])
        != m["membership_sha256"]
    ):
        raise ValueError("Frozen S24 author membership changed")
    return m, panel


def validate_packet(path, author, endpoints):
    """Validate a resumable complete four-cell author record."""
    row = read(path)
    if row["author_id"] != author["author_id"] or row["view"] != 0:
        raise ValueError("Resumed packet identity differs")
    if sha(path.with_suffix(".npz")) != row["arrays_sha256"]:
        raise ValueError("Resumed head arrays changed")
    endpoint_errors(row["cells"], endpoints["new_teacher"], endpoints["mean_score"])
    return row


def run(args):
    """Collect two endpoint heads and two crossed patches; no backpropagation."""
    out = args.output_dir
    m, panel = verify(out, args.device)
    previous = (
        read(out / "run_status.json")
        if (out / "run_status.json").exists()
        else {"calls": 0, "seconds": 0}
    )
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device=args.device, dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    alphabet = candidates("neuroticism", observer.tokenizer)
    with np.load(Path(m["s20"]) / "private_reference.npz", allow_pickle=False) as z:
        gradient = z["reference"]
    start = time.monotonic()

    def status(name):
        value = {
            "status": name,
            "authors": len(list((out / "private_packets").glob("author_*.json"))),
            "calls": previous["calls"] + observer.calls,
            "seconds": previous["seconds"] + time.monotonic() - start,
            "backwards": 0,
            "source_outcomes_read": False,
        }
        write(out / "run_status.json", value)
        print(value, flush=True)

    try:
        for ordinal, (i, author, _old_path, endpoints) in enumerate(panel):
            path = out / "private_packets" / f"author_{i:03d}.json"
            if path.exists():
                validate_packet(path, author, endpoints)
                continue
            if ordinal >= args.max_authors:
                status("PAUSED_PILOT_BOUNDARY")
                return
            if (
                previous["calls"] + observer.calls + 4 > m["max_forwards"]
                or previous["seconds"] + time.monotonic() - start >= m["max_seconds"]
            ):
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            payload = author["views"][0]
            ids, user, spans = tokenize_prompt(
                payload["texts"], "neuroticism", observer.tokenizer
            )
            if material_hash(ids) != payload["prompt_sha256"]:
                raise ValueError("Frozen source prompt changed")
            groups = token_groups(observer, user, spans, ids)
            f11, cache = capture_head(observer, ids, alphabet, layer=m["head_layer"])
            h1 = cache["native_head"].reshape(-1).copy()
            del cache
            with projected_body(observer.model.model.layers[m["body_layer"]], groups):
                f00, cache = capture_head(
                    observer, ids, alphabet, layer=m["head_layer"]
                )
                h0 = cache["native_head"].reshape(-1).copy()
                del cache
                f01 = patch_head_score(
                    observer, ids, alphabet, h1, layer=m["head_layer"]
                )
            f10 = patch_head_score(observer, ids, alphabet, h0, layer=m["head_layer"])
            cells = [f11, f00, f10, f01]
            replay = endpoint_errors(
                cells, endpoints["new_teacher"], endpoints["mean_score"]
            )
            delta = h1.astype(float) - h0.astype(float)
            np.savez_compressed(
                path.with_suffix(".npz"),
                native_head=h1,
                projected_head=h0,
                delta_head=delta,
            )
            write(
                path,
                {
                    "author_id": author["author_id"],
                    "source_index": i,
                    "view": 0,
                    "cells": cells,
                    "effects": {
                        k: float(v) for k, v in factorial_allocation(cells).items()
                    },
                    "linear_prediction": float(gradient @ delta),
                    "delta_head_norm": float(np.linalg.norm(delta)),
                    "replay": replay,
                    "arrays_sha256": sha(path.with_suffix(".npz")),
                },
            )
            status("RUNNING")
        status("COMPLETE")
    except Exception as exc:
        status("FAILED")
        write(out / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def inspect(args):
    """Report the bounded pilot's replay/cost without joining any outcomes."""
    m, panel = verify(args.output_dir, args.device)
    packets = []
    for i, author, _, endpoint in panel:
        path = args.output_dir / "private_packets" / f"author_{i:03d}.json"
        if path.exists():
            packets.append(validate_packet(path, author, endpoint))
    state = read(args.output_dir / "run_status.json")
    info = {
        **state,
        "max_endpoint_error": max(
            (max(r["replay"].values()) for r in packets), default=None
        ),
        "max_additive_error": max(
            (abs(r["effects"]["additive_error"]) for r in packets), default=None
        ),
        "projected_total_seconds": state["seconds"] * m["authors"] / len(packets)
        if packets
        else None,
    }
    write(args.output_dir / f"checkpoint_{len(packets):03d}.json", info)
    print(info, flush=True)


def evaluate(args):
    """Report signed exact allocations and frozen-gradient transport on all24."""
    out = args.output_dir
    _m, panel = verify(out, args.device)
    state = read(out / "run_status.json")
    if state["status"] != "COMPLETE" or state["authors"] != 24:
        raise ValueError("Complete24-author panel required")
    if (out / "summary.json").exists():
        raise ValueError("Preserve completed S24 analysis")
    packets = [
        validate_packet(out / "private_packets" / f"author_{i:03d}.json", a, e)
        for i, a, _, e in panel
    ]
    frame = pd.DataFrame(
        [
            {
                "author_id": r["author_id"],
                **r["effects"],
                "linear_prediction": r["linear_prediction"],
                **dict(zip(["F11", "F00", "F10", "F01"], r["cells"], strict=True)),
            }
            for r in packets
        ]
    )
    frame.to_csv(out / "private_effects.csv", index=False)
    rows = []
    for column in [
        "head_native_body",
        "head_projected_body",
        "head_allocation",
        "other_allocation",
        "interaction",
        "linear_prediction",
    ]:
        value = frame[column].to_numpy()
        target = frame.configuration.to_numpy()
        rows.append(
            {
                "component": column,
                "rms": float(np.sqrt(np.mean(value**2))),
                "mean": float(np.mean(value)),
                **effect_quality(target, value),
                "paired_vs_configuration": paired_intervals(
                    target, value, target, False
                ),
            }
        )
    predictions = {}
    for target in ["head_native_body", "head_projected_body", "head_allocation"]:
        y, p = frame[target].to_numpy(), frame.linear_prediction.to_numpy()
        predictions[target] = {
            **effect_quality(y, p),
            "paired_vs_exact": paired_intervals(y, p, y, False),
        }
    summary = {
        "status": "COMPLETE",
        "manifest_sha256": sha(out / "manifest.json"),
        "runtime": state,
        "authors": 24,
        "source_outcomes_read": False,
        "configuration_rms": float(np.sqrt(np.mean(frame.configuration**2))),
        "configuration_mean": float(frame.configuration.mean()),
        "max_endpoint_replay_error": max(max(p["replay"].values()) for p in packets),
        "max_additive_error": float(frame.additive_error.abs().max()),
        "components": rows,
        "frozen_gradient_transport": predictions,
        "effects_sha256": sha(out / "private_effects.csv"),
        "scope": "declared two-factor mixed-state game; signed allocations, not human causal variance",
    }
    pd.DataFrame(
        [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in rows]
    ).to_csv(out / "metrics.csv", index=False)
    write(out / "summary.json", summary)
    print(
        {
            k: v
            for k, v in summary.items()
            if k not in {"effects_sha256", "manifest_sha256"}
        },
        flush=True,
    )


def main():
    """Prepare, collect an eight-author resource pilot, resume24, then evaluate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run", "inspect", "evaluate"])
    parser.add_argument(
        "--source", type=Path, default=BASE / "functional_transfer_s20_r1"
    )
    parser.add_argument("--s9", type=Path, default=BASE / "source_remainder_s9_r1")
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "configuration_mediation_s24_r1"
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-authors", type=int, default=8)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.source, args.s9 = args.source.resolve(), args.s9.resolve()
    if not args.output_dir.is_relative_to(BASE) or not 1 <= args.max_authors <= 24:
        raise ValueError("Private output and bounded author count required")
    globals()[args.mode](args)


if __name__ == "__main__":
    main()
