#!/usr/bin/env python3
"""S19: bounded actual head gradients and held-author response sharing."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.direct_traits import candidates, tokenize_prompt
from grain_research.vector_mechanisms.head_gradient import (
    leave_author_out_geometry,
    projection_gradient,
    simpson_gradient,
)
from grain_research.vector_mechanisms.reader_pattern import split_reader_component
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.analyze_grain_teacher_student_bridge import paired_intervals
from scripts.run_grain_attention_pairing import capture_head
from scripts.run_grain_multiquery_sensitivity import conditional_torch, token_groups
from scripts.run_grain_query_representation import pairing_vector
from scripts.run_grain_reader_pattern import verify as verify_s18
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write


def prepare(args):
    """Freeze the complete S18 panel and numerical path before collecting gradients."""
    source, out = args.source.resolve(), args.output_dir
    _m, _s14, _s12, material = verify_s18(source, args.device)
    if read(source / "summary.json")["status"] != "COMPLETE":
        raise ValueError("Complete S18 intervention source required")
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    files = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/head_gradient.py",
        ROOT / "docs/GRAIN_FUNCTIONAL_GRADIENT_S19_PLAN.md",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_S19_FUNCTIONAL_GRADIENT_V1",
            "source": str(source),
            "source_manifest_sha256": sha(source / "manifest.json"),
            "source_summary_sha256": sha(source / "summary.json"),
            "source_packet_hashes": {
                f"author_{i:03d}.json": sha(
                    source / "private_packets" / f"author_{i:03d}.json"
                )
                for i in range(24)
            },
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "membership_sha256": material_hash([a["author_id"] for a in material]),
            "device": args.device,
            "authors": 24,
            "max_forwards": 144,
            "max_backwards": 120,
            "max_seconds": 900,
            "ranks": [1, 2, 4],
            "questionnaire_access": False,
            "first_checkpoint_authors": 8,
        },
    )
    print({"status": "PREPARED", "authors": 24, "pilot": 8}, flush=True)


def verify(out, device):
    """Revalidate source lineage, panel and the frozen gradient implementation."""
    m = read(out / "manifest.json")
    source = Path(m["source"])
    if (
        m["device"] != device
        or sha(source / "manifest.json") != m["source_manifest_sha256"]
        or sha(source / "summary.json") != m["source_summary_sha256"]
        or any(
            sha(source / "private_packets" / p) != h
            for p, h in m["source_packet_hashes"].items()
        )
        or any(sha(ROOT / p) != h for p, h in m["code_hashes"].items())
    ):
        raise ValueError("Frozen S19 inputs changed")
    _m, _s14, s12, material = verify_s18(source, device)
    if material_hash([a["author_id"] for a in material]) != m["membership_sha256"]:
        raise ValueError("Source author membership changed")
    return m, source, s12, material


def run(args):
    """Collect three-node gradients, adding two nodes only for completeness error."""
    out = args.output_dir
    manifest, source, s12, material = verify(out, args.device)
    previous = (
        read(out / "run_status.json")
        if (out / "run_status.json").exists()
        else {"calls": 0, "backwards": 0, "seconds": 0}
    )
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device=args.device, dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    device = next(observer.model.parameters()).device
    alphabet = candidates("neuroticism", observer.tokenizer)
    start, backwards = time.monotonic(), 0

    def status(name):
        state = {
            "status": name,
            "calls": previous["calls"] + observer.calls,
            "backwards": previous["backwards"] + backwards,
            "seconds": previous["seconds"] + time.monotonic() - start,
            "authors": len(list((out / "private_packets").glob("author_*.json"))),
            "labels_read": False,
        }
        write(out / "run_status.json", state)
        print(state, flush=True)

    try:
        for i, author in enumerate(material):
            path = out / "private_packets" / f"author_{i:03d}.json"
            if path.exists():
                packet = read(path)
                if (
                    packet["author_id"] != author["author_id"]
                    or sha(path.with_suffix(".npz")) != packet["arrays_sha256"]
                ):
                    raise ValueError("Resume author or gradient artifact changed")
                continue
            if (
                i >= args.max_authors
                or previous["seconds"] + time.monotonic() - start
                >= manifest["max_seconds"]
                or previous["calls"] + observer.calls + 6 > manifest["max_forwards"]
                or previous["backwards"] + backwards + 5 > manifest["max_backwards"]
            ):
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            old = read(source / "private_packets" / f"author_{i:03d}.json")
            payload = author["views"][0]
            ids, user, spans = tokenize_prompt(
                payload["texts"], "neuroticism", observer.tokenizer
            )
            if material_hash(ids) != payload["prompt_sha256"]:
                raise ValueError("Frozen source prompt changed")
            groups = token_groups(observer, user, spans, ids)
            native, cache = capture_head(observer, ids, alphabet)
            d, head_error = pairing_vector(cache, groups)
            h = cache["native_head"].reshape(-1)
            with np.load(
                s12 / "private_packets" / f"author_{i:03d}.npz", allow_pickle=False
            ) as z:
                feature_error = float(np.max(np.abs(d - z["kv_pairing"][0])))
            if feature_error > 1e-8 or abs(native - old["native"]) > 2e-5:
                raise ValueError("Native feature/score replay failed")
            tokens = torch.tensor([ids], device=device)

            def forward(tokens=tokens):
                observer.calls += 1
                logits = observer.model(
                    input_ids=tokens,
                    attention_mask=torch.ones_like(tokens),
                    use_cache=False,
                    logits_to_keep=1,
                    return_dict=True,
                ).logits[0, -1]
                return conditional_torch(logits, alphabet)

            gradients, scores = {}, {}

            def node(t, h=h, d=d, scores=scores, gradients=gradients, forward=forward):
                nonlocal backwards
                donor = torch.tensor(h - t * d, dtype=torch.float32, device=device)
                scores[t], gradients[t] = projection_gradient(
                    observer.model.model.layers[17].self_attn.o_proj, donor, forward
                )
                backwards += 1

            for t in (0.0, 0.5, 1.0):
                node(t)
            first = simpson_gradient(gradients)
            effect = old["native"] - old["full_removed"]
            first_gap = float(first @ d - effect)
            if abs(first_gap) > max(1e-5, 0.05 * abs(effect)):
                for t in (0.25, 0.75):
                    node(t)
            integrated = simpson_gradient(gradients)
            endpoint_error = max(
                abs(scores[0.0] - old["native"]), abs(scores[1.0] - old["full_removed"])
            )
            if endpoint_error > 2e-5 or any(
                p.grad is not None or p.requires_grad
                for p in observer.model.parameters()
            ):
                raise ValueError(
                    "Endpoint replay or frozen-parameter gradient check failed"
                )
            times = sorted(gradients)
            np.savez_compressed(
                path.with_suffix(".npz"),
                h=h,
                d=d,
                times=times,
                gradients=np.asarray([gradients[t] for t in times]),
                integrated=integrated,
                three_node_integral=first,
            )
            write(
                path,
                {
                    "author_id": author["author_id"],
                    "fold": old["fold"],
                    "view": 0,
                    "effect": effect,
                    "nodes": len(times),
                    "scores": {str(t): scores[t] for t in times},
                    "three_node_gap": first_gap,
                    "final_gap": float(integrated @ d - effect),
                    "endpoint_error": endpoint_error,
                    "feature_error": feature_error,
                    "head_error": head_error,
                    "parameter_gradients": False,
                    "arrays_sha256": sha(path.with_suffix(".npz")),
                },
            )
            status("RUNNING")
        status("COMPLETE")
    except BaseException as exc:
        status("FAILED")
        write(out / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def quality(y, prediction):
    """Report signed finite-effect reconstruction without calibration."""
    y, p = np.asarray(y, float), np.asarray(prediction, float)
    energy = float(np.mean(y**2))
    return {
        "r": scalar_metrics(y, p)["pearson_r"],
        "mse_over_effect_energy": float(np.mean((p - y) ** 2) / energy),
        "mean_error": float(np.mean(p - y)),
        "sign_agreement": float(np.mean(np.sign(p) == np.sign(y))),
    }


def analyze(args):
    """Inspect label-free pilot or report full fixed geometry without outcome joins."""
    out = args.output_dir
    manifest, source, _s12, material = verify(out, args.device)
    packets, arrays = [], []
    for i, author in enumerate(material):
        path = out / "private_packets" / f"author_{i:03d}.json"
        if not path.exists():
            break
        packet = read(path)
        if (
            packet["author_id"] != author["author_id"]
            or sha(path.with_suffix(".npz")) != packet["arrays_sha256"]
        ):
            raise ValueError("Complete ordered immutable gradient packets required")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
            arrays.append({name: z[name] for name in z.files})
        packets.append(packet)
    if len(packets) < 8:
        raise ValueError("At least the fixed8-author checkpoint required")
    g = np.asarray([a["integrated"] for a in arrays])
    d = np.asarray([a["d"] for a in arrays])
    g0 = np.asarray([a["gradients"][0] for a in arrays])
    y = np.asarray([p["effect"] for p in packets])
    with threadpool_limits(limits=2):
        rows = leave_author_out_geometry(g, d, tuple(manifest["ranks"]))
        _u, singular, _vh = np.linalg.svd(g, full_matrices=False)
    frame = pd.DataFrame(rows)
    frame["native_gradient_prediction"] = np.sum(g0 * d, axis=1)
    frame["effect"] = y
    frame["author_id"] = [p["author_id"] for p in packets]
    metrics = {
        name: quality(y, frame[name])
        for name in [
            "self_integral_prediction",
            "native_gradient_prediction",
            "loo_mean_prediction",
            "rank1_diagnostic",
            "rank2_diagnostic",
            "rank4_diagnostic",
        ]
    }
    unit = g / np.linalg.norm(g, axis=1, keepdims=True)
    cosine = unit @ unit.T
    energy = singular**2 / np.sum(singular**2)
    summary = {
        "status": "PILOT" if args.mode == "inspect" else "COMPLETE",
        "n": len(g),
        "labels_read": False,
        "metrics": metrics,
        "maximum_completeness_gap": max(abs(p["final_gap"]) for p in packets),
        "maximum_endpoint_error": max(p["endpoint_error"] for p in packets),
        "five_node_authors": sum(p["nodes"] == 5 for p in packets),
        "median_pairwise_gradient_cosine": float(
            np.median(cosine[np.triu_indices(len(g), 1)])
        ),
        "gradient_norm_cv": float(
            np.std(np.linalg.norm(g, axis=1)) / np.mean(np.linalg.norm(g, axis=1))
        ),
        "spectrum_energy": energy.tolist(),
        "gradient_projection_error": {
            f"rank{r}": float(frame[f"rank{r}_gradient_error"].mean())
            for r in manifest["ranks"]
        },
    }
    if args.mode == "inspect":
        path = out / f"pilot_{len(g):03d}.json"
        if path.exists():
            raise ValueError("Preserve recorded pilot")
        write(path, summary)
        print(summary, flush=True)
        return
    state = read(out / "run_status.json")
    if len(g) != 24 or state["status"] != "COMPLETE":
        raise ValueError("Complete24-author run required")
    if (out / "summary.json").exists():
        raise ValueError("Preserve completed analysis")
    frame.to_csv(out / "private_predictions.csv", index=False)
    summary["source_cost"] = state
    summary["mean_gradient_interval"] = paired_intervals(
        y, frame.loo_mean_prediction, frame.self_integral_prediction, False
    )
    summary["native_integrated_median_cosine"] = float(
        np.median(
            np.sum(g0 * g, axis=1)
            / np.linalg.norm(g0, axis=1)
            / np.linalg.norm(g, axis=1)
        )
    )
    with np.load(source / "private_patterns.npz", allow_pickle=False) as z:
        patterns = {name: z[name] for name in z.files}
    comparison = []
    for i, p in enumerate(packets):
        old = read(source / "private_packets" / f"author_{i:03d}.json")
        for row in old["effects"]:
            geometry, fold = row["geometry"], p["fold"]
            selected, null = split_reader_component(
                d[i], patterns[f"{fold}_beta"], patterns[f"{fold}_{geometry}"]
            )
            comparison.append(
                {
                    "author_id": p["author_id"],
                    "geometry": geometry,
                    "selected_observed": row["selected_effect"],
                    "null_observed": row["null_effect"],
                    "selected_gradient": float(g[i] @ selected),
                    "null_gradient": float(g[i] @ null),
                }
            )
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(out / "private_s18_contractions.csv", index=False)
    summary["s18_component_contractions"] = {
        geometry: {
            component: quality(
                group[f"{component}_observed"], group[f"{component}_gradient"]
            )
            for component in ("selected", "null")
        }
        for geometry, group in comparison.groupby("geometry")
    }
    pd.DataFrame([{"arm": name, **values} for name, values in metrics.items()]).to_csv(
        out / "metrics.csv", index=False
    )
    write(out / "summary.json", summary)
    print(summary, flush=True)


def main():
    """Expose frozen preparation, resumable collection and label-free summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run", "inspect", "evaluate"])
    parser.add_argument("--source", type=Path, default=BASE / "reader_pattern_s18_r1")
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "functional_gradient_s19_r1"
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-authors", type=int, default=8)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "inspect": analyze, "evaluate": analyze}[
        args.mode
    ](args)


if __name__ == "__main__":
    main()
