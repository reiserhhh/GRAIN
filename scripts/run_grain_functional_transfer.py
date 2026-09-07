#!/usr/bin/env python3
"""S20: freeze a functional reader, then test other source authors."""

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

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.direct_traits import candidates, tokenize_prompt
from grain_research.vector_mechanisms.functional_transfer import shared_gradient_scores
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.analyze_grain_teacher_student_bridge import paired_intervals
from scripts.run_grain_attention_pairing import capture_head, patch_head_score
from scripts.run_grain_conditional_moments import verify as verify_s15
from scripts.run_grain_functional_gradient import quality
from scripts.run_grain_functional_gradient import verify as verify_s19
from scripts.run_grain_multiquery_sensitivity import token_groups
from scripts.run_grain_query_representation import pairing_vector, validate_packet
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write
from scripts.run_grain_source_remainder import CONNECTION


def sources(source, device):
    """Resolve source and discovery panel through immutable inherited paths."""
    _m, s18, s12, pilot = verify_s19(source, device)
    s15 = Path(read(s18 / "manifest.json")["source"])
    _m, s14, inherited_s12, material = verify_s15(s15, device)
    if s12 != inherited_s12 or pilot != material[:24] or len(material) != 213:
        raise ValueError("Same213-author source and24-author panel required")
    return s14, s12, material


def prepare(args):
    """Freeze reference and all cached predictions without any source labels."""
    source, out = args.source.resolve(), args.output_dir
    s14, s12, material = sources(source, args.device)
    if read(source / "summary.json")["status"] != "COMPLETE":
        raise ValueError("Complete S19 gradient experiment required")
    gradients, vectors, hashes = [], [], {}
    for i, author in enumerate(material):
        path = s12 / "private_packets" / f"author_{i:03d}.json"
        validate_packet(path, author)
        for item in (path, path.with_suffix(".npz")):
            hashes[str(item)] = sha(item)
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
            vectors.append(z["kv_pairing"])
        if i < 24:
            packet = source / "private_packets" / path.name
            detail = read(packet)
            if (
                detail["author_id"] != author["author_id"]
                or sha(packet.with_suffix(".npz")) != detail["arrays_sha256"]
            ):
                raise ValueError("Gradient/reference author or array mismatch")
            for item in (packet, packet.with_suffix(".npz")):
                hashes[str(item)] = sha(item)
            with np.load(packet.with_suffix(".npz"), allow_pickle=False) as z:
                gradients.append(z["integrated"])
    reference, views, means = shared_gradient_scores(gradients, vectors)
    frame = pd.DataFrame(
        {
            "author_id": [a["author_id"] for a in material],
            "reference_author": np.arange(213) < 24,
            "mean_response": means,
            **{f"view{v}_response": views[:, v] for v in range(3)},
        }
    )
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    np.savez_compressed(out / "private_reference.npz", reference=reference)
    frame.to_csv(out / "private_scores_before_outcomes.csv", index=False)
    files = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/functional_transfer.py",
        ROOT / "docs/GRAIN_FUNCTIONAL_TRANSFER_S20_PLAN.md",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_S20_FUNCTIONAL_TRANSFER_V1",
            "source": str(source),
            "source_manifest_sha256": sha(source / "manifest.json"),
            "source_summary_sha256": sha(source / "summary.json"),
            "s14_inputs_sha256": sha(s14 / "private_inputs.csv"),
            "source_packet_hashes": hashes,
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "scores_sha256": sha(out / "private_scores_before_outcomes.csv"),
            "reference_sha256": sha(out / "private_reference.npz"),
            "membership_sha256": material_hash(frame.author_id.tolist()),
            "device": args.device,
            "authors": 213,
            "reference_authors": 24,
            "primary_transfer_authors": 189,
            "intervention_indices": list(range(24, 48)),
            "max_forwards": 48,
            "max_seconds": 300,
            "labels_used_in_scores": False,
        },
    )
    print({"status": "SCORES_FROZEN", "n": 213, "new_forwards": 48}, flush=True)


def verify(out, device):
    """Validate frozen reference/scores and complete author provenance."""
    m = read(out / "manifest.json")
    source = Path(m["source"])
    if (
        m["device"] != device
        or sha(source / "manifest.json") != m["source_manifest_sha256"]
        or sha(source / "summary.json") != m["source_summary_sha256"]
        or sha(out / "private_scores_before_outcomes.csv") != m["scores_sha256"]
        or sha(out / "private_reference.npz") != m["reference_sha256"]
        or any(sha(Path(p)) != h for p, h in m["source_packet_hashes"].items())
        or any(sha(ROOT / p) != h for p, h in m["code_hashes"].items())
    ):
        raise ValueError("Frozen S20 inputs changed")
    s14, s12, material = sources(source, device)
    if sha(s14 / "private_inputs.csv") != m["s14_inputs_sha256"]:
        raise ValueError("Source comparison table changed")
    frame = pd.read_csv(
        out / "private_scores_before_outcomes.csv", dtype={"author_id": str}
    )
    if (
        frame.author_id.tolist() != [a["author_id"] for a in material]
        or material_hash(frame.author_id.tolist()) != m["membership_sha256"]
        or not frame.author_id.is_unique
    ):
        raise ValueError("Unique full frozen author order required")
    return m, s14, s12, material, frame


def run(args):
    """Measure fixed new-author effects without fitting or backpropagation."""
    out = args.output_dir
    m, _s14, s12, material, frame = verify(out, args.device)
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
    start = time.monotonic()

    def status(name):
        state = {
            "status": name,
            "calls": previous["calls"] + observer.calls,
            "seconds": previous["seconds"] + time.monotonic() - start,
            "authors": len(list((out / "private_packets").glob("author_*.json"))),
            "labels_read": False,
            "backwards": 0,
        }
        write(out / "run_status.json", state)
        print(state, flush=True)

    try:
        for i in m["intervention_indices"]:
            author = material[i]
            path = out / "private_packets" / f"author_{i:03d}.json"
            if path.exists():
                if read(path)["author_id"] != author["author_id"]:
                    raise ValueError("Resume author mismatch")
                continue
            if (
                previous["seconds"] + time.monotonic() - start >= m["max_seconds"]
                or previous["calls"] + observer.calls + 2 > m["max_forwards"]
            ):
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            payload = author["views"][0]
            ids, user, spans = tokenize_prompt(
                payload["texts"], "neuroticism", observer.tokenizer
            )
            if material_hash(ids) != payload["prompt_sha256"]:
                raise ValueError("Frozen prompt changed")
            native, cache = capture_head(observer, ids, alphabet)
            d, head_error = pairing_vector(
                cache, token_groups(observer, user, spans, ids)
            )
            old = s12 / "private_packets" / f"author_{i:03d}.json"
            score_error = abs(native - read(old)["views"][0]["score"])
            with np.load(old.with_suffix(".npz"), allow_pickle=False) as z:
                feature_error = float(np.max(np.abs(d - z["kv_pairing"][0])))
            if feature_error > 1e-8 or score_error > 2e-5:
                raise ValueError("Native score/vector replay failed")
            removed = patch_head_score(
                observer, ids, alphabet, cache["native_head"].reshape(-1) - d
            )
            write(
                path,
                {
                    "author_id": author["author_id"],
                    "view": 0,
                    "effect": native - removed,
                    "prediction": float(frame.iloc[i].view0_response),
                    "native_score_error": score_error,
                    "feature_error": feature_error,
                    "head_error": head_error,
                },
            )
            status("RUNNING")
        status("COMPLETE")
    except Exception as exc:
        status("FAILED")
        write(out / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def evaluate(args):
    """Connect frozen functional scores to internal effects and source outcomes."""
    out = args.output_dir
    m, s14, _s12, _material, frame = verify(out, args.device)
    state = read(out / "run_status.json")
    if state["status"] != "COMPLETE" or state["authors"] != 24:
        raise ValueError("Complete24-author transport check required")
    if (out / "summary.json").exists():
        raise ValueError("Preserve completed analysis")
    packets = [
        read(out / "private_packets" / f"author_{i:03d}.json")
        for i in m["intervention_indices"]
    ]
    for i, p in zip(m["intervention_indices"], packets, strict=True):
        if p["author_id"] != frame.iloc[i].author_id or not np.isclose(
            p["prediction"], frame.iloc[i].view0_response, atol=1e-15, rtol=1e-12
        ):
            raise ValueError("Transport identity or frozen prediction mismatch")
    effect = np.array([p["effect"] for p in packets])
    prediction = np.array([p["prediction"] for p in packets])
    response = quality(effect, prediction)
    response["paired_vs_exact"] = paired_intervals(effect, prediction, effect, False)
    inputs = pd.read_csv(s14 / "private_inputs.csv", dtype={"author_id": str})
    labels = pd.read_csv(
        CONNECTION / "private_predictions.csv",
        dtype={"author_id": str},
        usecols=["author_id", "query", "y_true"],
    )
    labels = labels[labels["query"] == "neuroticism"][["author_id", "y_true"]]
    frame = frame.merge(
        inputs[["author_id", "teacher", "prediction"]],
        on="author_id",
        validate="one_to_one",
    ).merge(labels, on="author_id", validate="one_to_one")
    if (
        len(frame) != 213
        or not np.isfinite(frame.select_dtypes(include="number")).all().all()
    ):
        raise ValueError("Complete finite source coverage required")
    metrics, pairs = [], []
    for name, group in (
        ("primary189", frame[~frame.reference_author]),
        ("discovery24", frame[frame.reference_author]),
        ("all213", frame),
    ):
        for arm in ("mean_response", "view0_response", "teacher", "prediction"):
            values = scalar_metrics(group.y_true, group[arm])
            metrics.append(
                {
                    "cohort": name,
                    "n": len(group),
                    "arm": arm,
                    "source_r": values["pearson_r"],
                    "source_p": values["pearson_p"],
                    "source_r_lower": values["r_lower"],
                    "source_r_upper": values["r_upper"],
                    "configuration_r": scalar_metrics(group.teacher, group[arm])[
                        "pearson_r"
                    ],
                    "source_covariance": float(
                        np.cov(group.y_true, group[arm], ddof=0)[0, 1]
                    ),
                }
            )
        pairs.append(
            {
                "cohort": name,
                **paired_intervals(
                    group.y_true, group.mean_response, group.prediction, False
                ),
            }
        )
    frame.to_csv(out / "private_source_connections.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / "source_paired_intervals.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "source_cost": state,
        "response_transport": response,
        "metrics": metrics,
        "source_paired_intervals": pairs,
        "scores_sha256": m["scores_sha256"],
        "human_outcome_fit": False,
        "max_native_score_error": max(p["native_score_error"] for p in packets),
        "max_feature_error": max(p["feature_error"] for p in packets),
    }
    write(out / "summary.json", summary)
    print({"response_transport": response}, flush=True)
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


def main():
    """Expose frozen preparation, bounded transfer and one source connection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run", "evaluate"])
    parser.add_argument(
        "--source", type=Path, default=BASE / "functional_gradient_s19_r2"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "functional_transfer_s20_r1"
    )
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "evaluate": evaluate}[args.mode](args)


if __name__ == "__main__":
    main()
