#!/usr/bin/env python3
"""S16: fixed24-source-author gain/allocation pilot without questionnaire access."""

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
from grain_research.vector_mechanisms.direct_traits import tokenize_prompt
from grain_research.vector_mechanisms.prefix import capture_prefix, rotate_query
from grain_research.vector_mechanisms.readout_gain import ARMS, multihead_gain
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.run_grain_conditional_moments import verify as verify_s15
from scripts.run_grain_multiquery_sensitivity import token_groups
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write


def prepare(args):
    """Freeze pilot membership and source hashes before new measurements."""
    source, out = args.source.resolve(), args.output_dir
    _m, _s14, _s12, material = verify_s15(source, args.device)
    if read(source / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete S15 source required")
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    files = [Path(__file__), ROOT / "grain_research/vector_mechanisms/readout_gain.py"]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_S16_READOUT_GAIN_V1",
            "source": str(source),
            "source_manifest_sha256": sha(source / "manifest.json"),
            "source_packets": {
                f"author_{i:03d}.json": sha(
                    source / "private_packets" / f"author_{i:03d}.json"
                )
                for i in range(24)
            },
            "author_membership_hash": material_hash(
                [a["author_id"] for a in material[:24]]
            ),
            "device": args.device,
            "authors": 24,
            "views": 72,
            "max_seconds": 600,
            "max_prefix_calls": 72,
            "arms": ARMS,
            "questionnaire_access": False,
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
        },
    )
    print({"status": "PREPARED", "authors": 24, "prefix_calls": 72}, flush=True)


def verify(out, device):
    """Revalidate S15/S14/S12 provenance and the fixed new mathematical operators."""
    m = read(out / "manifest.json")
    source = Path(m["source"])
    if (
        m["device"] != device
        or sha(source / "manifest.json") != m["source_manifest_sha256"]
        or any(sha(ROOT / p) != h for p, h in m["code_hashes"].items())
        or any(
            sha(source / "private_packets" / p) != h
            for p, h in m["source_packets"].items()
        )
    ):
        raise ValueError("Frozen S16 inputs changed")
    _s15, s14, s12, material = verify_s15(source, device)
    material = material[:24]
    if material_hash([a["author_id"] for a in material]) != m["author_membership_hash"]:
        raise ValueError("Pilot author membership changed")
    return m, source, s14, s12, material


def run(args):
    """Collect only72 prefixes, preserving replay and per-head signed contributions."""
    out = args.output_dir
    manifest, source, s14, s12, material = verify(out, args.device)
    previous = (
        read(out / "run_status.json")
        if (out / "run_status.json").exists()
        else {"calls": 0, "seconds": 0}
    )
    with np.load(s14 / "private_references.npz", allow_pickle=False) as z:
        references = {name: z[name].astype(np.float32) for name in z.files}
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device=args.device, dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    start = time.monotonic()

    def status(name):
        state = {
            "status": name,
            "calls": previous["calls"] + observer.calls,
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
                if read(path)["author_id"] != author["author_id"]:
                    raise ValueError("Resumed author mismatch")
                continue
            if (
                previous["seconds"] + time.monotonic() - start
                >= manifest["max_seconds"]
            ):
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            old = read(source / "private_packets" / f"author_{i:03d}.json")
            fold = old["fold"]
            with np.load(
                s12
                / "analysis/private_models"
                / f"kv_pairing_pca64_configuration_response_fold{fold}.npz",
                allow_pickle=False,
            ) as z:
                beta, intercept = z["coef"], float(z["intercept"])
            rows = []
            for view, payload in enumerate(author["views"]):
                ids, user, spans = tokenize_prompt(
                    payload["texts"], "neuroticism", observer.tokenizer
                )
                if material_hash(ids) != payload["prompt_sha256"]:
                    raise ValueError("Source prompt changed")
                groups = token_groups(observer, user, spans, ids)
                cache, trace = capture_prefix(observer, ids)
                query = rotate_query(
                    references[str(fold)],
                    cache["last_cos"],
                    cache["last_sin"],
                    cache["scaling"],
                )
                vectors, diagnostics = multihead_gain(cache, groups, query, beta)
                scores = {
                    arm: float(vector @ beta + intercept)
                    for arm, vector in vectors.items()
                }
                pairs = (
                    ("exact", "exact"),
                    ("exact_mass_cov", "exact_mass_moment"),
                    ("uniform_cov", "uniform_moment"),
                )
                error = max(
                    abs(scores[a] - old["views"][view]["scores"][b]) for a, b in pairs
                )
                if (
                    error > 1e-8
                    or trace["forbidden_calls"]
                    or not np.isfinite(list(scores.values())).all()
                ):
                    raise ValueError(
                        "Replay, finite-value or prefix-boundary check failed"
                    )
                rows.append(
                    {
                        "view": view,
                        "scores": scores,
                        "head_diagnostics": diagnostics,
                        "replay_error": error,
                    }
                )
            write(path, {"author_id": author["author_id"], "fold": fold, "views": rows})
            if (i + 1) % 8 == 0:
                status("RUNNING")
            if previous["calls"] + observer.calls > 72:
                raise RuntimeError("Prefix call cap exceeded")
        status("COMPLETE")
    except BaseException as exc:
        status("FAILED")
        write(out / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def evaluate(args):
    """Write author-averaged fidelity and head summaries, never join human outcomes."""
    out = args.output_dir
    _m, _source, _s14, _s12, material = verify(out, args.device)
    state = read(out / "run_status.json")
    if state["status"] != "COMPLETE" or state["calls"] != 72:
        raise ValueError("Complete fixed pilot required")
    if (out / "summary.json").exists():
        raise ValueError("Preserve completed analysis")
    rows, heads, maximum = [], [], 0.0
    for i, author in enumerate(material):
        packet = read(out / "private_packets" / f"author_{i:03d}.json")
        if packet["author_id"] != author["author_id"] or len(packet["views"]) != 3:
            raise ValueError("Complete source author units required")
        rows.append(
            {
                "author_id": author["author_id"],
                **{
                    arm: float(np.mean([v["scores"][arm] for v in packet["views"]]))
                    for arm in ARMS
                },
            }
        )
        for view in packet["views"]:
            maximum = max(maximum, view["replay_error"])
            for head in view["head_diagnostics"]:
                heads.append(
                    {
                        "author_id": author["author_id"],
                        "view": view["view"],
                        **{
                            key: value
                            for key, value in head.items()
                            if key not in ("group_mass", "contributions")
                        },
                        **{
                            arm + "_contribution": value
                            for arm, value in head["contributions"].items()
                        },
                    }
                )
    frame, hf = pd.DataFrame(rows), pd.DataFrame(heads)
    frame.to_csv(out / "private_scores.csv", index=False)
    hf.to_csv(out / "private_head_diagnostics.csv", index=False)
    variance, metrics = float(np.var(frame.exact)), []
    for arm in ARMS:
        error = frame[arm] - frame.exact
        quality = scalar_metrics(frame.exact, frame[arm])
        metrics.append(
            {
                "arm": arm,
                "n": len(frame),
                "exact_r": quality["pearson_r"],
                "mse": float(np.mean(error**2)),
                "mse_over_exact_variance": float(np.mean(error**2)) / variance,
                "mean_error": float(error.mean()),
                "centered_error_variance": float(np.var(error)),
            }
        )
    head_rows = []
    author_head = (
        hf.drop(columns="view")
        .groupby(["author_id", "head"])
        .mean(numeric_only=True)
        .reset_index()
    )
    for head, h in author_head.groupby("head"):
        head_rows.append(
            {
                "head": int(head),
                "mean_body_mass": float(h.body_mass.mean()),
                "median_body_mass": float(h.body_mass.median()),
                "mean_first_token_mass": float(h.first_token_mass.mean()),
                "mean_comment_entropy": float(h.normalized_comment_entropy.mean()),
                "exact_contribution_sd": float(h.exact_contribution.std(ddof=0)),
                "gain_allocation_change_sd": float(
                    (h.exact_contribution - h.gain_only_exact_contribution).std(ddof=0)
                ),
            }
        )
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(head_rows).to_csv(out / "head_summary.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "authors": 24,
            "prefix_calls": 72,
            "seconds": state["seconds"],
            "replay_error": maximum,
            "labels_read": False,
            "fit": False,
            "scores_sha256": sha(out / "private_scores.csv"),
            "metrics": metrics,
            "head_summary": head_rows,
        },
    )
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    print(pd.DataFrame(head_rows).to_string(index=False), flush=True)


def main():
    """Prepare, run, or evaluate the immutable24-author mechanism pilot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "evaluate"))
    parser.add_argument(
        "--source", type=Path, default=BASE / "conditional_moments_s15_r1"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(BASE):
        raise ValueError("Private outputs must remain ignored")
    {"prepare": prepare, "run": run, "evaluate": evaluate}[args.mode](args)


if __name__ == "__main__":
    main()
