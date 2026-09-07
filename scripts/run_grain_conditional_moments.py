#!/usr/bin/env python3
"""S15: bounded frozen common-query readout with conditional joint-moment arms."""

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

from grain_research.vector_mechanisms.conditional_moments import (
    conditional_moment_readouts,
)
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.direct_traits import tokenize_prompt
from grain_research.vector_mechanisms.prefix import capture_prefix, rotate_query
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.analyze_grain_query_representation import target_quality
from scripts.analyze_grain_teacher_student_bridge import paired_intervals
from scripts.run_grain_multiquery_sensitivity import token_groups
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write
from scripts.run_grain_shared_query import verify_inputs
from scripts.run_grain_source_remainder import CONNECTION

ARMS = ("exact", "uniform_moment", "exact_mass_moment", "gaussian_mass_moment")


def prepare(args):
    """Freeze S14 sources and all four untrained approximation operators."""
    source, out = args.source.resolve(), args.output_dir
    verify_inputs(source, args.device)
    if read(source / "summary.json")["status"] != "COMPLETE":
        raise ValueError("Complete S14 source required")
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    files = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/conditional_moments.py",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_S15_CONDITIONAL_MOMENTS_V1",
            "source": str(source),
            "source_summary_sha256": sha(source / "summary.json"),
            "reference_sha256": sha(source / "private_references.npz"),
            "source_packet_hashes": {
                p.name: sha(p)
                for p in (source / "private_packets").glob("author_*.json")
            },
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "arms": ARMS,
            "device": args.device,
            "max_seconds": 1500,
            "max_prefix_calls": 639,
            "pilot_authors": 24,
            "outcome_fit": False,
        },
    )
    print({"status": "PREPARED", "pilot_authors": 24, "full_authors": 213}, flush=True)


def verify(out, device):
    """Validate immutable math, references and already-frozen S14 source provenance."""
    m = read(out / "manifest.json")
    source = Path(m["source"])
    if (
        m["device"] != device
        or sha(source / "summary.json") != m["source_summary_sha256"]
        or sha(source / "private_references.npz") != m["reference_sha256"]
        or any(sha(ROOT / p) != h for p, h in m["code_hashes"].items())
        or any(
            sha(source / "private_packets" / p) != h
            for p, h in m["source_packet_hashes"].items()
        )
    ):
        raise ValueError("Frozen S15 inputs changed")
    s14, material = verify_inputs(source, device)
    return m, source, Path(s14["collection"]), material


def head_readouts(cache, groups, query):
    """Join per-head fixed moment formulas in the source reader's coordinate order."""
    k, v = cache["k"], cache["v"]
    ratio = len(query) // k.shape[1]
    parts = {arm: [] for arm in ARMS}
    for head, q in enumerate(query):
        scores = conditional_moment_readouts(
            k[:, head // ratio], v[:, head // ratio], q, groups
        )
        for arm in ARMS:
            parts[arm].append(scores[arm])
    return {arm: np.concatenate(part) for arm, part in parts.items()}


def run(args):
    """Execute one resumable bounded cohort without inspecting source questionnaires."""
    out = args.output_dir
    manifest, source, s12, material = verify(out, args.device)
    inputs = pd.read_csv(source / "private_inputs.csv", dtype={"author_id": str})
    if list(inputs.author_id) != [a["author_id"] for a in material]:
        raise ValueError("Source author order changed")
    previous = (
        read(out / "run_status.json")
        if (out / "run_status.json").exists()
        else {"calls": 0, "seconds": 0}
    )
    with np.load(source / "private_references.npz", allow_pickle=False) as z:
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
                    raise ValueError("Resume author mismatch")
                continue
            if (
                i >= args.max_authors
                or previous["seconds"] + time.monotonic() - start
                >= manifest["max_seconds"]
            ):
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            fold = int(inputs.iloc[i].fold)
            with np.load(
                s12
                / "analysis/private_models"
                / f"kv_pairing_pca64_configuration_response_fold{fold}.npz",
                allow_pickle=False,
            ) as z:
                beta, intercept = z["coef"], float(z["intercept"])
            old = read(source / "private_packets" / f"author_{i:03d}.json")
            rows = []
            for view, payload in enumerate(author["views"]):
                ids, user, spans = tokenize_prompt(
                    payload["texts"], "neuroticism", observer.tokenizer
                )
                if material_hash(ids) != payload["prompt_sha256"]:
                    raise ValueError("Native source prompt changed")
                groups = token_groups(observer, user, spans, ids)
                cache, trace = capture_prefix(observer, ids)
                query = rotate_query(
                    references[str(fold)],
                    cache["last_cos"],
                    cache["last_sin"],
                    cache["scaling"],
                )
                vectors = head_readouts(cache, groups, query)
                scores = {
                    arm: float(vector @ beta + intercept)
                    for arm, vector in vectors.items()
                }
                error = abs(scores["exact"] - old["views"][view]["scores"]["0.0"])
                if (
                    error > 1e-8
                    or trace["forbidden_calls"]
                    or not np.isfinite(list(scores.values())).all()
                ):
                    raise ValueError(
                        "Readout replay/boundary/finite-value check failed"
                    )
                rows.append({"view": view, "scores": scores, "replay_error": error})
            write(path, {"author_id": author["author_id"], "fold": fold, "views": rows})
            if (i + 1) % 16 == 0:
                status("RUNNING")
            if previous["calls"] + observer.calls > manifest["max_prefix_calls"]:
                raise RuntimeError("Prefix call budget exceeded")
        status("COMPLETE")
    except BaseException as exc:
        status("FAILED")
        write(out / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def collected(out, material):
    """Aggregate complete three-view units, retaining source-author ordering."""
    rows, maximum = [], 0.0
    for i, author in enumerate(material):
        path = out / "private_packets" / f"author_{i:03d}.json"
        if not path.exists():
            break
        packet = read(path)
        if packet["author_id"] != author["author_id"] or len(packet["views"]) != 3:
            raise ValueError("Incomplete or misaligned author")
        row = {"author_id": author["author_id"]}
        row.update(
            {
                arm: float(np.mean([v["scores"][arm] for v in packet["views"]]))
                for arm in ARMS
            }
        )
        rows.append(row)
        maximum = max(maximum, max(v["replay_error"] for v in packet["views"]))
    return pd.DataFrame(rows), maximum


def fidelity(frame):
    """Measure absolute scale/bias and ordering without recalibrating an arm."""
    variance = float(np.var(frame.exact))
    rows = []
    for arm in ARMS:
        mse = float(np.mean((frame[arm] - frame.exact) ** 2))
        rows.append(
            {
                "arm": arm,
                "exact_reader_r": scalar_metrics(frame.exact, frame[arm])["pearson_r"],
                "exact_reader_mse": mse,
                "mse_over_exact_variance": mse / variance if variance else None,
                "mean_error": float(np.mean(frame[arm] - frame.exact)),
            }
        )
    return rows


def analyze(args):
    """Inspect a label-free checkpoint or freeze a full cohort before outcome joining."""
    out = args.output_dir
    _m, source, _s12, material = verify(out, args.device)
    frame, maximum = collected(out, material)
    if len(frame) < 3:
        raise ValueError("At least three complete authors required")
    rows = fidelity(frame)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    if args.mode == "inspect":
        path = out / f"pilot_{len(frame):03d}.json"
        if path.exists():
            raise ValueError("Preserve existing checkpoint")
        write(
            path,
            {
                "n": len(frame),
                "labels_read": False,
                "replay_error": maximum,
                "fidelity": rows,
            },
        )
        return
    state = read(out / "run_status.json")
    if len(frame) != 213 or state["status"] != "COMPLETE" or state["calls"] != 639:
        raise ValueError("Only a complete213-author cohort may connect to source Y")
    frozen = out / "private_scores_before_outcomes.csv"
    if frozen.exists() or (out / "summary.json").exists():
        raise ValueError("Preserve frozen analysis")
    frame.to_csv(frozen, index=False)
    write(out / "scores_frozen.json", {"sha256": sha(frozen), "labels_used": False})
    inputs = pd.read_csv(source / "private_inputs.csv", dtype={"author_id": str})
    labels = pd.read_csv(
        CONNECTION / "private_predictions.csv",
        dtype={"author_id": str},
        usecols=["author_id", "query", "y_true"],
    )
    labels = labels[labels["query"] == "neuroticism"][["author_id", "y_true"]]
    frame = frame.merge(
        inputs[["author_id", "teacher", "null_prediction"]],
        on="author_id",
        validate="one_to_one",
    ).merge(labels, on="author_id", validate="one_to_one")
    if len(frame) != 213 or not np.isfinite(frame.y_true).all():
        raise ValueError("Complete source outcome coverage required")
    metrics, pairs = [], []
    for row in rows:
        arm = row["arm"]
        external = scalar_metrics(frame.y_true, frame[arm])
        metrics.append(
            {
                **row,
                **target_quality(frame.teacher, frame[arm], frame.null_prediction),
                "source_r": external["pearson_r"],
                "source_p": external["pearson_p"],
            }
        )
        pairs.append(
            {
                "arm": arm,
                **paired_intervals(frame.y_true, frame[arm], frame.exact, False),
            }
        )
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / "source_paired_intervals.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "n": len(frame),
            "prefix_calls": state["calls"],
            "seconds": state["seconds"],
            "replay_error": maximum,
            "scores_sha256": sha(frozen),
            "metrics": metrics,
            "source_paired_intervals": pairs,
            "labels_fit": False,
        },
    )
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


def main():
    """Prepare, run a pilot/resume, inspect internally, or evaluate a complete cohort."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "inspect", "evaluate"))
    parser.add_argument("--source", type=Path, default=BASE / "shared_query_s14_r2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--max-authors", type=int, default=24)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(BASE) or not 1 <= args.max_authors <= 213:
        raise ValueError("Ignored output directory and bounded author count required")
    {"prepare": prepare, "run": run, "inspect": analyze, "evaluate": analyze}[
        args.mode
    ](args)


if __name__ == "__main__":
    main()
