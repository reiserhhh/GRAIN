#!/usr/bin/env python3
"""S13: verify source prefix-only vector/reader replay and paired runtime."""

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
from grain_research.vector_mechanisms.prefix import (
    affine_readout,
    capture_prefix,
    prefix_pairing,
)
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_multiquery_sensitivity import token_groups
from scripts.run_grain_query_representation import (
    REPRESENTATIONS,
    capture_representations,
    state_summaries,
    validate_packet,
    verify,
)
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write

READERS = (("kv_pairing", "pca64"), ("query_state", "pls2"))


def synchronize(device):
    """Finish scheduled GPU work for a local paired elapsed-time measurement."""
    if device == "mps":
        torch.mps.synchronize()


def timed(function, device):
    """Include all feature extraction in paired runtime while excluding model load."""
    synchronize(device)
    start = time.monotonic()
    value = function()
    synchronize(device)
    return value, time.monotonic() - start


def main():
    """Execute the fixed small replay panel; never read source questionnaire columns."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection", type=Path, default=BASE / "query_representation_s12_r1"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Keep private outputs ignored")
    manifest, material = verify(args.collection, args.device)
    analysis = args.collection / "analysis"
    oof_path = analysis / "private_oof_before_outcomes.csv"
    if sha(oof_path) != read(analysis / "summary.json")["oof_sha256"]:
        raise ValueError("S12 reader predictions changed")
    oof = pd.read_csv(oof_path, dtype={"author_id": str})
    oof = oof[oof.target == "configuration_response"]
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    models = {
        str(p.relative_to(analysis)): sha(p)
        for rep, method in READERS
        for p in (analysis / "private_models").glob(
            f"{rep}_{method}_configuration_response_fold*.npz"
        )
    }
    if len(models) != 10:
        raise ValueError("Missing fixed readers")
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_S13_PREFIX_REPLAY_V1",
            "collection_manifest_sha256": sha(args.collection / "manifest.json"),
            "oof_sha256": sha(oof_path),
            "reader_hashes": models,
            "script_sha256": sha(__file__),
            "prefix_code_sha256": sha(
                ROOT / "grain_research/vector_mechanisms/prefix.py"
            ),
            "authors": 24,
            "views": 72,
            "readers": READERS,
            "max_prefix_calls": 73,
            "max_full_calls": 13,
            "max_seconds": 600,
            "source_Y_read": False,
            "model": MODEL.name,
            "numerics": {"device": args.device, "dtype": "float32", "padding": None},
        },
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
    rows, timings, predictions = [], [], []
    start = time.monotonic()
    for i, author in enumerate(material[:24]):
        cached_path = args.collection / "private_packets" / f"author_{i:03d}.json"
        validate_packet(cached_path, author)
        with np.load(cached_path.with_suffix(".npz"), allow_pickle=False) as z:
            saved = {name: z[name] for name in REPRESENTATIONS}
        vectors = {name: [] for name in REPRESENTATIONS}
        for view, payload in enumerate(author["views"]):
            ids, user, spans = tokenize_prompt(
                payload["texts"], "neuroticism", observer.tokenizer
            )
            if material_hash(ids) != payload["prompt_sha256"]:
                raise ValueError("Source prompt changed")
            groups = token_groups(observer, user, spans, ids)

            def prefix():
                cache, trace = capture_prefix(observer, ids)
                body, query = state_summaries(cache["boundary"], groups)
                return {
                    "body_mean": body,
                    "query_state": query,
                    "kv_pairing": prefix_pairing(cache, groups),
                }, trace

            def full():
                return capture_representations(observer, ids, alphabet, groups)

            if i == 0 and view == 0:
                full()
                prefix()
            if i < 12 and view == 0:
                order = ("full", "prefix") if i % 2 == 0 else ("prefix", "full")
                measured = {}
                for name in order:
                    value, elapsed = timed(
                        full if name == "full" else prefix, args.device
                    )
                    measured[name] = elapsed
                    if name == "prefix":
                        result, trace = value
                timings.append(
                    {
                        "unit": i,
                        "full_seconds": measured["full"],
                        "prefix_seconds": measured["prefix"],
                        "prefix_over_full": measured["prefix"] / measured["full"],
                    }
                )
            else:
                (result, trace), _ = timed(prefix, args.device)
            errors = {
                name: float(np.max(np.abs(result[name] - saved[name][view])))
                for name in REPRESENTATIONS
            }
            if max(errors.values()) > 1e-5 or trace["forbidden_calls"]:
                raise ValueError("Prefix representation replay failed")
            for name in REPRESENTATIONS:
                vectors[name].append(result[name])
            rows.append(
                {
                    "unit": i,
                    "view": view,
                    **{k + "_max_error": v for k, v in errors.items()},
                    "completed_blocks": len(trace["completed_blocks"]),
                    "forbidden_calls": len(trace["forbidden_calls"]),
                }
            )
        for rep, method in READERS:
            ref = oof[
                (oof.author_id == author["author_id"])
                & (oof.representation == rep)
                & (oof.method == method)
            ]
            if len(ref) != 1:
                raise ValueError("Reader source join mismatch")
            ref = ref.iloc[0]
            path = (
                analysis
                / "private_models"
                / f"{rep}_{method}_configuration_response_fold{int(ref.fold)}.npz"
            )
            value = affine_readout(vectors[rep], path)
            error = abs(value - ref.prediction)
            if error > 1e-6:
                raise ValueError("Prefix affine reader replay failed")
            predictions.append(
                {
                    "author_id": author["author_id"],
                    "representation": rep,
                    "method": method,
                    "fold": int(ref.fold),
                    "prediction": value,
                    "replay_error": error,
                }
            )
        write(
            out / "private_packets" / f"author_{i:03d}.json",
            {"author_id": author["author_id"], "complete_views": 3},
        )
        if (i + 1) % 8 == 0:
            print(
                {
                    "status": "RUNNING",
                    "authors": i + 1,
                    "calls": observer.calls,
                    "seconds": time.monotonic() - start,
                },
                flush=True,
            )
        if time.monotonic() - start > 600:
            raise TimeoutError("S13 bounded runtime exceeded")
    if observer.calls != 86:
        raise ValueError("Unexpected prefix/full call count")
    pd.DataFrame(rows).to_csv(out / "replay_checks.csv", index=False)
    pd.DataFrame(timings).to_csv(out / "paired_runtime.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "private_predictions.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "manifest_sha256": sha(out / "manifest.json"),
        "calls": observer.calls,
        "prefix_calls": 73,
        "full_calls": 13,
        "backwards": 0,
        "seconds": time.monotonic() - start,
        "views": len(rows),
        "authors": 24,
        "source_Y_read": False,
        "max_vector_error": max(
            r[name + "_max_error"] for r in rows for name in REPRESENTATIONS
        ),
        "max_reader_error": max(r["replay_error"] for r in predictions),
        "forbidden_calls": sum(r["forbidden_calls"] for r in rows),
        "median_prefix_over_full": float(
            np.median([r["prefix_over_full"] for r in timings])
        ),
        "total_prefix_over_full": sum(r["prefix_seconds"] for r in timings)
        / sum(r["full_seconds"] for r in timings),
        "scope": "numerical replay and paired local feature-extraction cost; no new label evidence",
    }
    write(out / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
