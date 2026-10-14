#!/usr/bin/env python3
"""Cold raw-text GRAIN/query observations using frozen source operators only."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.query_moment_operator import apply_operator, observation_moments
from grain_research.vector_mechanisms.native_material_reader import allocate_slices
from grain_research.vector_mechanisms.prefix import capture_prefix
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_full_state_edit_connection import token_view
from scripts.run_grain_native_material_reader import BUNDLE, FC, REPRESENTATION
from scripts.run_grain_query_moment_operator import (
    BASE,
    INPUT,
    NATIVE,
    load_models,
    read,
    sha,
    write,
)
from scripts.run_grain_real_author_path import MODEL
from scripts.run_grain_within_comment_channels import lexical_slices
from suica_core.v7_observations import ObservationSpec, canonicalize_comments
from suica_core.v7_operator_controls import score_operator_from_comments
from suica_core.v7_psychometric import read_factor_bundle

PUBLIC_TEXTS = [
    "Yesterday I joined a small discussion at the library. I listened to several explanations and wrote down the parts I could not follow. When my turn came I asked one question and described an example from my own experience. Afterwards I went home, reviewed my notes, and decided to try the suggestion before asking anyone whether I had understood it correctly.",
    "Our group had to finish a report before the room closed. The first draft had several gaps, so I made a short list and divided the remaining work with the others. I felt uneasy when nobody answered my message. After a few minutes I checked what was already complete, finished my section, and left a note explaining which decisions still needed discussion.",
    "At breakfast I changed the route I had planned because the usual train was delayed. I looked at two alternatives and chose the one that left enough time to walk. The change bothered me briefly, but I stopped thinking about it once I reached the station. Later I told a friend what happened and asked how they would have handled the same situation.",
]


def prepare(out):
    """Create two raw-input cases before cold execution, without output targets."""
    jobs = read(NATIVE / "private_jobs.json")
    job = next(j for j in jobs if j["cohort"] == "new8" and j["index"] == 0)
    directory = (ROOT / job["input"]).parent.parent
    person = next(j for j in read(directory / "private_design.json") if j["index"] == 0)
    raw = read(BASE / "multiquery_sensitivity_s1_r1/private_material.json")
    material = next(
        r for r in raw if r["author_id"] == person["author_id"] and r["view"] == 0
    )
    dst = out / "cold"
    dst.mkdir(exist_ok=False)
    write(
        dst / "private_raw_inputs.json",
        [
            {"key": "reused_author0", "texts": material["texts"]},
            {"key": "public_new", "texts": PUBLIC_TEXTS},
        ],
    )
    write(
        dst / "input_contract.json",
        {
            "input_sha256": sha(dst / "private_raw_inputs.json"),
            "source_bundle_sha256": sha(BUNDLE),
            "source_representation_sha256": sha(REPRESENTATION),
            "model_config_sha256": sha(MODEL / "config.json"),
            "script_sha256": sha(__file__),
            "public_texts": PUBLIC_TEXTS,
            "cases": 2,
            "max_calls": 3,
            "max_seconds": 300,
        },
    )
    print("Prepared one raw replay and one new public case", flush=True)


def raw_coordinates(texts, tokenizer, representation, bundle):
    """Losslessly slice eligible comments and map fixed GRAIN scores to token spans."""
    _, body_spans, encoded = token_view(tokenizer, texts, "TF_cont")
    rows, spans = [], []
    for ci, text in enumerate(texts):
        parts = lexical_slices(text)
        offset = body_spans[ci][0]
        for si, part in enumerate(parts):
            rows.append({"author": f"c{ci:03d}_s{si}", "body": part})
            spans.append((offset, offset + len(part)))
            offset += len(part)
    if not rows:
        raise ValueError("No eligible 48-token comments; do not fabricate coverage")
    frame = canonicalize_comments(
        pd.DataFrame(rows), order_col=None, condition_col=None, min_tokens=12
    )
    frame["split"] = "confirmation"
    _, _, scored = score_operator_from_comments(
        frame,
        spec=ObservationSpec(**bundle.operator),
        representation=representation,
        bundle=bundle,
    )
    if len(scored) != len(rows) or not scored.score_status.eq("SCORED").all():
        raise ValueError("Every eligible slice must score under the frozen runtime")
    g = scored.set_index("user_id").loc[[r["author"] for r in rows], FC].to_numpy(float)
    positions, token_g, _ = allocate_slices(encoded["offset_mapping"], spans, g)
    return encoded["input_ids"], positions, token_g


def run(out, device):
    """Extract raw states without suffix/teacher calls; deny historical cache reads."""
    import torch

    dst = out / "cold"
    contract = read(dst / "input_contract.json")
    if (
        sha(__file__) != contract["script_sha256"]
        or sha(dst / "private_raw_inputs.json") != contract["input_sha256"]
    ):
        raise ValueError("Cold input/implementation changed")
    if (dst / "summary.json").exists():
        raise FileExistsError("Cold result is immutable")
    inputs = read(dst / "private_raw_inputs.json")
    models = load_models(out)
    representation, bundle = joblib.load(REPRESENTATION), read_factor_bundle(BUNDLE)
    denied = []
    allowed = [
        MODEL.resolve(),
        dst.resolve(),
        (out / "models").resolve(),
        BUNDLE.parent.resolve(),
        REPRESENTATION.parent.resolve(),
    ]

    def guard(event, args):
        if event != "open" or isinstance(args[0], int):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to((ROOT / "results").resolve()) and not any(
            path.is_relative_to(p) for p in allowed
        ):
            denied.append(str(path))
            raise PermissionError("Cold scorer cannot read historical states/targets")

    sys.addaudithook(guard)
    for path in (INPUT / "private_jobs.json", NATIVE / "private_jobs.json"):
        try:
            path.read_bytes()
        except PermissionError:
            pass
        else:
            raise AssertionError("Historical cache probe was not denied")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    torch.set_num_threads(2)
    start = time.monotonic()
    observer = DecoderObserver.load(
        str(MODEL), device=device, dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    load_seconds = time.monotonic() - start
    start = time.monotonic()
    rows, traces = [], []
    for item in inputs:
        ids, positions, g = raw_coordinates(
            item["texts"], observer.tokenizer, representation, bundle
        )
        prefix, trace = capture_prefix(observer, ids)
        h = prefix["boundary"][positions].astype(float)
        np.savez_compressed(dst / (item["key"] + ".npz"), h=h, g=g, positions=positions)
        for query, model in models.items():
            values, _ = apply_operator(g, h, model)
            moment = observation_moments(g, h, model)
            rows.append(
                {
                    "case": item["key"],
                    "query": query,
                    "scores": values.tolist(),
                    "within_energy": float(np.trace(moment["within_covariance"])),
                    "tokens": len(positions),
                }
            )
        traces.append({"case": item["key"], **trace})
        if time.monotonic() - start > contract["max_seconds"]:
            raise TimeoutError("Bounded cold execution exceeded budget")
    if observer.calls > contract["max_calls"] or len(denied) != 2:
        raise AssertionError("Unexpected prefix/cache access count")
    write(
        dst / "summary.json",
        {
            "status": "complete",
            "load_seconds": load_seconds,
            "execution_seconds": time.monotonic() - start,
            "prefix_calls": observer.calls,
            "denied_cache_probes": len(denied),
            "traces": traces,
            "outputs": rows,
            "dependency": "D1: frozen Qwen blocks0..16 plus next QKV; no suffix/logit/backward",
            "files": {
                p.name: sha(p) for p in dst.glob("*.npz") if not p.name.startswith("._")
            },
        },
    )
    print(
        {
            "status": "complete",
            "prefix_calls": observer.calls,
            "seconds": time.monotonic() - start,
        },
        flush=True,
    )


def main():
    """Prepare raw fixtures or score them in an isolated process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.output)
    else:
        run(args.output, args.device)


if __name__ == "__main__":
    main()
