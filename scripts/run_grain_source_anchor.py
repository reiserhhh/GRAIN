#!/usr/bin/env python3
"""Locate source remainder in conditional anchor values versus readout transport."""

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

from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
from grain_research.vector_mechanisms.full_state_edit import replay_suffix
from grain_research.vector_mechanisms.level_response import correlation
from grain_research.vector_mechanisms.prefix import capture_prefix
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_level_response import question_mask
from scripts.run_grain_multiquery_sensitivity import native_alphabet
from scripts.run_grain_prefix_source_connection import LEVEL, inputs
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write
from scripts.run_grain_source_remainder import covariance

PREFIX = BASE / "prefix_source_connection_v1"
PLAN = ROOT / "docs/GRAIN_SOURCE_ANCHOR_DIAGNOSIS_PLAN_V1.md"


def prepare(out):
    """Choose fixed source-index prefixes of each query, without reading individual Y."""
    _, _, rows = inputs(PREFIX)
    chosen = [
        a
        for q in ("TF_cont", "neuroticism")
        for a in sorted((r for r in rows if r["query"] == q), key=lambda r: r["index"])[
            :24
        ]
    ]
    if len(chosen) != 48:
        raise ValueError("48 fixed authors required")
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    write(out / "private_design.json", chosen)
    files = [
        Path(__file__),
        PLAN,
        ROOT / "grain_research/vector_mechanisms/prefix.py",
        ROOT / "grain_research/vector_mechanisms/full_state_edit.py",
        ROOT / "grain_research/vector_mechanisms/context_carrier.py",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_SOURCE_ANCHOR_V1",
            "authors": 48,
            "max_calls": 288,
            "max_seconds": 600,
            "input_sha256": sha(out / "private_design.json"),
            "reference_sha256": sha(LEVEL / "reference.npz"),
            "source_manifest_sha256": sha(PREFIX / "manifest.json"),
            "model_sha256": read(PREFIX / "manifest.json")["model_sha256"],
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
        },
    )
    write(
        out / "run_status.json",
        {"status": "PREPARED", "calls": 0, "seconds": 0, "authors": 0},
    )
    print({"status": "PREPARED", "authors": 48, "calls": 288})


def run(out):
    """Measure one actual anchor per native view and preserve full boundary states."""
    m, prior = read(out / "manifest.json"), read(out / "run_status.json")
    if prior["status"] == "COMPLETE":
        print("Already complete")
        return
    for p, digest in (
        (out / "private_design.json", m["input_sha256"]),
        (LEVEL / "reference.npz", m["reference_sha256"]),
        (MODEL / "model.safetensors", m["model_sha256"]),
        *[(ROOT / p, h) for p, h in m["code_hashes"].items()],
    ):
        if sha(p) != digest:
            raise ValueError("Frozen source/implementation changed")
    with np.load(LEVEL / "reference.npz", allow_pickle=False) as a:
        reference = dict(a)
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    now, old = observer.manifest(), read(PREFIX / "observer.json")
    for k in (
        "torch",
        "transformers",
        "dtype",
        "device",
        "attention",
        "template_sha256",
        "pad_to_length",
    ):
        if now[k] != old[k]:
            raise ValueError("Observer changed")
    write(out / "observer.json", now)
    start, completed = time.monotonic(), 0

    def status(state):
        """Retain actual bounded work, with no source-outcome access during collection."""
        value = {
            "status": state,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "authors": completed,
            "planned_authors": 48,
            "gradient_calls": 0,
            "labels_read": False,
        }
        write(out / "run_status.json", value)
        print(value, flush=True)

    try:
        for a in read(out / "private_design.json"):
            dst = out / "private_packets" / f"author_{a['index']:03d}.json"
            if dst.exists():
                completed += 1
                continue
            if (
                prior["calls"] + observer.calls + 6 > m["max_calls"]
                or prior["seconds"] + time.monotonic() - start > m["max_seconds"]
            ):
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            q = a["query"]
            with np.load(
                PREFIX / "private_packets" / dst.with_suffix(".npz").name,
                allow_pickle=False,
            ) as z:
                ps, hs = z["P"], z["Q"]
            states, views = {}, []
            for i, v in enumerate(a["views"]):
                _, _, encoded, mask = question_mask(observer.tokenizer, v["texts"], q)
                cache, trace = capture_prefix(observer, encoded["input_ids"])
                h = cache["boundary"]
                error = max(
                    float(np.max(np.abs(h[mask] - ps[i]))),
                    float(np.max(np.abs(h[-1] - hs[i]))),
                )
                if error > 2e-5:
                    raise ValueError("Source prefix replay mismatch")
                state = torch.tensor(
                    h[None], device=next(observer.model.parameters()).device
                )
                state[:, mask] = torch.as_tensor(
                    reference[f"{q}__Z"], device=state.device, dtype=state.dtype
                )
                state[:, -1] = torch.as_tensor(
                    reference[f"{q}__h"], device=state.device, dtype=state.dtype
                )
                b = replay_suffix(
                    observer,
                    state,
                    cached_suffix_kwargs(observer.model, state),
                    native_alphabet(q, observer.tokenizer),
                )["conditional_score"]
                states[f"view_{i}"] = h
                views.append(
                    {
                        "view": i,
                        "anchor": b,
                        "prefix_replay_error": error,
                        "trace": trace,
                    }
                )
            np.savez_compressed(dst.with_suffix(".npz"), **states)
            write(
                dst,
                {
                    "author_id": a["author_id"],
                    "query": q,
                    "index": a["index"],
                    "views": views,
                    "arrays_sha256": sha(dst.with_suffix(".npz")),
                },
            )
            completed += 1
            if completed % 8 == 0:
                status("RUNNING")
        status("COMPLETE")
    except BaseException as exc:
        status("FAILED_RETAINED")
        write(out / "error_log.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


def evaluate(out):
    """Freeze anchor values first, then account for already-observed author associations."""
    if read(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete anchor packet required")
    anchors = []
    for a in read(out / "private_design.json"):
        path = out / "private_packets" / f"author_{a['index']:03d}.json"
        r = read(path)
        if (
            r["author_id"] != a["author_id"]
            or sha(path.with_suffix(".npz")) != r["arrays_sha256"]
        ):
            raise ValueError("Anchor archive changed")
        anchors.append(
            {
                "author_id": a["author_id"],
                "query": a["query"],
                "anchor": float(np.mean([v["anchor"] for v in r["views"]])),
            }
        )
    pd.DataFrame(anchors).to_csv(
        out / "private_anchors_before_outcomes.csv", index=False
    )
    write(
        out / "anchors_frozen.json",
        {
            "sha256": sha(out / "private_anchors_before_outcomes.csv"),
            "outcomes_used": False,
        },
    )
    source = pd.read_csv(
        PREFIX / "private_s9_remainder_connection.csv", dtype={"author_id": str}
    )
    frame = pd.DataFrame(anchors).merge(
        source, on=["author_id", "query"], validate="one_to_one"
    )
    with np.load(LEVEL / "reference.npz", allow_pickle=False) as a:
        ref = dict(a)
    frame["base_ref"] = [float(ref[f"{q}__b"]) for q in frame["query"]]
    frame["baseline_shift"] = frame.anchor - frame.base_ref
    frame["transport_remainder"] = (
        frame.native - frame.anchor - (frame.PQ - frame.base_ref)
    )
    if len(frame) != 48 or not np.allclose(
        frame.remainder,
        frame.baseline_shift + frame.transport_remainder,
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("Exact anchor source ledger mismatch")
    metrics = []
    for q, g in frame.groupby("query"):
        for col in (
            "native",
            "PQ",
            "remainder",
            "baseline_shift",
            "transport_remainder",
            "configuration_response",
        ):
            metrics.append(
                {
                    "query": q,
                    "component": col,
                    "authors": len(g),
                    "mean": float(g[col].mean()),
                    "sd": float(g[col].std(ddof=0)),
                    "r_y": correlation(g.y_true.to_numpy(), g[col].to_numpy()),
                    "cov_y": covariance(g.y_true.to_numpy(), g[col].to_numpy()),
                    "r_remainder": correlation(
                        g.remainder.to_numpy(), g[col].to_numpy()
                    ),
                    "r_configuration": correlation(
                        g.configuration_response.to_numpy(), g[col].to_numpy()
                    ),
                }
            )
    frame.to_csv(out / "private_anchor_source_connection.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "scope": "24 fixed source authors per query; exact allocation, not a new confirmatory sample",
            "cost": read(out / "run_status.json"),
            "metrics": metrics,
        },
    )
    print(pd.DataFrame(metrics).to_string(index=False))


def main():
    """Keep diagnosis phased and all private data under ignored results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "evaluate"))
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "source_anchor_diagnosis_v1"
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError("Private output outside results")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    {"prepare": prepare, "run": run, "evaluate": evaluate}[args.phase](out)


if __name__ == "__main__":
    main()
