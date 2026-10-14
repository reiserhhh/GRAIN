#!/usr/bin/env python3
"""Cold raw-text application of a frozen local measurement construction."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.contextual_instrument import score_records, text_key
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts.run_grain_entailment_reference import MODEL, REVISION
from scripts.run_grain_evidence_calibration import infer_pairs
from scripts.run_grain_query_moment_operator import read, sha, write


def run(args):
    """Load raw input plus source bundle only; no simulated truth or cached margins."""
    started = time.monotonic()
    bundle, payload = read(args.bundle), read(args.input)
    if payload["instrument_id"] != bundle["spec"]["id"]:
        raise ValueError("Raw input does not name this instrument")
    if bundle["reader"] != {"model": MODEL, "revision": REVISION}:
        raise ValueError("Pinned observation reader differs")
    raw = payload["records"]
    spec = bundle["spec"]
    unique = {
        text_key(spec["query"], r["text"]): r["text"]
        for r in raw
        if r["scope"] == spec["allowed_scope"] and r["condition"] in (-1, 1)
    }
    pairs = [
        {
            "audit_id": key + ":" + sign,
            "key": key,
            "sign": sign,
            "sentence": text,
            "item_text": spec[sign + "_hypothesis"],
        }
        for key, text in sorted(unique.items())
        for sign in ("positive", "negative")
    ]
    if len(pairs) > args.max_pairs:
        raise ValueError("Cold inference cap exceeded; do not truncate raw input")
    asset = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    assets = [p for p in asset.rglob("*") if p.is_file()]
    args.output.mkdir(parents=True, exist_ok=False)
    outputs = [
        args.output / n
        for n in (
            "predictions.json",
            "evidence.json",
            "reader_results.json",
            "manifest.json",
        )
    ]
    parent = args.bundle.parent.parent
    probes = [parent / "reader_cache.json", parent / "source_anchors.csv"]
    probes += [
        p
        for folder in ("truth", "application", "raw")
        for p in (parent / folder).glob("*")
        if p.is_file() and not p.name.startswith("._")
    ]
    probes = [p for p in probes if p.resolve() != args.input.resolve()]
    denied = install(
        [*assets, args.bundle, args.input],
        outputs,
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )

    def alarm(_sig, _frame):
        raise TimeoutError("Bounded cold application wall cap reached")

    old = signal.signal(signal.SIGALRM, alarm)
    signal.alarm(args.max_seconds)
    try:
        frame = pd.DataFrame(pairs)
        if pairs:
            result = frame[["audit_id", "key", "sign"]].merge(
                infer_pairs(frame), on="audit_id", validate="one_to_one"
            )
            pp = result.pivot(index="key", columns="sign", values="p_entailment")
            margins = (pp.positive - pp.negative).to_dict()
            neural_rows = result.to_dict("records")
        else:
            margins, neural_rows = {}, []
        predictions, evidence = score_records(
            raw, margins, bundle, clock_unit=payload["clock_unit"]
        )
        write(outputs[0], predictions)
        write(outputs[1], evidence)
        write(outputs[2], neural_rows)
        write(
            outputs[3],
            {
                "status": "COMPLETE",
                "input_sha256": sha(args.input),
                "bundle_sha256": sha(args.bundle),
                "files": {p.name: sha(p) for p in outputs[:3]},
                "model": MODEL,
                "revision": REVISION,
                "dtype": "float32",
                "device": "cpu",
                "torch": torch.__version__,
                "pairs": len(pairs),
                "unique_texts": len(unique),
                "authors": len(predictions),
                "wall_seconds": time.monotonic() - started,
                "denial_probes": len(probes),
                "extra_denials": len(denied) - len(probes),
                "cached_margins_read": False,
                "evaluation_truth_read": False,
            },
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    print(
        {
            "status": "COMPLETE",
            "pairs": len(pairs),
            "authors": len(predictions),
            "seconds": time.monotonic() - started,
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=160)
    parser.add_argument("--max-seconds", type=int, default=180)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args)
